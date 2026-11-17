"""Records (signed size, gap) to tokens (size bin, time bin) and back.

Size value = bytes / S_norm, time value = ln(gap + epsilon), both rounded to
float32; a value's bin is the number of interior edges <= it. Size bins are
fitted separately for sent and received records, with a bin of its own for
each frequent size; time bins follow a log grid above 1 ms, plus a narrow bin
at each timer peak. [EOF] is (B_s, B_t); a dataset stores the token id
size_bin * (B_t + 1) + time_bin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

FORMAT = "obfsgen-tokenizer-v1"
_PEAK_CELLS = 12  # cells per grid bin in the peak search
_PEAK_MIN_COUNT = 20  # a peak cell holds at least this many gaps


def window(gaps: np.ndarray, max_records: int, max_seconds: float) -> Tuple[int, bool]:
    """(records kept, whether the window cut the sequence). At most max_records
    records, stopping before the cumulative gap passes max_seconds; a sequence of
    max_records or more records, or max_seconds or more of gaps, counts as cut."""
    n = len(gaps)
    if n == 0:
        return 0, False
    cum = np.cumsum(np.asarray(gaps[1:], dtype=np.float64))
    total = float(cum[-1]) if cum.size else 0.0
    cut = n >= max_records or total >= max_seconds
    over = np.flatnonzero(cum > max_seconds)
    kept = min(n, max_records, 1 + int(over[0]) if over.size else n)
    return kept, cut


def size_values(sizes: Sequence[int], s_norm: int) -> np.ndarray:
    return (np.asarray(sizes, dtype=np.float64) / s_norm).astype(np.float32)


def time_values(gaps: Sequence[float], epsilon: float) -> np.ndarray:
    return np.log(np.asarray(gaps, dtype=np.float64) + epsilon).astype(np.float32)


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """The number of interior edges ≤ each value (outside values fall in the end bins)."""
    return np.searchsorted(edges[1:-1], values, side="right")


@dataclass
class Tokenizer:
    """A fitted tokenizer; ``fit`` makes one, ``load`` reads one back."""

    form: str  # "async" or "sync"
    size_edges: np.ndarray  # B_s + 1 edges on the size value
    time_edges: np.ndarray  # B_t + 1 edges on the time value
    size_intra: List[Tuple[np.ndarray, np.ndarray]]  # per size bin: values, probs
    time_intra: List[np.ndarray]  # per time bin: a sample of values
    time_bin_kinds: List[str]  # floor / log / peak
    time_peaks: List[float]  # peak centres, seconds
    S_norm: int
    epsilon: float
    max_records: int
    max_time_seconds: float
    min_records: int
    time_floor: float
    time_bins_per_decade: int

    @property
    def B_s(self) -> int:
        return len(self.size_edges) - 1

    @property
    def B_t(self) -> int:
        return len(self.time_edges) - 1

    @property
    def eof(self) -> int:
        """The token id of [EOF]."""
        return self.B_s * (self.B_t + 1) + self.B_t

    def encode(self, sizes: Sequence[int], gaps: Sequence[float], complete: bool) -> np.ndarray:
        """Token ids of one sequence (already cut to the window), [EOF] last if complete."""
        s = assign_bins(size_values(sizes, self.S_norm), self.size_edges)
        t = assign_bins(time_values(gaps, self.epsilon), self.time_edges)
        ids = s * (self.B_t + 1) + t
        if complete:
            ids = np.append(ids, self.eof)
        return ids.astype(np.uint16)

    def split_ids(self, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Token ids → (size bins, time bins)."""
        ids = np.asarray(ids, dtype=np.int64)
        return ids // (self.B_t + 1), ids % (self.B_t + 1)

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "form": self.form,
            "B_s": self.B_s,
            "B_t": self.B_t,
            "S_norm": self.S_norm,
            "epsilon": self.epsilon,
            "max_records": self.max_records,
            "max_time_seconds": self.max_time_seconds,
            "min_records": self.min_records,
            "size_edges": [float(v) for v in self.size_edges],
            "time_edges": [float(v) for v in self.time_edges],
            "size_bin_direction": [1 if e >= 0 else -1 for e in self.size_edges[:-1]],
            "size_intra": [
                {"values": [float(v) for v in vals], "probs": [float(p) for p in probs]}
                for vals, probs in self.size_intra
            ],
            "time_intra": [[float(v) for v in vals] for vals in self.time_intra],
            "time_floor": self.time_floor,
            "time_bins_per_decade": self.time_bins_per_decade,
            "time_peaks": list(self.time_peaks),
            "time_bin_kinds": list(self.time_bin_kinds),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Tokenizer":
        if d.get("format") != FORMAT:
            raise ValueError(f"not an {FORMAT} tokenizer: format = {d.get('format')!r}")
        return cls(
            form=d["form"],
            size_edges=np.asarray(d["size_edges"], dtype=np.float64),
            time_edges=np.asarray(d["time_edges"], dtype=np.float64),
            size_intra=[
                (np.asarray(e["values"], np.float32), np.asarray(e["probs"], np.float32))
                for e in d["size_intra"]
            ],
            time_intra=[np.asarray(v, dtype=np.float32) for v in d["time_intra"]],
            time_bin_kinds=list(d["time_bin_kinds"]),
            time_peaks=list(d["time_peaks"]),
            S_norm=d["S_norm"],
            epsilon=d["epsilon"],
            max_records=d["max_records"],
            max_time_seconds=d["max_time_seconds"],
            min_records=d["min_records"],
            time_floor=d["time_floor"],
            time_bins_per_decade=d["time_bins_per_decade"],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit(
    form: str, sequences: Iterable[Tuple[Sequence[int], Sequence[float]]], cfg: dict
) -> Tokenizer:
    """Fit a tokenizer on training sequences of (signed sizes, gaps), already cut
    to the window."""
    t, b = cfg["tokenizer"], cfg["build"]
    size_parts, time_parts = [], []
    for sizes, gaps in sequences:
        size_parts.append(size_values(sizes, t["s_norm"]))
        time_parts.append(time_values(gaps, t["epsilon"]))
    if not size_parts:
        raise ValueError("no training sequences to fit the tokenizer on")
    size_dim = np.concatenate(size_parts).astype(np.float64)
    time_dim = np.concatenate(time_parts).astype(np.float64)

    size_edges = _split_size_edges(size_dim, t["size_bins"], t["atom_share"])
    time_edges, kinds, peaks = _time_edges(time_dim, t, b["max_seconds"])

    size_bins_of = assign_bins(size_dim, size_edges)
    size_intra = []
    for k in range(len(size_edges) - 1):
        vals = size_dim[size_bins_of == k]
        if vals.size == 0:
            mid = 0.5 * (size_edges[k] + size_edges[k + 1])
            size_intra.append((np.array([mid], np.float32), np.array([1.0], np.float32)))
            continue
        unique, counts = np.unique(vals.astype(np.float32), return_counts=True)
        size_intra.append((unique, (counts / counts.sum()).astype(np.float32)))

    rng = np.random.default_rng(t["seed"])
    time_bins_of = assign_bins(time_dim, time_edges)
    time_intra = []
    for k in range(len(time_edges) - 1):
        vals = time_dim[time_bins_of == k]
        if vals.size == 0:
            time_intra.append(np.array([0.5 * (time_edges[k] + time_edges[k + 1])], np.float32))
            continue
        if vals.size > t["intra_samples"]:
            vals = rng.choice(vals, size=t["intra_samples"], replace=False)
        time_intra.append(np.sort(vals).astype(np.float32))

    return Tokenizer(
        form=form,
        size_edges=size_edges,
        time_edges=time_edges,
        size_intra=size_intra,
        time_intra=time_intra,
        time_bin_kinds=kinds,
        time_peaks=peaks,
        S_norm=t["s_norm"],
        epsilon=t["epsilon"],
        max_records=b["max_records"],
        max_time_seconds=b["max_seconds"],
        min_records=b["min_records"],
        time_floor=t["time_floor"],
        time_bins_per_decade=t["bins_per_decade"],
    )


def _quantile_edges(values: np.ndarray, n_bins: int, atom_share: float) -> np.ndarray:
    """Equal-frequency edges, plus a bin of its own for each value holding at least
    atom_share of the data."""
    n = values.size
    if n == 0:
        return np.array([-1e-9, 1e-9])
    uniques, counts = np.unique(values, return_counts=True)
    candidates = np.flatnonzero(counts / n >= atom_share)
    max_atoms = max(0, n_bins - 2)
    if candidates.size == 0 or max_atoms == 0:
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 2:
            v = float(edges[0]) if edges.size == 1 else 0.0
            return np.array([v - 1e-9, v + 1e-9])
        return edges

    order = np.lexsort((uniques[candidates], -counts[candidates]))
    atoms = np.sort(uniques[candidates[order][:max_atoms]])
    brackets = []
    for a in atoms:
        i = np.searchsorted(uniques, a)
        prev = float(uniques[i - 1]) if i > 0 else a - 1.0
        nxt = float(uniques[i + 1]) if i + 1 < uniques.size else a + 1.0
        brackets.append(
            (float(a - max((a - prev) / 2.0, 1e-12)), float(a + max((nxt - a) / 2.0, 1e-12)))
        )

    rest = values[~np.isin(values, atoms)]
    n_rest_bins = n_bins - atoms.size
    seg_lo = [-np.inf] + [hi for _, hi in brackets]
    seg_hi = [lo for lo, _ in brackets] + [np.inf]
    segments = [rest[(rest > lo) & (rest < hi)] for lo, hi in zip(seg_lo, seg_hi)]
    seg_sizes = np.array([s.size for s in segments])
    total = int(seg_sizes.sum())
    if total > 0:
        per_seg = np.floor(seg_sizes / total * n_rest_bins).astype(int)
        leftover = n_rest_bins - int(per_seg.sum())
        if leftover > 0:
            score = seg_sizes / np.maximum(per_seg, 1)
            for i in np.argsort(-score)[:leftover]:
                per_seg[int(i)] += 1
        for i, s in enumerate(segments):  # a non-empty segment gets at least one bin
            if s.size > 0 and per_seg[i] == 0:
                j = int(np.argmax(per_seg))
                if per_seg[j] > 1:
                    per_seg[j] -= 1
                    per_seg[i] += 1
    else:
        per_seg = np.zeros(len(segments), dtype=int)

    pieces = []
    for i, (seg, k, lo, hi) in enumerate(zip(segments, per_seg, seg_lo, seg_hi)):
        if k > 0:
            if seg.size > 0:
                e = np.quantile(seg, np.linspace(0.0, 1.0, k + 1))
                if np.isfinite(lo):
                    e[0] = lo
                if np.isfinite(hi):
                    e[-1] = hi
                e = np.unique(e)
            else:
                lo = hi - 1.0 if not np.isfinite(lo) else lo
                hi = lo + 1.0 if not np.isfinite(hi) else hi
                e = np.linspace(lo, hi, k + 1)
            pieces.append(e)
        if i < atoms.size:
            pieces.append(np.array(brackets[i]))
    edges = np.unique(np.concatenate(pieces))
    if edges.size < 2:
        v = float(edges[0]) if edges.size == 1 else 0.0
        edges = np.array([v - 1e-9, v + 1e-9])
    return edges


def _split_size_edges(values: np.ndarray, n_bins: int, atom_share: float) -> np.ndarray:
    """Size edges fitted separately for received (< 0) and sent (>= 0) records."""
    neg, pos = values[values < 0], values[values >= 0]
    if neg.size == 0 and pos.size == 0:
        return np.array([-1e-9, 0.0, 1e-9])
    if neg.size == 0:
        edges = _quantile_edges(pos, n_bins, atom_share)
        edges[0] = 0.0
    elif pos.size == 0:
        edges = _quantile_edges(neg, n_bins, atom_share)
        edges[-1] = 0.0
    else:
        share_neg = neg.size / values.size
        n_neg = int(np.clip(round(n_bins * share_neg), 1, max(1, n_bins - 1)))
        e_neg = _quantile_edges(neg, n_neg, atom_share)
        e_pos = _quantile_edges(pos, max(1, n_bins - n_neg), atom_share)
        e_neg[-1] = 0.0
        e_pos[0] = 0.0
        edges = np.concatenate([e_neg, e_pos[1:]])
    return np.unique(edges)


def _time_edges(
    time_dim: np.ndarray, t: dict, max_seconds: float
) -> Tuple[np.ndarray, List[str], List[float]]:
    """Time edges: the floor bin, the log grid and the peak bins; plus kinds and peaks."""
    eps, floor, per_decade, halfwidth = (
        t["epsilon"],
        t["time_floor"],
        t["bins_per_decade"],
        t["peak_halfwidth"],
    )
    lo_outer = float(np.log(eps)) - 1.0
    floor_u = float(np.log(floor + eps))
    ceil_u = float(np.log(max_seconds + eps))
    n_grid = int(np.ceil(per_decade * np.log10(max_seconds / floor) - 1e-9))
    grid_s = floor * 10.0 ** (np.arange(1, n_grid) / per_decade)
    grid_s = grid_s[grid_s < max_seconds * (1 - 1e-9)]
    grid = np.concatenate([[floor_u], np.log(grid_s + eps), [ceil_u]])

    peaks = _find_peaks(time_dim[time_dim >= floor_u], grid, t)
    merged: List[Tuple[float, float]] = []  # peaks whose bins would touch become one
    for centre, share in sorted(peaks):
        if merged and centre / (1 + halfwidth) <= merged[-1][0] * (1 + halfwidth):
            c0, s0 = merged[-1]
            merged[-1] = ((c0 * s0 + centre * share) / (s0 + share), s0 + share)
        else:
            merged.append((centre, share))
    peaks = sorted(merged, key=lambda p: -p[1])[: t["max_peaks"]]

    brackets: List[Tuple[float, float, float]] = []
    for centre, _share in peaks:
        lo = float(np.log(centre / (1 + halfwidth) + eps))
        hi = float(np.log(centre * (1 + halfwidth) + eps))
        if lo <= floor_u or hi >= ceil_u:
            continue
        if any(lo < b_hi and hi > b_lo for b_lo, b_hi, _ in brackets):
            continue  # overlaps a heavier peak
        brackets.append((lo, hi, centre))

    # A grid edge inside a peak bin would split it in two; the peak bin wins.
    keep = np.array([not any(lo < g < hi for lo, hi, _ in brackets) for g in grid])
    edges = np.unique(
        np.concatenate([[lo_outer], grid[keep]] + [[lo, hi] for lo, hi, _ in brackets])
    )

    kinds = []
    for k in range(edges.size - 1):
        mid = 0.5 * (edges[k] + edges[k + 1])
        if mid < floor_u:
            kinds.append("floor")
        elif any(lo <= mid <= hi for lo, hi, _ in brackets):
            kinds.append("peak")
        else:
            kinds.append("log")
    return edges, kinds, [c for _, _, c in brackets]


def _find_peaks(u: np.ndarray, grid: np.ndarray, t: dict) -> List[Tuple[float, float]]:
    """Timer peaks above the floor as (centre in seconds, share), heaviest first:
    runs of cells holding peak_ratio times their share of the grid bin."""
    n = u.size
    if n == 0 or grid.size < 2:
        return []
    cell_edges = np.concatenate(
        [
            np.linspace(grid[i], grid[i + 1], _PEAK_CELLS, endpoint=False)
            for i in range(grid.size - 1)
        ]
        + [grid[-1:]]
    )
    counts, _ = np.histogram(u, bins=cell_edges)
    bin_mass = counts.reshape(grid.size - 1, _PEAK_CELLS).sum(axis=1)
    uniform = np.repeat(bin_mass / _PEAK_CELLS, _PEAK_CELLS)
    candidate = (
        (counts >= t["peak_ratio"] * np.maximum(uniform, 1e-12))
        & (counts >= t["peak_min_share"] * n)
        & (counts >= _PEAK_MIN_COUNT)
    )
    peaks = []
    i = 0
    while i < candidate.size:
        if not candidate[i]:
            i += 1
            continue
        j = i
        while j + 1 < candidate.size and candidate[j + 1]:
            j += 1
        sel = (u >= cell_edges[i]) & (u < cell_edges[j + 1])
        if sel.any():
            peaks.append((float(np.exp(u[sel].mean()) - t["epsilon"]), float(sel.sum()) / n))
        i = j + 1
    peaks.sort(key=lambda p: -p[1])
    return peaks[: t["max_peaks"]]
