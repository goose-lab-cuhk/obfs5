"""Token streams of a dataset split, packed in one array, and batches by token count.

The async form has two streams per flow (client role 0, server role 1); the
sync form has one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class Streams(Dataset):
    def __init__(self, tokens: np.ndarray, lengths: np.ndarray, roles: Optional[np.ndarray]):
        self.tokens = tokens  # (T, 2) int16
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        self.roles = roles  # (n,) int8, or None for the sync form

    def __len__(self) -> int:
        return len(self.lengths)

    def __getitems__(self, indices: Sequence[int]) -> Dict[str, torch.Tensor]:
        """A padded batch: tokens (B, L, 2), mask (B, L) True at real tokens, roles (B,)."""
        idx = np.asarray(indices, dtype=np.int64)
        starts, lengths = self.offsets[idx], self.lengths[idx]
        rows = np.repeat(np.arange(idx.size), lengths)
        cols = np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        tokens = np.zeros((idx.size, int(lengths.max()), 2), dtype=np.int64)
        tokens[rows, cols] = self.tokens[np.repeat(starts, lengths) + cols]
        mask = np.zeros(tokens.shape[:2], dtype=bool)
        mask[rows, cols] = True
        batch = {"tokens": torch.from_numpy(tokens), "mask": torch.from_numpy(mask)}
        if self.roles is not None:
            batch["roles"] = torch.from_numpy(self.roles[idx].astype(np.int64))
        return batch


def load_split(path: Path, form: str, B_t: int, batch_rows: int = 262_144) -> Streams:
    """The streams of one split file (train/val/test.parquet) for ``form``."""
    columns = ["client_tokens", "server_tokens"] if form == "async" else ["sync_tokens"]
    tokens, lengths = [], []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=columns):
        lens = [pc.list_value_length(batch[c]).to_numpy().astype(np.int64) for c in columns]
        ids = [pc.list_flatten(batch[c]).to_numpy().astype(np.int64) for c in columns]
        if form == "async":  # client, then server, of each flow
            n = np.stack(lens, axis=1).reshape(-1)
            starts = np.concatenate([[0], np.cumsum(n)[:-1]])
            flat = np.empty(int(n.sum()), dtype=np.int64)
            for k in range(2):
                pos = np.repeat(starts[k::2], lens[k]) + (
                    np.arange(len(ids[k])) - np.repeat(np.cumsum(lens[k]) - lens[k], lens[k])
                )
                flat[pos] = ids[k]
            lens, ids = [n], [flat]
        lengths.append(lens[0])
        tokens.append(np.stack([ids[0] // (B_t + 1), ids[0] % (B_t + 1)], axis=1).astype(np.int16))
    lengths = np.concatenate(lengths) if lengths else np.zeros(0, dtype=np.int64)
    tokens = np.concatenate(tokens) if tokens else np.zeros((0, 2), dtype=np.int16)
    roles = np.tile(np.array([0, 1], np.int8), len(lengths) // 2) if form == "async" else None
    return Streams(tokens, lengths, roles)


def _cuts(lengths: np.ndarray, max_tokens: int) -> List[int]:
    """Where to cut ``lengths`` (in order) into runs of at most ``max_tokens``
    (a stream above the budget gets a run of its own)."""
    cum = np.concatenate([[0], np.cumsum(lengths)])
    cuts, start = [], 0
    while start < lengths.size:
        end = max(int(np.searchsorted(cum, cum[start] + max_tokens, side="right")) - 1, start + 1)
        cuts.append(end)
        start = end
    return cuts[:-1]


class TokenBatches(Sampler):
    """Batches of at most ``max_tokens`` tokens, grouping streams of similar length
    within random chunks of ``chunk_batches`` batches (all streams when not shuffling)."""

    def __init__(self, lengths, max_tokens: int, shuffle: bool, chunk_batches: int, seed: int):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.max_tokens, self.shuffle = max_tokens, shuffle
        self.chunk_batches, self.seed, self.epoch = chunk_batches, seed, 0
        self._batches = None

    def set_epoch(self, epoch: int) -> None:
        if epoch != self.epoch:
            self.epoch, self._batches = epoch, None

    def batches(self) -> List[np.ndarray]:
        if self._batches is None:
            self._batches = self._build()
        return self._batches

    def _build(self) -> List[np.ndarray]:
        n = self.lengths.size
        if n == 0:
            return []
        if not self.shuffle:
            order = np.argsort(self.lengths, kind="stable")
            return np.split(order, _cuts(self.lengths[order], self.max_tokens))
        rng = np.random.default_rng([self.seed, self.epoch])
        order = rng.permutation(n)
        cum = np.cumsum(self.lengths[order])
        chunk = self.chunk_batches * self.max_tokens
        batches = []
        for piece in np.split(
            order, np.searchsorted(cum, np.arange(chunk, int(cum[-1]), chunk), side="right")
        ):
            if piece.size:
                piece = piece[np.argsort(self.lengths[piece], kind="stable")]
                batches.extend(np.split(piece, _cuts(self.lengths[piece], self.max_tokens)))
        return [batches[i] for i in rng.permutation(len(batches))]

    def __iter__(self):
        for b in self.batches():
            yield b.tolist()

    def __len__(self) -> int:
        return len(self.batches())


def _whole_batch(batch):
    return batch  # built by Streams.__getitems__


def loader(streams: Streams, max_tokens: int, shuffle: bool, cfg_train: dict) -> DataLoader:
    sampler = TokenBatches(
        streams.lengths, max_tokens, shuffle, cfg_train["chunk_batches"], cfg_train["seed"]
    )
    return DataLoader(
        streams,
        batch_sampler=sampler,
        collate_fn=_whole_batch,
        num_workers=cfg_train["loader_workers"],
        pin_memory=cfg_train["device"] == "cuda",
    )
