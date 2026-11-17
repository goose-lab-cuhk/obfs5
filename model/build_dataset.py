"""Build a dataset from Zeek logs.

    python model/build_dataset.py LOGS... -o data/ [--config FILE]

Stage 1 shards the log rows by connection, stage 2 turns every connection that
passes the checks into its three sequences, stage 3 splits the flows, fits the
two tokenizers and writes the dataset. Stages 1 and 2 resume after an
interruption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import config
from representation import tokenizer, zeek_logs
from representation.flows import build_flow, flag_bits
from representation.streams import endpoint_streams, sync_stream
from representation.tokenizer import Tokenizer, window

SPLITS = ("train", "val", "test")
SEQUENCES = ("client", "server", "sync")
TOO_SHORT = "too_short"  # a sequence keeps fewer than min_records records

log = logging.getLogger("obfsgen.build")

_HEAD = [
    ("flow", pa.string()),
    ("start", pa.float64()),
    ("server_name", pa.string()),
    ("rtt", pa.float64()),
    ("rtt_client", pa.float64()),
    ("rtt_server", pa.float64()),
]
STREAM_SCHEMA = pa.schema(  # stage 2's output
    _HEAD
    + [
        (f"{s}_{c}", t)
        for s in SEQUENCES
        for c, t in (
            ("size", pa.list_(pa.int32())),
            ("gap", pa.list_(pa.float64())),
            ("complete", pa.bool_()),
        )
    ]
)
DATASET_SCHEMA = pa.schema(  # train/val/test.parquet
    _HEAD
    + [
        (f"{s}_{c}", t)
        for s in SEQUENCES
        for c, t in (
            ("size", pa.list_(pa.int32())),
            ("gap", pa.list_(pa.float64())),
            ("tokens", pa.list_(pa.uint16())),
        )
    ]
)


def split_hash(flow: str, seed: int) -> float:
    """A uniform number in [0, 1) from the flow id; it decides the split."""
    h = hashlib.blake2b(f"{seed}/{flow}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2.0**64


def split_of(u: float, shares: Sequence[float]) -> str:
    """train when u < shares[0], val when u < shares[0] + shares[1], else test."""
    return SPLITS[0] if u < shares[0] else SPLITS[1] if u < shares[0] + shares[1] else SPLITS[2]


def _run(fn, tasks, workers: int, fresh: bool = True):
    # forkserver: pyarrow's threads make fork unsafe. fresh: one process per
    # task, so its memory is freed when it ends.
    if workers <= 1 or len(tasks) <= 1:
        yield from map(fn, tasks)
        return
    ctx = mp.get_context("forkserver")
    with ctx.Pool(min(workers, len(tasks)), maxtasksperchild=1 if fresh else None) as pool:
        yield from pool.imap_unordered(fn, tasks)


# Stage 1: shard


def shard_logs(log_paths: Sequence[str], work: Path, b: dict) -> int:
    """Stage 1. Returns the number of shards."""
    dirs = zeek_logs.find_log_dirs(log_paths)
    if not dirs:
        raise SystemExit(f"no obf_flow logs found under {list(log_paths)}")
    files = [
        (unit, lg, p)
        for unit, d in dirs
        for lg in zeek_logs.LOGS
        for p in zeek_logs.log_files(d, lg)
    ]
    for _unit, lg, p in files:
        zeek_logs.check_header(p, lg)
    previous = zeek_logs.load_manifests(work)
    if previous:
        n_shards = previous[0]["n_shards"]
    else:
        size = sum(p.stat().st_size * (6 if p.name.endswith(".gz") else 1) for *_, p in files)
        n_shards = max(1, round(size / (b["shard_mb"] << 20)))
    jobs = [zeek_logs.ShardJob(unit, lg, p, work, n_shards) for unit, lg, p in files]
    todo = sorted((j for j in jobs if not j.done()), key=lambda j: -j.path.stat().st_size)
    log.info(
        "stage 1: %d log directories, %d files (%d to do) -> %d shards",
        len(dirs),
        len(files),
        len(todo),
        n_shards,
    )
    for i, _ in enumerate(_run(zeek_logs.shard_file, todo, b["workers"], fresh=False), 1):
        if i % 50 == 0 or i == len(todo):
            log.info("  sharded %d/%d files", i, len(todo))
    return n_shards


# Stage 2: sequences


class _Groups:
    """The rows of a table grouped by uid (each group keeps the table's order)."""

    def __init__(self, table: pa.Table | None, columns: Sequence[str]):
        self.index: Dict[str, Tuple[int, int]] = {}
        self.cols: Dict[str, np.ndarray] = {}
        if table is None or table.num_rows == 0:
            return
        enc = pc.dictionary_encode(table["uid"].cast(pa.large_string()).combine_chunks())
        codes = enc.indices.to_numpy(zero_copy_only=False).astype(np.int64)
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        bounds = np.flatnonzero(np.diff(sorted_codes)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(codes)]])
        names = enc.dictionary.to_pylist()
        self.index = {names[sorted_codes[s]]: (int(s), int(e)) for s, e in zip(starts, ends)}
        for name in columns:
            col = table[name]
            if name == "flags":
                fe = pc.dictionary_encode(pc.fill_null(col, "").combine_chunks())
                bits = np.array([flag_bits(f) for f in fe.dictionary.to_pylist()], dtype=np.int64)
                arr = bits[fe.indices.to_numpy(zero_copy_only=False)]
            elif pa.types.is_uint64(col.type):  # wrapped-around values become negative
                arr = pc.fill_null(col, 0).to_numpy().astype(np.uint64).view(np.int64)
            elif pa.types.is_boolean(col.type):
                arr = pc.fill_null(col, False).to_numpy(zero_copy_only=False).astype(bool)
            else:
                arr = pc.fill_null(col, 0).to_numpy(zero_copy_only=False)
            self.cols[name] = arr[order]

    def get(self, uid: str) -> Dict[str, np.ndarray] | None:
        if uid not in self.index:
            return None
        s, e = self.index[uid]
        return {k: v[s:e] for k, v in self.cols.items()}


def _read(paths: Sequence[Path]) -> pa.Table | None:
    return pa.concat_tables([pq.read_table(p) for p in paths]) if paths else None


def _flow_rows(table: pa.Table) -> List[dict]:
    rows = table.to_pylist()
    for r in rows:  # Zeek writes an empty string as "(empty)"
        for k in ("history", "sni"):
            if r.get(k) == "(empty)":
                r[k] = ""
    return rows


def sequences_of_shard(task) -> dict:
    """Stage 2 for one shard: checked flows → their three sequences, cut to the window."""
    k, pieces, out, b = task
    zeek_logs.limit_arrow_threads()
    cols: Dict[str, list] = {f.name: [] for f in STREAM_SCHEMA}
    drops: Counter = Counter()
    seen = 0
    for unit, by_log in sorted(pieces.items()):
        flows = _read(by_log["flow"])
        if flows is None:
            continue
        small = pc.less_equal(pc.fill_null(flows["n_pkt"], 0), b["max_packets"])
        keep = pc.unique(flows.filter(small)["uid"]).cast(pa.large_string())
        pkt_parts = []
        for p in by_log["pkt"]:  # skip packets of connections that are too long
            t = pq.read_table(p)
            pkt_parts.append(t.filter(pc.is_in(t["uid"].cast(pa.large_string()), value_set=keep)))
        pkts = pa.concat_tables(pkt_parts) if pkt_parts else None
        recs = _read(by_log["rec"])
        rec_total: Counter = Counter()
        if recs is not None:
            vc = pc.value_counts(recs["uid"])
            rec_total.update(
                dict(zip(vc.field("values").to_pylist(), vc.field("counts").to_pylist()))
            )
            appdata = pc.and_(
                pc.and_(pc.fill_null(recs["encrypted"], False), pc.equal(recs["ctype"], 23)),
                pc.invert(pc.fill_null(recs["probable_hs"], False)),
            )
            recs = recs.filter(appdata)
        pkt_groups = _Groups(pkts, ("ts", "is_orig", "flags", "seq", "ack", "len"))
        rec_groups = _Groups(recs, ("is_orig", "off", "length"))
        del pkts, recs

        for row in _flow_rows(flows):
            seen += 1
            uid = row["uid"]
            flow, reason = build_flow(
                row, pkt_groups.get(uid), rec_groups.get(uid), rec_total[uid], b["max_packets"]
            )
            if reason:
                drops[reason] += 1
                continue
            client, server = endpoint_streams(flow)
            seqs = {"client": client, "server": server, "sync": sync_stream(flow)}
            cut = {s: window(g, b["max_records"], b["max_seconds"]) for s, (_, g) in seqs.items()}
            if any(n < b["min_records"] for n, _ in cut.values()):
                drops[TOO_SHORT] += 1
                continue
            for c, v in (
                ("flow", f"{unit}/{uid}"),
                ("start", row["ts"]),
                ("server_name", row.get("sni") or ""),
                ("rtt", flow.rtt),
                ("rtt_client", flow.rtt_client),
                ("rtt_server", flow.rtt_server),
            ):
                cols[c].append(v)
            for s, (sizes, gaps) in seqs.items():
                n, was_cut = cut[s]
                cols[f"{s}_size"].append(sizes[:n])
                cols[f"{s}_gap"].append(gaps[:n])
                cols[f"{s}_complete"].append(flow.closed and not was_cut)

    table = pa.table(cols, schema=STREAM_SCHEMA)
    tmp = out.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, out)
    stats = {"shard": k, "flows_seen": seen, "kept": table.num_rows, "dropped": dict(drops)}
    out.with_suffix(".json").write_text(json.dumps(stats))
    return stats


def build_sequences(work: Path, n_shards: int, b: dict) -> List[Path]:
    """Stage 2. Returns the paths of the shards' sequence files."""
    options = {k: b[k] for k in ("max_packets", "max_records", "max_seconds", "min_records")}
    out_dir = work / "sequences"
    out_dir.mkdir(parents=True, exist_ok=True)
    opt_path = out_dir / "options.json"
    if opt_path.exists() and json.loads(opt_path.read_text()) != options:
        raise SystemExit(f"{work} was built with other settings; use a fresh output directory")
    opt_path.write_text(json.dumps(options))

    manifests = zeek_logs.load_manifests(work)
    paths = [out_dir / f"{k:04d}.parquet" for k in range(n_shards)]
    tasks = [
        (k, zeek_logs.shard_pieces(work, manifests, k), paths[k], b)
        for k in range(n_shards)
        if not paths[k].with_suffix(".json").exists()
    ]
    log.info("stage 2: %d shards (%d to do)", n_shards, len(tasks))
    for i, _ in enumerate(_run(sequences_of_shard, tasks, b["workers"]), 1):
        if i % 20 == 0 or i == len(tasks):
            log.info("  built %d/%d shards", i, len(tasks))
    return paths


# Stage 3: dataset


def _read_sequences(path: Path) -> Tuple[List[str], Dict[str, list]]:
    t = pq.read_table(path)
    return t["flow"].to_pylist(), {c: t[c].to_pylist() for c in t.column_names if c != "flow"}


def _sample_shard(task) -> List[tuple]:
    """Flows whose split hash is below ``limit``: (hash, flow, client, server, sync)."""
    path, limit, seed = task
    flows, cols = _read_sequences(path)
    out = []
    for i, f in enumerate(flows):
        u = split_hash(f, seed)
        if u < limit:
            out.append((u, f, *[(cols[f"{s}_size"][i], cols[f"{s}_gap"][i]) for s in SEQUENCES]))
    return out


def _tokenize_shard(task) -> Tuple[Dict[str, int], Counter, int]:
    """Tokenize one shard and write its rows into a part file per split.

    Also counts, over the shard's training flows, the size of the first record
    of each flow the client opens, and the flows the server opens."""
    path, tok_paths, parts, b = task
    toks = {form: Tokenizer.load(p) for form, p in tok_paths.items()}
    flows, cols = _read_sequences(path)
    rows = {name: {f.name: [] for f in DATASET_SCHEMA} for name in SPLITS}
    openers: Counter = Counter()
    server_first = 0
    for i, f in enumerate(flows):
        split = split_of(split_hash(f, b["split_seed"]), b["split"])
        if split == "train":
            first = cols["sync_size"][i][0]  # signed: > 0 from the client
            if first > 0:
                openers[first] += 1
            else:
                server_first += 1
        out = rows[split]
        out["flow"].append(f)
        for c in ("start", "server_name", "rtt", "rtt_client", "rtt_server"):
            out[c].append(cols[c][i])
        for s in SEQUENCES:
            sizes, gaps = cols[f"{s}_size"][i], cols[f"{s}_gap"][i]
            tok = toks["sync" if s == "sync" else "async"]
            out[f"{s}_size"].append(sizes)
            out[f"{s}_gap"].append(gaps)
            out[f"{s}_tokens"].append(tok.encode(sizes, gaps, cols[f"{s}_complete"][i]))
    counts = {}
    for name, cols_out in rows.items():
        counts[name] = len(cols_out["flow"])
        if counts[name]:
            p = parts / name / path.name
            p.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.table(cols_out, schema=DATASET_SCHEMA), p, compression="zstd")
    return counts, openers, server_first


def _merge(parts: Sequence[Path], out: Path) -> None:
    """Concatenate part files row group by row group."""
    tmp = out.with_suffix(".tmp")
    with pq.ParquetWriter(tmp, DATASET_SCHEMA, compression="zstd") as w:
        for p in parts:
            f = pq.ParquetFile(p)
            for g in range(f.num_row_groups):
                w.write_table(f.read_row_group(g))
    os.replace(tmp, out)


def write_dataset(paths: Sequence[Path], work: Path, out: Path, cfg: dict) -> dict:
    """Stage 3: split, fit both tokenizers, tokenize, write train/val/test.parquet
    and opener.json (the first record's size over the training flows)."""
    b = cfg["build"]
    stats = [json.loads(p.with_suffix(".json").read_text()) for p in paths]
    n_flows = sum(s["kept"] for s in stats)
    if n_flows == 0:
        raise SystemExit("no flow passed the checks; see the drop counts in the log")
    # fit on the training flows with the smallest hash: collect a few more, then sort
    limit = min(b["split"][0], 1.1 * b["fit_flows"] / n_flows + 1e-4)
    tasks = [(p, limit, b["split_seed"]) for p in paths]
    sample = sorted((r for rows in _run(_sample_shard, tasks, b["workers"]) for r in rows))
    sample = sample[: b["fit_flows"]]
    log.info("stage 3: %d flows; fitting the tokenizers on %d training flows", n_flows, len(sample))

    toks = {
        "async": tokenizer.fit("async", (seq for r in sample for seq in (r[2], r[3])), cfg),
        "sync": tokenizer.fit("sync", (r[4] for r in sample), cfg),
    }
    tok_paths = {}
    for form, tok in toks.items():
        tok_paths[form] = out / f"tokenizer_{form}.json"
        tok.save(tok_paths[form])
        log.info(
            "  %s tokenizer: B_s %d, B_t %d, timer peaks %s s",
            form,
            tok.B_s,
            tok.B_t,
            [round(p, 3) for p in tok.time_peaks],
        )
    parts = work / "parts"
    shutil.rmtree(parts, ignore_errors=True)
    per_split: Counter = Counter()
    openers: Counter = Counter()
    server_first = 0
    for c, o, n in _run(_tokenize_shard, [(p, tok_paths, parts, b) for p in paths], b["workers"]):
        per_split.update(c)
        openers.update(o)
        server_first += n
    for name in SPLITS:
        _merge(sorted((parts / name).glob("*.parquet")), out / f"{name}.parquet")
    log.info("  wrote %s", dict(per_split))
    # The endpoint draws each connection's first record from this distribution.
    sizes = sorted(openers)
    opener = {"sizes": sizes, "counts": [openers[s] for s in sizes], "server_first": server_first}
    (out / "opener.json").write_text(json.dumps({"format": "obfsgen-opener-v1", **opener}))
    log.info(
        "  opener.json: %d training flows the client opens (%d sizes), %d the server opens",
        sum(openers.values()),
        len(sizes),
        server_first,
    )
    dropped: Counter = Counter()
    for s in stats:
        dropped.update(s["dropped"])
    return {
        "flows_seen": sum(s["flows_seen"] for s in stats),
        "dropped": dict(dropped),
        "flows": n_flows,
        "splits": dict(per_split),
        "fit_flows": len(sample),
        "opener_flows": sum(openers.values()),
        "server_first": server_first,
        "async": {"B_s": toks["async"].B_s, "B_t": toks["async"].B_t},
        "sync": {"B_s": toks["sync"].B_s, "B_t": toks["sync"].B_t},
    }


def run(cfg: dict, log_paths: Sequence[str | Path], out: Path) -> dict:
    """The whole build; returns what it writes to build.json."""
    out = Path(out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_shards = shard_logs(log_paths, work, cfg["build"])
    paths = build_sequences(work, n_shards, cfg["build"])
    report = write_dataset(paths, work, out, cfg)
    report = {
        "logs": [str(p) for p in log_paths],
        "shards": n_shards,
        **report,
        "seconds": round(time.time() - t0, 1),
        "config": {k: cfg[k] for k in ("build", "tokenizer")},
    }
    (out / "build.json").write_text(json.dumps(report, indent=1))
    log.info(
        "kept %d of %d flows; dropped: %s", report["flows"], report["flows_seen"], report["dropped"]
    )
    if not cfg["build"]["keep_work"]:
        shutil.rmtree(work)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build a dataset from Zeek logs.")
    ap.add_argument("logs", nargs="+", help="directories holding obf_* logs (searched recursively)")
    ap.add_argument("-o", "--out", required=True, type=Path, help="output directory")
    config.add_argument(ap)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run(config.load(args.config), args.logs, args.out)


if __name__ == "__main__":
    main()
