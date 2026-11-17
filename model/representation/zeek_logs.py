"""Find and read the logs written by capture/obfsgen.zeek, and shard their rows by uid.

Sharding by uid puts all rows of a connection in one shard, even when log
rotation spread them over several files.
"""

from __future__ import annotations

import gzip
import json
import os
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

LOGS = ("flow", "rec", "pkt")

# The columns the build reads from each log; any others are ignored.
COLUMNS: Dict[str, Tuple[str, ...]] = {
    "flow": (
        "ts",
        "uid",
        "proto",
        "history",
        "t_syn",
        "t_synack",
        "t_ack",
        "sni",
        "tls_estab",
        "missed",
        "n_pkt",
        "n_rec",
    ),
    "rec": ("uid", "is_orig", "ctype", "length", "off", "encrypted", "probable_hs"),
    "pkt": ("ts", "uid", "is_orig", "flags", "seq", "ack", "len"),
}

_ARROW_TYPES = {
    "time": pa.float64(),
    "interval": pa.float64(),
    "double": pa.float64(),
    "count": pa.uint64(),  # relative seq/ack can wrap around
    "int": pa.int64(),
    "port": pa.int64(),
    "bool": pa.bool_(),
}


class LogError(RuntimeError):
    pass


def log_of(name: str) -> str | None:
    """Which log a file name belongs to: ``obf_<log>*.log`` or ``*.log.gz``."""
    if name.endswith(".log") or name.endswith(".log.gz"):
        for log in LOGS:
            if name.startswith(f"obf_{log}"):
                return log
    return None


def find_log_dirs(paths: Sequence[str | Path]) -> List[Tuple[str, Path]]:
    """(name, path) of every directory under ``paths`` holding an obf_flow log;
    the name is the path relative to the argument. Skips dot and symlinked dirs."""
    found: Dict[str, Path] = {}
    for root in map(Path, paths):
        if not root.is_dir():
            raise LogError(f"{root}: not a directory")
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            if not any(log_of(f) == "flow" for f in filenames):
                continue
            rel = Path(dirpath).relative_to(root)
            name = root.resolve().name if rel == Path(".") else rel.as_posix()
            if name in found:
                raise LogError(
                    f"two log directories are both named {name!r}: " f"{found[name]} and {dirpath}"
                )
            found[name] = Path(dirpath)
    return sorted(found.items())


def log_files(directory: Path, log: str) -> List[Path]:
    """The files of one log in a directory, in time order (rotated names sort by time)."""
    return sorted(p for p in Path(directory).iterdir() if p.is_file() and log_of(p.name) == log)


@dataclass
class Header:
    n_lines: int
    separator: str = "\t"
    unset: str = "-"
    fields: Tuple[str, ...] = ()
    types: Tuple[str, ...] = ()


def read_header(path: Path) -> Header:
    """The ``#`` lines at the top of a Zeek ASCII log."""
    opener = gzip.open if str(path).endswith(".gz") else open
    hdr = Header(n_lines=0)
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            hdr.n_lines += 1
            line = line.rstrip("\n")
            if line.startswith("#separator "):
                hdr.separator = line.split(" ", 1)[1].encode().decode("unicode_escape")
                continue
            tag, *rest = line[1:].split(hdr.separator)
            if tag == "unset_field":
                hdr.unset = rest[0]
            elif tag == "fields":
                hdr.fields = tuple(rest)
            elif tag == "types":
                hdr.types = tuple(rest)
    return hdr


def check_header(path: Path, log: str) -> Header:
    hdr = read_header(path)
    missing = [c for c in COLUMNS[log] if c not in hdr.fields]
    if not hdr.fields:
        raise LogError(f"{path}: no #fields line (empty or truncated log?)")
    if missing:
        raise LogError(
            f"{path}: lacks the columns {missing}; was it written by capture/obfsgen.zeek?"
        )
    return hdr


def iter_batches(path: Path, log: str, block_size: int = 32 << 20) -> Iterator[pa.RecordBatch]:
    """Stream the needed columns of a log as Arrow batches (``-`` becomes null)."""
    hdr = check_header(path, log)
    types = {f: _ARROW_TYPES.get(t, pa.string()) for f, t in zip(hdr.fields, hdr.types)}
    with pcsv.open_csv(
        str(path),
        read_options=pcsv.ReadOptions(
            skip_rows=hdr.n_lines, column_names=list(hdr.fields), block_size=block_size
        ),
        parse_options=pcsv.ParseOptions(
            delimiter=hdr.separator,
            quote_char=False,
            escape_char=False,
            invalid_row_handler=lambda _row: "skip",
        ),  # #close
        convert_options=pcsv.ConvertOptions(
            column_types=types,
            null_values=[hdr.unset],
            true_values=["T"],
            false_values=["F"],
            strings_can_be_null=True,
            include_columns=list(COLUMNS[log]),
        ),
    ) as reader:
        for batch in reader:
            if batch.num_rows:
                yield batch


# Stage 1 of the build


def limit_arrow_threads(threads: int = 4) -> None:
    """pyarrow starts a thread (and its buffers) per core unless limited."""
    pa.set_cpu_count(threads)
    pa.set_io_thread_count(min(2, threads))
    try:
        pa.jemalloc_set_decay_ms(0)
    except (AttributeError, NotImplementedError):
        pass


def shard_of(uids: pa.Array, n_shards: int) -> np.ndarray:
    """crc32(uid) mod n_shards for every row (hashing each distinct uid once)."""
    enc = pc.dictionary_encode(uids)
    if isinstance(enc, pa.ChunkedArray):
        enc = enc.combine_chunks()
    per_uid = np.array(
        [zlib.crc32(u.encode()) % n_shards for u in enc.dictionary.to_pylist()], dtype=np.int64
    )
    return per_uid[enc.indices.to_numpy(zero_copy_only=False)]


@dataclass(frozen=True)
class ShardJob:
    unit: str  # the log directory's name
    log: str
    path: Path
    work: Path
    n_shards: int

    @property
    def job_id(self) -> str:
        return f"{self.unit}/{self.path.name}".replace("/", "__")

    @property
    def manifest(self) -> Path:
        return self.work / "shards" / "_jobs" / f"{self.job_id}.json"

    def piece(self, k: int) -> Path:
        return self.work / "shards" / self.log / f"{k:04d}" / f"{self.job_id}.parquet"

    def done(self) -> bool:
        try:
            m = json.loads(self.manifest.read_text())
        except (OSError, ValueError):
            return False
        return (
            m["n_shards"] == self.n_shards
            and m["source"] == str(self.path)
            and m["source_bytes"] == self.path.stat().st_size
        )


def shard_file(job: ShardJob) -> dict:
    """Scatter one log file into its shard pieces; rows keep their file order."""
    limit_arrow_threads()
    for k in range(job.n_shards):
        job.piece(k).unlink(missing_ok=True)
    writers: Dict[int, pq.ParquetWriter] = {}
    counts = np.zeros(job.n_shards, dtype=np.int64)
    try:
        for batch in iter_batches(job.path, job.log):
            shards = shard_of(batch.column("uid"), job.n_shards)
            order = np.argsort(shards, kind="stable")
            batch = batch.take(pa.array(order))
            n = np.bincount(shards, minlength=job.n_shards)
            starts = np.concatenate([[0], np.cumsum(n)])
            for k in np.flatnonzero(n):
                k = int(k)
                if k not in writers:
                    job.piece(k).parent.mkdir(parents=True, exist_ok=True)
                    writers[k] = pq.ParquetWriter(job.piece(k), batch.schema, compression="zstd")
                writers[k].write_batch(batch.slice(int(starts[k]), int(n[k])))
            counts += n
    finally:
        for w in writers.values():
            w.close()
    m = {
        "unit": job.unit,
        "log": job.log,
        "source": str(job.path),
        "source_bytes": job.path.stat().st_size,
        "n_shards": job.n_shards,
        "rows_per_shard": counts.tolist(),
    }
    job.manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp = job.manifest.with_suffix(".tmp")
    tmp.write_text(json.dumps(m))
    os.replace(tmp, job.manifest)
    return m


def load_manifests(work: Path) -> List[dict]:
    jobs = Path(work) / "shards" / "_jobs"
    return [json.loads(p.read_text()) for p in sorted(jobs.glob("*.json"))] if jobs.exists() else []


def shard_pieces(work: Path, manifests: Sequence[dict], k: int) -> Dict[str, Dict[str, List[Path]]]:
    """Shard ``k``'s pieces: unit → log → files in time order."""
    out: Dict[str, Dict[str, List[Path]]] = {}
    for m in sorted(manifests, key=lambda m: (m["unit"], Path(m["source"]).name)):
        if m["rows_per_shard"][k] == 0:
            continue
        job = ShardJob(m["unit"], m["log"], Path(m["source"]), Path(work), m["n_shards"])
        out.setdefault(m["unit"], {log: [] for log in LOGS})[m["log"]].append(job.piece(k))
    return out
