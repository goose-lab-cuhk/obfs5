"""Score flows: the mean NLL of each flow's scored tokens (lower = more normal).

    python model/score.py MODEL INPUT... -o scores.csv [--config FILE]

MODEL is a folder holding model.pt, such as a run of train.py or
params/example/campus-gateway/async.

INPUT is dataset .parquet files or Zeek log directories. Scored tokens: each
endpoint's own sends and [EOF] (async model), every token (sync model).
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

import build_dataset
import config
from representation.tokenizer import Tokenizer
from seqmodel import data
from seqmodel.model import load

log = logging.getLogger("obfsgen.score")


def read_flows(inputs: Sequence[Path], tok: Tokenizer, cfg: dict) -> Tuple[List[str], dict]:
    """(flow ids, columns) of the input: per sequence its sizes, gaps and whether it is complete."""
    inputs = [Path(p) for p in inputs]
    if all(p.suffix == ".parquet" for p in inputs):
        names = ["flow", "server_name"] + [
            f"{s}_{c}" for s in build_dataset.SEQUENCES for c in ("size", "gap", "tokens")
        ]
        tables = [pq.read_table(p, columns=names).to_pydict() for p in inputs]
        cols = {k: sum((t[k] for t in tables), []) for k in tables[0]}
        for s in build_dataset.SEQUENCES:  # a complete sequence has one token more than records
            cols[f"{s}_complete"] = [
                len(t) == len(z) + 1 for t, z in zip(cols[f"{s}_tokens"], cols[f"{s}_size"])
            ]
        return cols.pop("flow"), cols
    if any(p.suffix == ".parquet" for p in inputs):
        raise SystemExit("give either dataset files or log directories, not both")
    # Logs: the build's first two stages, with the run's window.
    b = {
        **cfg["build"],
        "max_records": tok.max_records,
        "max_seconds": tok.max_time_seconds,
        "min_records": tok.min_records,
    }
    work = Path(tempfile.mkdtemp(prefix="obfsgen-score-"))
    try:
        paths = build_dataset.build_sequences(work, build_dataset.shard_logs(inputs, work, b), b)
        tables = [pq.read_table(p).to_pydict() for p in paths]
    finally:
        shutil.rmtree(work)
    cols = {k: sum((t[k] for t in tables), []) for k in tables[0]}
    return cols.pop("flow"), cols


def score(run: Path, inputs: Sequence[Path], out: Path, cfg: dict) -> np.ndarray:
    """Write ``out`` (one row per flow); return the scores."""
    device = torch.device(cfg["score"]["device"])
    model, ckpt = load(Path(run) / "model.pt", device)
    tok = Tokenizer.from_dict(ckpt["tokenizer"])
    form = ckpt["form"]
    flows, cols = read_flows(inputs, tok, cfg)
    names = ["client", "server"] if form == "async" else ["sync"]
    send = np.append(tok.size_edges[:-1] >= 0, True)  # per size bin, [EOF] last

    # One stream per flow (sync) or two (async), and the flow each belongs to.
    tokens, lengths, roles, owner, scored = [], [], [], [], []
    for i in range(len(flows)):
        for role, name in enumerate(names):
            ids = tok.encode(
                cols[f"{name}_size"][i], cols[f"{name}_gap"][i], cols[f"{name}_complete"][i]
            )
            s, t = tok.split_ids(ids)
            tokens.append(np.stack([s, t], 1).astype(np.int16))
            lengths.append(len(ids))
            roles.append(role)
            owner.append(i)
            scored.append(send[s[1:]] if form == "async" else np.ones(len(ids) - 1, bool))
    streams = data.Streams(
        np.concatenate(tokens),
        np.array(lengths),
        np.array(roles, np.int8) if form == "async" else None,
    )

    sums = np.zeros((len(flows), 3))  # size NLL, time NLL, scored tokens
    batches = data.TokenBatches(streams.lengths, cfg["score"]["batch_tokens"], False, 1, 0)
    with torch.no_grad():
        for idx in batches.batches():
            b = streams.__getitems__(idx)
            tok_b = b["tokens"].to(device)
            r = b.get("roles")
            size_logits, time_logits = model(tok_b, None if r is None else r.to(device))
            nll_s = F.cross_entropy(size_logits.transpose(1, 2), tok_b[:, 1:, 0], reduction="none")
            nll_t = F.cross_entropy(time_logits.transpose(1, 2), tok_b[:, 1:, 1], reduction="none")
            nll_s, nll_t = nll_s.cpu().numpy(), nll_t.cpu().numpy()
            for row, j in enumerate(idx):
                keep = scored[j]
                n = len(keep)
                sums[owner[j]] += [
                    nll_s[row, :n][keep].sum(),
                    nll_t[row, :n][keep].sum(),
                    keep.sum(),
                ]

    n = np.maximum(sums[:, 2], 1)
    nll = (sums[:, 0] + sums[:, 1]) / n
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["flow", "server_name", "records", "scored", "nll", "nll_size", "nll_time"])
        for i, f in enumerate(flows):
            w.writerow(
                [
                    f,
                    cols["server_name"][i],
                    len(cols[f"{names[0]}_size"][i]),
                    int(sums[i, 2]),
                    f"{nll[i]:.6f}",
                    f"{sums[i, 0] / n[i]:.6f}",
                    f"{sums[i, 1] / n[i]:.6f}",
                ]
            )
    log.info(
        "%d flows scored with the %s model: median NLL %.4f -> %s",
        len(flows),
        form,
        float(np.median(nll)) if len(nll) else float("nan"),
        out,
    )
    return nll


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Score flows with a trained model.")
    ap.add_argument(
        "run", type=Path, help="a folder holding model.pt (a run, or params/NAME/<form>)"
    )
    ap.add_argument(
        "inputs", nargs="+", type=Path, help="dataset .parquet files or Zeek log directories"
    )
    ap.add_argument("-o", "--out", required=True, type=Path, help="CSV file, one row per flow")
    config.add_argument(ap)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    score(args.run, args.inputs, args.out, config.load(args.config))


if __name__ == "__main__":
    main()
