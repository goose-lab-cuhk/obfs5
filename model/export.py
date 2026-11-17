"""Export the trained models into a parameter folder, which the endpoint reads.

    python model/export.py --async runs/async --sync runs/sync --data data/ -o params/NAME/

Writes <form>/weights.obfw (int8 weights for the endpoint's engine) and
<form>/tokenizer.json of each model, copies its <form>/model.pt (the PyTorch
checkpoint, for score.py and further training), and copies data/opener.json.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import struct
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

import config
from representation.tokenizer import Tokenizer
from seqmodel.model import Model, load

log = logging.getLogger("obfsgen.export")
ABI = 3
WTYPE = {"fp32": 0, "int8": 2}
# The integer fields and the buffers of obfs_model, in the order of
# endpoint/engine/obfs.h. A matrix the engine multiplies by gets its int8 row
# scales in <name>_s.
INTS = tuple(
    "d_model d_inner d_conv d_state dt_rank n_layers size_out time_out d_cond n_roles "
    "max_seq_len x_out emb_Bs emb_Bt abi wtype B_s B_t".split()
)
BUFFERS = tuple(
    "emb_w emb_b pe role_emb norm_w norm_b in_proj_w conv_w conv_b x_proj_w dt_proj_w "
    "dt_proj_b A D out_proj_w fnorm_w fnorm_b size_w size_b t1_wh t1_tab t2_w t2_b".split()
)
MATRICES = dict(
    zip(
        "in_proj_w x_proj_w dt_proj_w out_proj_w size_w t1_wh t2_w".split(),
        "in_proj_s x_proj_s dt_proj_s out_proj_s size_s t1_s t2_s".split(),
    )
)


def pack_rows(w: np.ndarray, dtype: str) -> Tuple[np.ndarray, np.ndarray | None]:
    """A (rows, n) float32 matrix in the engine's format: (values, row scales).

    int8 is symmetric per row: scale = max|row| / 127 (1 for a zero row),
    q = round(w / scale)."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    if dtype == "fp32":
        return w, None
    amax = np.abs(w).max(axis=1)
    scale = np.where(amax > 0, amax / 127, 1.0).astype(np.float32)
    return np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8), scale


def engine_buffers(model: Model, dtype: str) -> Tuple[Dict[str, int], Dict[str, np.ndarray]]:
    """The integers and arrays of the engine's model struct."""
    d = model.dims
    sd = {k: v.detach().cpu().float().numpy() for k, v in model.state_dict().items()}
    n, dm = d["n_layers"], d["d_model"]

    def stack(key: str) -> np.ndarray:
        return np.ascontiguousarray(np.stack([sd[f"backbone.layers.{i}.{key}"] for i in range(n)]))

    w1, b1 = sd["head.time_mlp.0.weight"], sd["head.time_mlp.0.bias"]
    cond = sd["head.size_cond.weight"]
    # The time head's first layer on [h, size embedding] = W1[:, :d_model] h +
    # (W1[:, d_model:] e(s) + b1): the second part, one row per size bin.
    t1_tab = (cond.astype(np.float64) @ w1[:, dm:].astype(np.float64).T + b1).astype(np.float32)
    d_inner = d["expand"] * dm
    bufs = {
        "emb_w": sd["embedding.linear.weight"],
        "emb_b": sd["embedding.linear.bias"],
        "pe": sd["embedding.pos_enc.pe"],
        "role_emb": sd["embedding.role_emb.weight"],
        "norm_w": stack("norm.weight"),
        "norm_b": stack("norm.bias"),
        "in_proj_w": stack("mamba.in_proj.weight"),
        "conv_w": stack("mamba.conv1d.weight").reshape(n, d_inner, d["d_conv"]),
        "conv_b": stack("mamba.conv1d.bias"),
        "x_proj_w": stack("mamba.x_proj.weight"),
        "dt_proj_w": stack("mamba.dt_proj.weight"),
        "dt_proj_b": stack("mamba.dt_proj.bias"),
        "A": -np.exp(stack("mamba.A_log")),
        "D": stack("mamba.D"),
        "out_proj_w": stack("mamba.out_proj.weight"),
        "fnorm_w": sd["backbone.final_norm.weight"],
        "fnorm_b": sd["backbone.final_norm.bias"],
        "size_w": sd["head.size_head.weight"],
        "size_b": sd["head.size_head.bias"],
        "t1_wh": w1[:, :dm],
        "t1_tab": t1_tab,
        "t2_w": sd["head.time_mlp.2.weight"],
        "t2_b": sd["head.time_mlp.2.bias"],
    }
    out = {name: np.ascontiguousarray(bufs[name], dtype=np.float32) for name in BUFFERS}
    for name, scale_name in MATRICES.items():
        q, scale = pack_rows(out[name].reshape(-1, out[name].shape[-1]), dtype)
        out[name] = q
        if scale is not None:
            out[scale_name] = scale
    dt_rank = sd["backbone.layers.0.mamba.dt_proj.weight"].shape[1]
    ints = {
        "d_model": dm,
        "d_inner": d_inner,
        "d_conv": d["d_conv"],
        "d_state": d["d_state"],
        "dt_rank": dt_rank,
        "n_layers": n,
        "size_out": d["B_s"] + 1,
        "time_out": d["B_t"] + 1,
        "d_cond": d["d_cond"],
        "n_roles": d["n_roles"],
        "max_seq_len": d["max_seq_len"],
        "x_out": dt_rank + 2 * d["d_state"],
        "emb_Bs": d["B_s"],
        "emb_Bt": d["B_t"],
        "abi": ABI,
        "wtype": WTYPE[dtype],
        "B_s": d["B_s"],
        "B_t": d["B_t"],
    }
    return ints, out


def write_weights(path: Path, ints: Dict[str, int], bufs: Dict[str, np.ndarray]) -> None:
    """The OBFSW001 file: header, buffer table, buffers at 64-byte aligned offsets."""
    names = [b for b in BUFFERS + tuple(MATRICES.values()) if b in bufs]
    header = b"OBFSW001" + struct.pack("<II", 1, len(INTS))
    header += struct.pack(f"<{len(INTS)}i", *(ints[k] for k in INTS)) + struct.pack(
        "<I", len(names)
    )
    offset = _align(len(header) + len(names) * 48)
    table, blobs = [], []
    for name in names:
        raw = bufs[name].tobytes()
        table.append(struct.pack("<32sQQ", name.encode(), offset, len(raw)))
        blobs.append((offset, raw))
        offset = _align(offset + len(raw))
    with open(path, "wb") as f:
        f.write(header + b"".join(table))
        for off, raw in blobs:
            f.write(b"\0" * (off - f.tell()) + raw)


def _align(n: int) -> int:
    return (n + 63) // 64 * 64


def export(run: Path, out: Path, cfg: dict) -> None:
    dtype = cfg["export"]["dtype"]
    if dtype not in WTYPE:
        raise SystemExit(f"export.dtype must be one of {list(WTYPE)}")
    model, ckpt = load(Path(run) / "model.pt")
    target = Path(out) / ckpt["form"]
    target.mkdir(parents=True, exist_ok=True)
    ints, bufs = engine_buffers(model, dtype)
    write_weights(target / "weights.obfw", ints, bufs)
    Tokenizer.from_dict(ckpt["tokenizer"]).save(target / "tokenizer.json")
    shutil.copy(Path(run) / "model.pt", target / "model.pt")
    log.info(
        "%s model (epoch %d, validation loss %.4f) -> %s",
        ckpt["form"],
        ckpt["epoch"],
        ckpt["val_loss"],
        target,
    )


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Export the trained models into a parameter folder.")
    ap.add_argument("--async", dest="async_run", type=Path, help="run directory of the async model")
    ap.add_argument("--sync", dest="sync_run", type=Path, help="run directory of the sync model")
    ap.add_argument("--data", required=True, type=Path, help="the dataset (for opener.json)")
    ap.add_argument("-o", "--out", required=True, type=Path, help="parameter folder")
    config.add_argument(ap)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    if not (args.async_run or args.sync_run):
        ap.error("give --async, --sync or both")
    cfg = config.load(args.config)
    for run in (args.async_run, args.sync_run):
        if run:
            export(run, args.out, cfg)
    shutil.copy(args.data / "opener.json", args.out / "opener.json")


if __name__ == "__main__":
    main()
