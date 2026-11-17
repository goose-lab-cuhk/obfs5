"""Train the async or the sync model.

    python model/train.py DATA --form async|sync -o RUN [--config FILE] [--resume]

Writes RUN/model.pt (best validation loss), RUN/last.pt (for --resume) and
RUN/log.csv.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import config
from representation.tokenizer import Tokenizer
from seqmodel import data
from seqmodel.model import Model, kernels_available, nll_sums, save

log = logging.getLogger("obfsgen.train")
LOG_FIELDS = (
    "epoch",
    "train_loss",
    "train_size",
    "train_time",
    "val_loss",
    "val_size",
    "val_time",
    "lr",
    "seconds",
    "tokens_per_s",
    "skipped_steps",
)


def lr_factor(step: int, warmup: int, total: int, min_ratio: float) -> float:
    """Linear warm-up over ``warmup`` updates, then a cosine from 1 to ``min_ratio`` at ``total``."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _to(batch: dict, device: torch.device):
    roles = batch.get("roles")
    return (
        batch["tokens"].to(device, non_blocking=True),
        batch["mask"].to(device, non_blocking=True),
        None if roles is None else roles.to(device, non_blocking=True),
    )


@torch.no_grad()
def evaluate(model: Model, batches, device: torch.device) -> tuple[float, float, float]:
    """Mean (loss, size loss, time loss) per predicted token."""
    model.eval()
    sums = torch.zeros(3, device=device)
    for batch in batches:
        sums += torch.stack(nll_sums(model, *_to(batch, device)))
    s, t, n = sums.tolist()
    n = max(n, 1.0)
    return (s + t) / n, s / n, t / n


def run(cfg: dict, data_dir: Path, form: str, out: Path, resume: bool = False) -> dict:
    """Train; returns the last line of log.csv."""
    tc = cfg["train"]
    random.seed(tc["seed"])
    np.random.seed(tc["seed"])
    torch.manual_seed(tc["seed"])
    device = torch.device(tc["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("train.device is cuda but no GPU is visible; set it to cpu in the config")
    kernels = device.type == "cuda" and kernels_available()
    if device.type == "cuda" and not kernels:
        log.warning("mamba-ssm is not installed: the model runs the slow PyTorch block")

    tok = Tokenizer.load(Path(data_dir) / f"tokenizer_{form}.json")
    train_set = data.load_split(Path(data_dir) / "train.parquet", form, tok.B_t)
    val_set = data.load_split(Path(data_dir) / "val.parquet", form, tok.B_t)
    dims = {
        **cfg["model"],
        "n_roles": 2 if form == "async" else 1,
        "max_seq_len": tok.max_records + 1,
        "B_s": tok.B_s,
        "B_t": tok.B_t,
    }
    model = Model(dims, kernels).to(device)
    train_batches = data.loader(train_set, tc["batch_tokens"], True, tc)
    val_batches = data.loader(val_set, tc["eval_batch_tokens"], False, tc)
    n_steps = len(train_batches)
    # summed loss / mean targets per batch: every target weighs the same
    divisor = float(np.clip(train_set.lengths - 1, 0, None).sum()) / max(n_steps, 1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_factor(
            step, tc["warmup_steps"], tc["epochs"] * n_steps, tc["min_lr_ratio"]
        ),
    )
    fp16 = tc["fp16"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start, best_loss, best_epoch = 0, math.inf, 0
    if resume:
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        if (
            state["config"] != {k: cfg[k] for k in ("model", "train")}
            or state["tokenizer"] != tok.to_dict()
        ):
            raise SystemExit(f"{out}/last.pt was trained with other settings or data")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        scaler.load_state_dict(state["scaler"])
        start, best_loss, best_epoch = state["epoch"], state["best_loss"], state["best_epoch"]
        log.info("resuming after epoch %d (best %.4f at epoch %d)", start, best_loss, best_epoch)
    elif (out / "log.csv").exists():
        (out / "log.csv").unlink()

    log.info(
        "%s model: %d parameters, %d train / %d val streams, %d steps per epoch, %s",
        form,
        sum(p.numel() for p in model.parameters()),
        len(train_set),
        len(val_set),
        n_steps,
        "mamba-ssm kernels" if kernels else "PyTorch block",
    )
    row = {}
    for epoch in range(start, tc["epochs"]):
        train_batches.batch_sampler.set_epoch(epoch)
        model.train()
        t0 = time.perf_counter()
        sums = torch.zeros(4, device=device)  # size, time, targets, skipped steps
        n_tokens = 0
        for step, batch in enumerate(train_batches):
            tokens, mask, roles = _to(batch, device)
            optimizer.zero_grad()
            with torch.autocast(device.type, dtype=torch.float16, enabled=fp16):
                s, t, n = nll_sums(model, tokens, mask, roles)
                loss = (s + t) / divisor
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(optimizer)  # skipped when the gradient is not finite
            scaler.update()
            schedule.step()
            sums += torch.stack(
                [s.detach().float(), t.detach().float(), n.float(), (~torch.isfinite(norm)).float()]
            )
            n_tokens += int(mask.sum())
            if tc["log_every"] and step % tc["log_every"] == 0:
                log.info(
                    "  epoch %d step %d/%d, loss %.4f, lr %.2e",
                    epoch + 1,
                    step,
                    n_steps,
                    float((s + t).detach() / n.clamp_min(1)),
                    optimizer.param_groups[0]["lr"],
                )
        seconds = time.perf_counter() - t0
        size_sum, time_sum, n_sum, skipped = sums.tolist()
        val_loss, val_size, val_time = evaluate(model, val_batches, device)
        n_sum = max(n_sum, 1.0)
        row = {
            "epoch": epoch + 1,
            "train_loss": (size_sum + time_sum) / n_sum,
            "train_size": size_sum / n_sum,
            "train_time": time_sum / n_sum,
            "val_loss": val_loss,
            "val_size": val_size,
            "val_time": val_time,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.perf_counter() - t0, 1),
            "tokens_per_s": round(n_tokens / max(seconds, 1e-9)),
            "skipped_steps": int(skipped),
        }
        with open(out / "log.csv", "a", newline="") as fh:
            w = csv.DictWriter(fh, LOG_FIELDS)
            if fh.tell() == 0:
                w.writeheader()
            w.writerow(row)
        log.info(
            "epoch %d: train %.4f, val %.4f (size %.4f + time %.4f), %d tok/s",
            epoch + 1,
            row["train_loss"],
            val_loss,
            val_size,
            val_time,
            row["tokens_per_s"],
        )

        if not math.isfinite(val_loss):
            log.error(
                "the validation loss is not finite: training diverged; model.pt keeps epoch %d",
                best_epoch,
            )
            break
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            save(out / "model.pt", model, form, tok.to_dict(), cfg, epoch + 1, val_loss)
        save(
            out / "last.pt",
            model,
            form,
            tok.to_dict(),
            cfg,
            epoch + 1,
            val_loss,
            optimizer=optimizer.state_dict(),
            schedule=schedule.state_dict(),
            scaler=scaler.state_dict(),
            best_loss=best_loss,
            best_epoch=best_epoch,
        )
        if epoch + 1 - best_epoch >= tc["patience"]:
            log.info("no better validation loss for %d epochs: stopping", tc["patience"])
            break
    log.info("best validation loss %.4f at epoch %d -> %s", best_loss, best_epoch, out / "model.pt")
    return row


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train the async or the sync model.")
    ap.add_argument("data", type=Path, help="a dataset directory written by build_dataset.py")
    ap.add_argument("--form", required=True, choices=("async", "sync"))
    ap.add_argument("-o", "--out", required=True, type=Path, help="run directory")
    ap.add_argument("--resume", action="store_true", help="continue from OUT/last.pt")
    config.add_argument(ap)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    run(config.load(args.config), args.data, args.form, args.out, args.resume)


if __name__ == "__main__":
    main()
