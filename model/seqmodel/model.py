"""The sequence model: token embedding, Mamba blocks, a size head and a time head
conditioned on the size bin. Uses the mamba-ssm kernels on a GPU when they are
installed, else the PyTorch block in mamba.py (same weights)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import TorchMamba

FORMAT = "obfsgen-model-v1"


def kernels_available() -> bool:
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)


class Embedding(nn.Module):
    def __init__(self, d_model: int, B_s: int, B_t: int, max_len: int, n_roles: int):
        super().__init__()
        self.B_s, self.B_t = B_s, B_t
        self.linear = nn.Linear(2, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.role_emb = nn.Embedding(n_roles, d_model)  # unused by the sync model (n_roles 1)

    def forward(self, tokens: torch.Tensor, roles: Optional[torch.Tensor]) -> torch.Tensor:
        x = torch.stack([tokens[..., 0].float() / self.B_s, tokens[..., 1].float() / self.B_t], -1)
        x = self.linear(x) + self.pos_enc.pe[: tokens.shape[1]]
        if roles is not None:
            x = x + self.role_emb(roles).unsqueeze(1)
        return x


class Block(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, i: int, kernels: bool):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if kernels:
            from mamba_ssm import Mamba

            self.mamba = Mamba(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, layer_idx=i
            )
        else:
            self.mamba = TorchMamba(d_model, d_state, d_conv, expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class Backbone(nn.Module):
    def __init__(self, d: dict, kernels: bool):
        super().__init__()
        self.layers = nn.ModuleList(
            Block(d["d_model"], d["d_state"], d["d_conv"], d["expand"], i, kernels)
            for i in range(d["n_layers"])
        )
        self.final_norm = nn.LayerNorm(d["d_model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class Head(nn.Module):
    def __init__(self, d_model: int, B_s: int, B_t: int, d_cond: int):
        super().__init__()
        self.size_head = nn.Linear(d_model, B_s + 1)
        self.size_cond = nn.Embedding(B_s + 1, d_cond)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model + d_cond, d_model), nn.ReLU(), nn.Linear(d_model, B_t + 1)
        )

    def time_logits(self, h: torch.Tensor, size_bin: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(torch.cat([h, self.size_cond(size_bin)], dim=-1))


class Model(nn.Module):
    """``dims``: n_layers, d_model, d_state, d_conv, expand, d_cond (the config's
    ``model`` section) plus n_roles, max_seq_len, B_s, B_t (from the data)."""

    def __init__(self, dims: dict, kernels: bool = False):
        super().__init__()
        self.dims = dict(dims)
        d = self.dims
        self.embedding = Embedding(d["d_model"], d["B_s"], d["B_t"], d["max_seq_len"], d["n_roles"])
        self.backbone = Backbone(d, kernels)
        self.head = Head(d["d_model"], d["B_s"], d["B_t"], d["d_cond"])

    def hidden(self, tokens: torch.Tensor, roles: Optional[torch.Tensor] = None) -> torch.Tensor:
        """h at every position of ``tokens`` (B, L, 2)."""
        if (roles is None) != (self.dims["n_roles"] == 1):
            raise ValueError(
                "the async model needs roles (0 client, 1 server); the sync model none"
            )
        return self.backbone(self.embedding(tokens, roles))

    def forward(
        self, tokens: torch.Tensor, roles: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Logits for positions 1..L-1 of ``tokens`` (B, L, 2), each predicted from
        the tokens before it: size (B, L-1, B_s+1) and time (B, L-1, B_t+1)."""
        h = self.hidden(tokens, roles)[:, :-1]
        return self.head.size_head(h), self.head.time_logits(h, tokens[:, 1:, 0])


def nll_sums(
    model: Model, tokens: torch.Tensor, mask: torch.Tensor, roles: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(summed size NLL, summed time NLL, predicted tokens) of a padded batch."""
    size_logits, time_logits = model(tokens, roles)
    nll_s = F.cross_entropy(size_logits.transpose(1, 2), tokens[:, 1:, 0], reduction="none")
    nll_t = F.cross_entropy(time_logits.transpose(1, 2), tokens[:, 1:, 1], reduction="none")
    target = mask[:, 1:].to(nll_s.dtype)
    return (nll_s * target).sum(), (nll_t * target).sum(), target.sum()


def save(
    path: Path,
    model: Model,
    form: str,
    tokenizer: dict,
    cfg: dict,
    epoch: int,
    val_loss: float,
    **extra,
) -> None:
    """Write a run's model file; ``extra`` adds the state --resume needs."""
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(
        {
            "format": FORMAT,
            "form": form,
            "dims": model.dims,
            "tokenizer": tokenizer,
            "config": {k: cfg[k] for k in ("model", "train")},
            "epoch": epoch,
            "val_loss": val_loss,
            "state_dict": state,
            **extra,
        },
        path,
    )


def load(path: Path, device: str | torch.device = "cpu") -> Tuple[Model, dict]:
    """A model file → (model in eval mode on ``device``, the file's contents)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("format") != FORMAT:
        raise ValueError(f"{path}: not an {FORMAT} file")
    device = torch.device(device)
    model = Model(ckpt["dims"], kernels=device.type == "cuda" and kernels_available())
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
