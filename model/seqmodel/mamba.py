"""The Mamba block in plain PyTorch, with the parameters of ``mamba_ssm.Mamba``.

Runs anywhere, but the scan is sequential and slow: full-size training needs
the mamba-ssm kernels.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchMamba(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)

        # initialisation as in mamba-ssm, in the same order
        std = self.dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        a = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` (B, L, d_model) → (B, L, d_model)."""
        seqlen = x.shape[1]
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(self.conv1d(u.transpose(1, 2))[..., :seqlen]).transpose(1, 2)
        dt, b, c = torch.split(self.x_proj(u), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        a = -torch.exp(self.A_log.float())
        state = u.new_zeros(x.shape[0], self.d_inner, self.d_state)
        ys = []
        for t in range(seqlen):
            state = state * torch.exp(dt[:, t].unsqueeze(-1) * a) + (dt[:, t] * u[:, t]).unsqueeze(
                -1
            ) * b[:, t].unsqueeze(1)
            ys.append((state * c[:, t].unsqueeze(1)).sum(-1))
        y = (torch.stack(ys, dim=1) + u * self.D) * F.silu(z)
        return self.out_proj(y)
