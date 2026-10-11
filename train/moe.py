# moe.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoEOutput:
    y: torch.Tensor
    aux_loss: torch.Tensor


class Top2Router(nn.Module):
    """
    Top-2 router producing:
      - top2_idx: (T, 2) expert indices
      - top2_w:   (T, 2) normalized weights over top2 experts
      - aux_loss: load balancing loss (Switch-style)
    """
    def __init__(self, d_model: int, n_experts: int = 6, router_init_scale: float = 1e-3):
        super().__init__()
        self.n_experts = n_experts
        self.linear = nn.Linear(d_model, n_experts)

        # Near-uniform routing at init: small weights -> near-zero logits
        nn.init.normal_(self.linear.weight, mean=0.0, std=router_init_scale)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (T, D)
        returns:
          top2_idx: (T,2) long
          top2_w:   (T,2) float
          aux_loss: scalar
        """
        logits = self.linear(x)  # (T, E)
        top2_val, top2_idx = torch.topk(logits, k=2, dim=-1)  # (T,2), (T,2)

        # normalized weights among top-2 only
        top2_w = F.softmax(top2_val, dim=-1)  # (T,2)

        # Build sparse gate matrix for aux loss computation: (T,E)
        T = x.size(0)
        E = self.n_experts
        gate = x.new_zeros((T, E))  # float

        gate.scatter_(dim=1, index=top2_idx, src=top2_w)

        # Switch-style load balancing:
        # importance = sum of gate weights per expert
        # load = count of tokens routed to expert (non-zero gates)
        importance = gate.sum(dim=0)                           # (E,)
        load = (gate > 0).sum(dim=0).to(gate.dtype)            # (E,)

        # l_aux = E * sum(importance * load) / T^2
        denom = float(T * T) if T > 0 else 1.0
        aux_loss = (E * (importance * load).sum()) / denom

        return top2_idx, top2_w, aux_loss


class MLPExpert(nn.Module):
    """Standard ConvNeXt-style MLP: Linear(d->4d) -> GELU -> Linear(4d->d)"""
    def __init__(self, d_model: int, expansion: int = 4):
        super().__init__()
        hidden = expansion * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class MoETop2MLP(nn.Module):
    """
    Top-2 MoE over an MLP.
    Expert 0 is a "base" MLP implemented by caller (to preserve checkpoint keys).
    Experts 1..E-1 live in extra_experts ModuleList.

    x can be (..., D) and will be flattened to (T,D) for routing.
    """
    def __init__(
        self,
        d_model: int,
        n_experts: int = 6,
        expansion: int = 4,
        router_init_scale: float = 1e-3,
    ):
        super().__init__()
        assert n_experts >= 2, "Need at least 2 experts for top-2 routing."
        self.d_model = d_model
        self.n_experts = n_experts
        self.router = Top2Router(d_model, n_experts=n_experts, router_init_scale=router_init_scale)
        self.extra_experts = nn.ModuleList([MLPExpert(d_model, expansion=expansion) for _ in range(n_experts - 1)])

    def forward(self, x: torch.Tensor, base_expert_fn) -> MoEOutput:
        """
        base_expert_fn: callable taking (N,D) -> (N,D) for expert 0.
        """
        orig_shape = x.shape
        D = orig_shape[-1]
        x2 = x.reshape(-1, D)  # (T,D)
        T = x2.size(0)

        top2_idx, top2_w, aux_loss = self.router(x2)  # (T,2), (T,2), scalar

        y2 = x2.new_zeros((T, D))

        # Route sparsely: compute only for assigned experts
        for e in range(self.n_experts):
            # tokens where expert e appears in top-2 (either slot)
            m0 = top2_idx[:, 0] == e
            m1 = top2_idx[:, 1] == e
            if not (m0.any() or m1.any()):
                continue

            idx = (m0 | m1).nonzero(as_tuple=False).squeeze(1)  # (Ne,)
            x_e = x2.index_select(0, idx)  # (Ne,D)

            if e == 0:
                y_e = base_expert_fn(x_e)
            else:
                y_e = self.extra_experts[e - 1](x_e)

            # weights: pick from slot 0 or slot 1
            w = x2.new_zeros((idx.numel(),))
            if m0.any():
                idx0 = m0.nonzero(as_tuple=False).squeeze(1)
                # positions within idx array:
                pos0 = torch.searchsorted(idx, idx0)
                w[pos0] = top2_w.index_select(0, idx0)[:, 0]
            if m1.any():
                idx1 = m1.nonzero(as_tuple=False).squeeze(1)
                pos1 = torch.searchsorted(idx, idx1)
                w[pos1] = top2_w.index_select(0, idx1)[:, 1]

            y2.index_add_(0, idx, y_e * w.unsqueeze(1))

        y = y2.reshape(orig_shape)
        return MoEOutput(y=y, aux_loss=aux_loss)


class FFNExpert(nn.Module):
    """Transformer FFN: Linear(d->ff) -> act -> Dropout -> Linear(ff->d)"""
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float, activation: str = "gelu"):
        super().__init__()
        if activation.lower() == "relu":
            act = nn.ReLU()
        elif activation.lower() == "silu" or activation.lower() == "swish":
            act = nn.SiLU()
        elif activation.lower() == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.act = act
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class MoETop2FFN(nn.Module):
    """
    Top-2 MoE over a Transformer FFN.
    Expert 0 is a base FFN callable (to preserve checkpoint keys).
    """
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        n_experts: int = 6,
        dropout: float = 0.1,
        activation: str = "gelu",
        router_init_scale: float = 1e-3,
    ):
        super().__init__()
        assert n_experts >= 2
        self.d_model = d_model
        self.n_experts = n_experts
        self.router = Top2Router(d_model, n_experts=n_experts, router_init_scale=router_init_scale)
        self.extra_experts = nn.ModuleList([
            FFNExpert(d_model, dim_feedforward, dropout=dropout, activation=activation)
            for _ in range(n_experts - 1)
        ])

    def forward(self, x: torch.Tensor, base_expert_fn) -> MoEOutput:
        """
        x: (..., D)
        base_expert_fn: callable (N,D)->(N,D) for expert 0
        """
        orig_shape = x.shape
        D = orig_shape[-1]
        x2 = x.reshape(-1, D)
        T = x2.size(0)

        top2_idx, top2_w, aux_loss = self.router(x2)

        y2 = x2.new_zeros((T, D))

        for e in range(self.n_experts):
            m0 = top2_idx[:, 0] == e
            m1 = top2_idx[:, 1] == e
            if not (m0.any() or m1.any()):
                continue

            idx = (m0 | m1).nonzero(as_tuple=False).squeeze(1)
            x_e = x2.index_select(0, idx)

            if e == 0:
                y_e = base_expert_fn(x_e)
            else:
                y_e = self.extra_experts[e - 1](x_e)

            w = x2.new_zeros((idx.numel(),))
            if m0.any():
                idx0 = m0.nonzero(as_tuple=False).squeeze(1)
                pos0 = torch.searchsorted(idx, idx0)
                w[pos0] = top2_w.index_select(0, idx0)[:, 0]
            if m1.any():
                idx1 = m1.nonzero(as_tuple=False).squeeze(1)
                pos1 = torch.searchsorted(idx, idx1)
                w[pos1] = top2_w.index_select(0, idx1)[:, 1]

            y2.index_add_(0, idx, y_e * w.unsqueeze(1))

        y = y2.reshape(orig_shape)
        return MoEOutput(y=y, aux_loss=aux_loss)
