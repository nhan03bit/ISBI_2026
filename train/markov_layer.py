"""Markov label-refinement layer for Stage-2 v4 (train_2_v4.py), 2026-09-26.

Self-contained: train_2_v3.py, convnext.py and evaluate/evaluate_tta.py are untouched.
v4 builds `ConvNeXt2Markov` (a ConvNeXt2 subclass) instead of `ConvNeXt2`;
evaluate/evaluate_tta_v4.py loads checkpoints with `load_convnext2_for_state_dict`.

Offline, a *fixed* one-step random walk over the label co-occurrence graph lowered the best
ensemble's internal mAP by 0.0006-0.0053 (docs/result/2026-09-26-markov-applicability.md).
This layer is the trainable version, applied to the ML-Decoder logits u (B, C):

    P      = row_softmax(A, diagonal masked)        # C x C, P[i, j] ~ P(j | i)
    z_0    = u
    z_k    = u + g * (sigmoid(z_{k-1}) @ P) + b,     k = 1..K
    output = z_K

A starts at log P_cooc (training-label co-occurrence, diagonal removed, rows normalised),
so P starts as the empirical label-transition matrix and remains a valid row-stochastic
Markov matrix while it trains. The per-class gate g (signed: a message can excite or
inhibit, e.g. Normal vs any finding) and bias b start at 0, so at initialisation the layer
is the identity and the network reproduces the v3 baseline exactly.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from convnext import ConvNeXt2

STATE_PREFIX = "label_refine."
CONVNEXT2_KW = dict(depths=[3, 3, 27, 3, 2], dims=[128, 256, 512, 1024, 1024])


def cooccurrence_transition(labels: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """(N, C) binary label matrix -> (C, C) row-stochastic P[i, j] = P(j | i), zero diagonal."""
    co = labels.astype(np.float64).T @ labels.astype(np.float64)
    co += eps  # every transition possible, so log() stays finite
    np.fill_diagonal(co, 0.0)
    return co / co.sum(axis=1, keepdims=True)


class MarkovLabelRefiner(nn.Module):
    def __init__(self, num_classes: int, n_steps: int = 1,
                 transition_init: np.ndarray | None = None, learn_transition: bool = True):
        super().__init__()
        c = num_classes
        if transition_init is None:
            transition_init = np.full((c, c), 1.0 / (c - 1))
        logp = np.log(np.clip(transition_init, 1e-12, None))
        np.fill_diagonal(logp, 0.0)  # masked in transition(); keep the parameter finite
        self.A = nn.Parameter(torch.tensor(logp, dtype=torch.float32),
                              requires_grad=learn_transition)
        self.gate = nn.Parameter(torch.zeros(c))
        self.bias = nn.Parameter(torch.zeros(c))
        # float so EMA / SWA averaging keeps it intact; read back by from_state_dict
        self.register_buffer("n_steps", torch.tensor(float(n_steps)))
        self.register_buffer("offdiag", 1.0 - torch.eye(c), persistent=False)

    def transition(self) -> torch.Tensor:
        a = self.A.float().masked_fill(self.offdiag == 0, float("-inf"))
        return torch.softmax(a, dim=1)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        u = logits.float()
        with torch.autocast(device_type=u.device.type, enabled=False):
            P = self.transition()
            z = u
            for _ in range(int(round(self.n_steps.item()))):
                z = u + self.gate * (torch.sigmoid(z) @ P) + self.bias
        return z

    @torch.no_grad()
    def summary(self, class_names=None, top: int = 5) -> str:
        g = self.gate.detach().float().cpu()
        names = list(class_names) if class_names is not None else [str(i) for i in range(len(g))]
        order = torch.argsort(g.abs(), descending=True)[:top].tolist()
        strongest = ", ".join(f"{names[i]}={g[i]:+.3f}" for i in order)
        return (f"markov: steps={int(round(self.n_steps.item()))} |gate| mean={g.abs().mean():.4f} "
                f"max={g.abs().max():.4f} | |bias| mean={self.bias.detach().abs().mean():.4f} "
                f"| strongest gates: {strongest}")


class ConvNeXt2Markov(ConvNeXt2):
    """ConvNeXt2 + Markov label refinement on the head logits. Same state-dict keys as
    ConvNeXt2 plus `label_refine.*`, so Stage-1 checkpoints load unchanged (strict=False)."""

    def __init__(self, *args, markov_steps: int = 1, transition_init=None,
                 learn_transition: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_refine = MarkovLabelRefiner(kwargs.get("num_classes", 30), markov_steps,
                                               transition_init, learn_transition)

    def forward(self, x):
        logits, embedding_spatial = super().forward(x)
        return self.label_refine(logits), embedding_spatial


def load_convnext2_for_state_dict(sd: dict, num_classes: int = 30):
    """Plain ConvNeXt2 for v3 checkpoints, ConvNeXt2Markov (with the checkpoint's step count)
    for v4 Markov checkpoints. Loads `sd` (strict=False, as evaluate_tta does) and returns
    (model, load_result); fails loudly if the Markov keys do not load."""
    if STATE_PREFIX + "gate" in sd:
        steps = int(round(float(sd[STATE_PREFIX + "n_steps"])))
        model = ConvNeXt2Markov(**CONVNEXT2_KW, num_classes=num_classes, markov_steps=steps)
    else:
        model = ConvNeXt2(**CONVNEXT2_KW, num_classes=num_classes)
    res = model.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys + res.unexpected_keys if k.startswith(STATE_PREFIX)]
    if bad:
        raise RuntimeError(f"Markov layer keys did not load cleanly: {bad}")
    return model, res
