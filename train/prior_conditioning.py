"""Patient-prior label-query conditioning for Stage-2 v5 (train_2_v5.py), 2026-09-30.

"B-early" arm of docs/plan/2026-09-30-markov-ab-training-plan.md. Each image's prior-study state is
first passed through the patient Markov chain (analysis/patient_prior.py), giving r (31 values):

    r_c = logit pi_c(prior) - logit prevalence_c   (c = 1..30; 0 if no strictly earlier study)
    r_31 = has_prior

That is exactly the vector the offline HMM late fusion ("B-late") adds to the logits; here it is
fed into the ML-Decoder instead, as a per-image offset on the 30 label queries before they
cross-attend to the image:

    offset[b, c] = r[b, c] * u_c + W r[b]        u: (30, 768), W: Linear(31 -> 768, no bias)

u and W start at zero and there is no bias, so (1) at initialisation the model is exactly the v3
network, and (2) an image without a prior (r = 0) always gets a zero offset. Self-contained:
train_2_v3.py, train_2_v4.py, convnext.py and markov_layer.py are untouched; ml_decoder.Decoder
only gained an optional `query_offset` argument (None = unchanged).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from convnext import ConvNeXt2
from markov_layer import CONVNEXT2_KW, load_convnext2_for_state_dict

STATE_PREFIX = "prior_cond."
PRIOR_DIM = 31  # 30 per-class log prior ratios + has_prior


class PriorQueryConditioner(nn.Module):
    def __init__(self, num_classes: int = 30, dim: int = 768):
        super().__init__()
        self.num_classes = num_classes
        self.u = nn.Parameter(torch.zeros(num_classes, dim))
        self.W = nn.Linear(num_classes + 1, dim, bias=False)
        nn.init.zeros_(self.W.weight)

    def forward(self, prior: torch.Tensor) -> torch.Tensor:
        """prior (B, 31) -> query offset (B, 30, dim), computed in fp32."""
        r = prior.float()
        with torch.autocast(device_type=r.device.type, enabled=False):
            per_class = r[:, : self.num_classes, None] * self.u.float()[None]   # (B, 30, dim)
            shared = self.W(r)[:, None, :]                                     # (B, 1, dim)
            return per_class + shared

    @torch.no_grad()
    def summary(self, has_prior_frac: float | None = None) -> str:
        un = self.u.detach().float().norm(dim=1)
        s = (f"prior: |u_c| mean={un.mean():.4f} max={un.max():.4f} | "
             f"|W| fro={self.W.weight.detach().float().norm():.4f}")
        return s + (f" | batch has_prior={has_prior_frac:.3f}" if has_prior_frac is not None else "")


class ConvNeXt2Prior(ConvNeXt2):
    """ConvNeXt2 whose ML-Decoder label queries are offset by a per-image patient prior. Same
    state-dict keys as ConvNeXt2 plus `prior_cond.*`, so Stage-1 checkpoints load unchanged
    (strict=False). forward(x) without a prior is exactly ConvNeXt2.forward(x)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_cond = PriorQueryConditioner(kwargs.get("num_classes", 30),
                                                self.head.query_embed.weight.shape[1])

    def forward(self, x, prior=None):
        x = self.forward_features(x)
        x = self.pos_encoding(x)   # (B, C, H, W)
        offset = self.prior_cond(prior) if prior is not None else None
        return self.head(x, query_offset=offset)


def load_model_for_state_dict(sd: dict, num_classes: int = 30):
    """ConvNeXt2Prior for v5 prior checkpoints; otherwise defer to the v3/v4 loader in
    markov_layer. Loads with strict=False and fails loudly if the prior keys do not load."""
    if STATE_PREFIX + "u" not in sd:
        return load_convnext2_for_state_dict(sd, num_classes)
    model = ConvNeXt2Prior(**CONVNEXT2_KW, num_classes=num_classes)
    res = model.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys + res.unexpected_keys if k.startswith(STATE_PREFIX)]
    if bad:
        raise RuntimeError(f"prior-conditioning keys did not load cleanly: {bad}")
    return model, res


def is_prior_model(model: nn.Module) -> bool:
    return isinstance(model, ConvNeXt2Prior)
