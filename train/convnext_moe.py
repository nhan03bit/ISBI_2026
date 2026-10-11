"""Mixture-of-Experts ConvNeXt2 for the matched-recipe MoE baseline, 2026-10-08.

Rebuilds the ISBI 2026 MoE Stage 2 (b12; root convnext3.py + moe.py, which is the only MoE code in
the repo) on top of train/convnext.py's ConvNeXt2, so it runs inside the same trainer and evaluator
as Triplet Memory (train_2_v5.py, evaluate_tta_v5.py):

  - stages 3 and 4: every Block's pointwise MLP becomes a top-2 MoE (BlockMoE, convnext3.py:13-55);
  - ML-Decoder: in each layer only the FFN after cross-attention becomes a top-2 MoE
    (DecoderLayer_CASA_MoE, convnext3.py:58-160); self-attention and the FFN after it are unchanged;
  - expert 0 is the original layer, so a Stage-1 checkpoint loads into the same keys and only
    `extra_experts` / `router` keys are new;
  - forward() stores the mean of the stage-3/4 block aux losses and the decoder's mean aux loss in
    `self.last_aux_loss`, exactly as ConvNeXt3.forward (convnext3.py:328-349).

Differences from convnext3.py, all needed to share the Triplet Memory pipeline: forward returns
(logits, embedding_spatial) like ConvNeXt2; the decoder is train/ml_decoder.Decoder (keeps `last_h`
and the `query_offset` hook used by the Markov prior); `query_embed` stays frozen as in ConvNeXt2.
New modules get ConvNeXt2._init_weights, as ConvNeXt3 did via self.apply (so the router weights are
trunc-normal std 0.02, not moe.py's 1e-3); `init_experts_from_base` then copies expert 0 into the
extra experts with small noise (train_moe.py:47-71).
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn

from convnext import ConvNeXt2, Block
from ml_decoder import DecoderLayer_CASA
from moe import MoETop2MLP, MoETop2FFN
from prior_conditioning import PriorQueryConditioner

MOE_KEY_MARKERS = (".extra_experts.", ".router.")


class BlockMoE(Block):
    """ConvNeXt Block whose pointwise MLP is a top-2 MoE; expert 0 = pwconv1 -> GELU -> pwconv2."""

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, n_experts: int = 4):
        super().__init__(dim, drop_path=drop_path, layer_scale_init_value=layer_scale_init_value)
        self.moe_mlp = MoETop2MLP(d_model=dim, n_experts=n_experts, expansion=4)
        self.last_aux_loss = None

    def _base_expert(self, x: torch.Tensor) -> torch.Tensor:
        return self.pwconv2(self.act(self.pwconv1(x)))

    def forward(self, x):
        residual = x
        x = self.dwconv(x)                 # (B,C,H,W)
        x = x.permute(0, 2, 3, 1)          # (B,H,W,C)
        x = self.norm(x)
        moe_out = self.moe_mlp(x, base_expert_fn=self._base_expert)
        x = moe_out.y
        self.last_aux_loss = moe_out.aux_loss
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)          # (B,C,H,W)
        x = residual + self.drop_path(x)
        return x


class DecoderLayer_CASA_MoE(DecoderLayer_CASA):
    """DecoderLayer_CASA with only the FFN after cross-attention replaced by a top-2 MoE."""

    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu", layer_norm_eps: float = 1e-5,
                 batch_first: bool = False, n_experts: int = 4):
        super().__init__(d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
                         activation=activation, layer_norm_eps=layer_norm_eps,
                         batch_first=batch_first)
        self.moe_ffn_ca = MoETop2FFN(d_model=d_model, dim_feedforward=dim_feedforward,
                                     n_experts=n_experts, dropout=dropout, activation=activation)
        self.last_aux_loss = None

    def forward(self, tgt, memory, memory_key_padding_mask=None, tgt_key_padding_mask=None,
                tgt_mask=None, memory_mask=None):
        # ---- CA ----
        q = self.ln_q_ca(tgt)
        m = self.ln_mem(memory)
        ca_out = self.ca(query=q, key=m, value=m, attn_mask=memory_mask,
                         key_padding_mask=memory_key_padding_mask, need_weights=False)[0]
        tgt = tgt + self.drop_ca(ca_out)

        # ---- MoE FFN after CA (expert 0 = ffn_ca) ----
        moe_out = self.moe_ffn_ca(self.ln_ffn_ca(tgt), base_expert_fn=self.ffn_ca)
        self.last_aux_loss = moe_out.aux_loss
        tgt = tgt + self.drop_ffn_ca(moe_out.y)

        # ---- SA ----
        q2 = self.ln_q_sa(tgt)
        sa_out = self.sa(query=q2, key=q2, value=q2, attn_mask=tgt_mask,
                         key_padding_mask=tgt_key_padding_mask, need_weights=False)[0]
        tgt = tgt + self.drop_sa(sa_out)

        # ---- FFN after SA (unchanged) ----
        tgt = tgt + self.drop_ffn_sa(self.ffn_sa(self.ln_ffn_sa(tgt)))
        return tgt


class ConvNeXt2MoE(ConvNeXt2):
    """ConvNeXt2 with MoE in stages 3-4 and in the decoder FFN after cross-attention."""

    MOE_STAGES = (3, 4)

    def __init__(self, *args, moe_n_experts: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.moe_n_experts = moe_n_experts
        new_modules = []
        for si in self.MOE_STAGES:
            blocks = []
            for blk in self.stages[si]:
                dp = blk.drop_path.drop_prob if hasattr(blk.drop_path, "drop_prob") else 0.0
                ls = 0.0 if blk.gamma is None else 1e-6
                nb = BlockMoE(blk.dwconv.in_channels, drop_path=dp, layer_scale_init_value=ls,
                              n_experts=moe_n_experts)
                blocks.append(nb)
            self.stages[si] = nn.Sequential(*blocks)
            new_modules.append(self.stages[si])

        head = self.head
        old = head.decoder_layers[0]
        d_model = old.ln_q_ca.normalized_shape[0]
        layers = nn.ModuleList([
            DecoderLayer_CASA_MoE(
                d_model=d_model,
                nhead=old.ca.num_heads,
                dim_feedforward=old.ffn_ca[0].out_features,
                dropout=old.drop_ca.p,
                activation="gelu",
                layer_norm_eps=old.ln_q_ca.eps,
                batch_first=old.batch_first,
                n_experts=moe_n_experts,
            )
            for _ in range(len(head.decoder_layers))
        ])
        head.decoder_layers = layers
        new_modules.append(layers)
        for mod in new_modules:
            mod.apply(self._init_weights)
        self.last_aux_loss = None
        self._load_counts = None
        self._hooks = []

    # ---- aux loss -------------------------------------------------------------------------
    def _collect_aux(self, device):
        aux, cnt = 0.0, 0
        for si in self.MOE_STAGES:
            for blk in self.stages[si]:
                if getattr(blk, "last_aux_loss", None) is not None:
                    aux = aux + blk.last_aux_loss
                    cnt += 1
        dec = [l.last_aux_loss for l in self.head.decoder_layers
               if getattr(l, "last_aux_loss", None) is not None]
        if dec:
            aux = aux + sum(dec) / len(dec)
            cnt += 1
        return aux / cnt if cnt > 0 else torch.zeros((), device=device)

    def forward(self, x):
        logits, emb = super().forward(x)
        self.last_aux_loss = self._collect_aux(logits.device)
        return logits, emb

    # ---- routing statistics (logging only) ------------------------------------------------
    def routers(self):
        for name, mod in self.named_modules():
            if name.endswith(".router"):
                yield name, mod

    def start_load_tracking(self):
        """Count, per router, how many token slots go to each expert (top-2 -> 2 per token)."""
        self.stop_load_tracking()
        self._load_counts = defaultdict(lambda: torch.zeros(self.moe_n_experts, dtype=torch.long))

        def make_hook(name):
            def hook(_mod, _inp, out):
                idx = out[0].detach().reshape(-1).to("cpu")
                self._load_counts[name] += torch.bincount(idx, minlength=self.moe_n_experts)
            return hook

        self._hooks = [m.register_forward_hook(make_hook(n)) for n, m in self.routers()]

    def stop_load_tracking(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def load_summary(self) -> str:
        """One line: min/max expert share over all routers, and the most unbalanced router."""
        if not self._load_counts:
            return "moe load: no data"
        worst, worst_name, mins, maxs = 0.0, "", [], []
        for name, c in self._load_counts.items():
            share = c.double() / max(int(c.sum()), 1)
            mins.append(float(share.min()))
            maxs.append(float(share.max()))
            if float(share.max()) > worst:
                worst, worst_name = float(share.max()), name
        return (f"moe load: expert share min {min(mins):.3f} max {max(maxs):.3f} "
                f"(uniform {1 / self.moe_n_experts:.3f}); most loaded router {worst_name} "
                f"max share {worst:.3f}")


class ConvNeXt2MoEPrior(ConvNeXt2MoE):
    """ConvNeXt2MoE whose label queries are offset by the per-image patient Markov prior
    (same PriorQueryConditioner as ConvNeXt2Prior; zero-init, so it starts as ConvNeXt2MoE)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_cond = PriorQueryConditioner(kwargs.get("num_classes", 30),
                                                self.head.query_embed.weight.shape[1])

    def forward(self, x, prior=None):
        x = self.forward_features(x)
        x = self.pos_encoding(x)
        offset = self.prior_cond(prior) if prior is not None else None
        logits, emb = self.head(x, query_offset=offset)
        self.last_aux_loss = self._collect_aux(logits.device)
        return logits, emb


@torch.no_grad()
def _copy_with_noise(dst: nn.Linear, src: nn.Linear, noise: float, gen: torch.Generator) -> None:
    for d, s in ((dst.weight, src.weight), (dst.bias, src.bias)):
        d.copy_(s)
        if noise > 0:
            d.add_((torch.randn(d.shape, generator=gen) * (noise * s.float().std().item()))
                   .to(d.device, d.dtype))


@torch.no_grad()
def init_experts_from_base(model: ConvNeXt2MoE, noise: float, seed: int) -> int:
    """Experts 1..E-1 <- expert 0 (the Stage-1 weights) + N(0, (noise*std)^2) (train_moe.py:47-71).
    Call after the Stage-1 weights are loaded. Returns the number of experts set."""
    gen = torch.Generator().manual_seed(seed)
    n = 0
    for si in model.MOE_STAGES:
        for blk in model.stages[si]:
            for ex in blk.moe_mlp.extra_experts:
                _copy_with_noise(ex.fc1, blk.pwconv1, noise, gen)
                _copy_with_noise(ex.fc2, blk.pwconv2, noise, gen)
                n += 1
    for layer in model.head.decoder_layers:
        for ex in layer.moe_ffn_ca.extra_experts:
            _copy_with_noise(ex.fc1, layer.ffn_ca[0], noise, gen)
            _copy_with_noise(ex.fc2, layer.ffn_ca[3], noise, gen)
            n += 1
    return n


def is_moe_state_dict(sd: dict) -> bool:
    return any(any(m in k for m in MOE_KEY_MARKERS) for k in sd)


def is_moe_key(k: str) -> bool:
    return any(m in k for m in MOE_KEY_MARKERS)
