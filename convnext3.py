# convnext.py (additions)
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from moe import MoETop2MLP, MoETop2FFN #type: ignore
from convnext import Block, LayerNorm, DropPath, trunc_normal_
from ml_decoder import Decoder, _get_activation_module
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer


class BlockMoE(nn.Module):
    """
    ConvNeXt Block with MoE inserted into the MLP part.
    - Expert 0 uses original pwconv1/pwconv2 (checkpoint-compatible)
    - Experts 1..E-1 are new (in moe_mlp.extra_experts)
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, n_experts: int = 6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)  # channels_last LN
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.moe_mlp = MoETop2MLP(d_model=dim, n_experts=n_experts, expansion=4)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # will be filled at forward for logging/training
        self.last_aux_loss = None

    def _base_expert(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C) channels-last token vectors
        return self.pwconv2(self.act(self.pwconv1(x)))

    def forward(self, x):
        residual = x
        x = self.dwconv(x)                 # (B,C,H,W)
        x = x.permute(0, 2, 3, 1)          # (B,H,W,C)
        x = self.norm(x)

        moe_out = self.moe_mlp(x, base_expert_fn=self._base_expert)  # MoEOutput
        x = moe_out.y
        self.last_aux_loss = moe_out.aux_loss

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 3, 1, 2)          # (B,C,H,W)
        x = residual + self.drop_path(x)
        return x


class DecoderLayer_CASA_MoE(nn.Module):
    """
    Same as your DecoderLayer_CASA, but replace only FFN after CA with MoE.
    Keep original ffn_ca as expert0 for checkpoint compatibility.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        n_experts: int = 6,
    ):
        super().__init__()
        self.batch_first = batch_first

        act = _get_activation_module(activation)

        # --- Cross-attn ---
        self.ln_q_ca = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ln_mem  = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ca = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.drop_ca = nn.Dropout(dropout)

        # Keep the original FFN as expert0 (checkpoint compatible)
        self.ln_ffn_ca = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn_ca = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop_ffn_ca = nn.Dropout(dropout)

        # MoE wrapper for FFN-after-CA (experts 1..E-1 are new)
        self.moe_ffn_ca = MoETop2FFN(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            n_experts=n_experts,
            dropout=dropout,
            activation=activation,
        )

        # --- Self-attn (unchanged) ---
        self.ln_q_sa = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.sa = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.drop_sa = nn.Dropout(dropout)

        self.ln_ffn_sa = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn_sa = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act.__class__(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop_ffn_sa = nn.Dropout(dropout)

        self.last_aux_loss = None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # ---- CA ----
        q = self.ln_q_ca(tgt)
        m = self.ln_mem(memory)
        ca_out = self.ca(
            query=q, key=m, value=m,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False
        )[0]
        tgt = tgt + self.drop_ca(ca_out)

        # ---- MoE FFN after CA ----
        z = self.ln_ffn_ca(tgt)
        moe_out = self.moe_ffn_ca(z, base_expert_fn=self.ffn_ca)
        self.last_aux_loss = moe_out.aux_loss
        tgt = tgt + self.drop_ffn_ca(moe_out.y)

        # ---- SA ----
        q2 = self.ln_q_sa(tgt)
        sa_out = self.sa(
            query=q2, key=q2, value=q2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False
        )[0]
        tgt = tgt + self.drop_sa(sa_out)

        # ---- FFN after SA (unchanged) ----
        ffn2 = self.ffn_sa(self.ln_ffn_sa(tgt))
        tgt = tgt + self.drop_ffn_sa(ffn2)

        return tgt


class DecoderMoE(Decoder):
    def __init__(
        self,
        num_classes,
        num_of_groups=-1,
        decoder_embedding=768,
        initial_num_features=2048,
        zsl=0,
        num_layers: int = 3,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        n_experts: int = 6,
    ):
        # 1. Initialize the base Decoder
        super().__init__(
            num_classes=num_classes,
            num_of_groups=num_of_groups,
            decoder_embedding=decoder_embedding,
            initial_num_features=initial_num_features,
            zsl=zsl,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

        # if self.query_embed is not None:
        #     self.query_embed.requires_grad_(True)

        # 3. Override decoder_layers with MoE version
        self.decoder_layers = nn.ModuleList([
            DecoderLayer_CASA_MoE(
                d_model=decoder_embedding,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=False,
                n_experts=n_experts,
            )
            for _ in range(num_layers)
        ])

        self.last_aux_loss = None

    def forward(self, x, mask=None):
        logits = super().forward(x, mask=mask)

        # aggregate aux losses from decoder layers (if any)
        aux = 0.0
        cnt = 0
        for layer in self.decoder_layers:
            if getattr(layer, "last_aux_loss", None) is not None:
                aux = aux + layer.last_aux_loss
                cnt += 1
        self.last_aux_loss = aux / max(cnt, 1) if cnt > 0 else torch.tensor(0.0, device=logits.device)

        return logits


class ConvNeXt3(nn.Module):
    """
    ConvNeXt2 flow + MoE:
      - MoE in MLP part of blocks in last 2 stages (stage indices 3 and 4)
      - MoE in Decoder FFN after CA only
    """
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 27, 3, 2],
        dims=[128, 256, 512, 1024, 1024],
        drop_path_rate=0.,
        layer_scale_init_value=1e-6,
        head_init_scale=1.,
        moe_n_experts: int = 6,
    ):
        super().__init__()
        assert len(depths) == len(dims) == 5, "ConvNeXt3 expects 5-stage (/64) config."

        # ---- Downsample layers (identical naming/structure as ConvNeXt2) ----
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        extra_downsample = nn.Sequential(
            LayerNorm(dims[3], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[3], dims[4], kernel_size=2, stride=2),
        )
        self.downsample_layers.append(extra_downsample)

        # ---- Stages ----
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(5):
            BlockCls = BlockMoE if i in (3, 4) else Block  # MoE only in last 2 stages
            stage = nn.Sequential(
                *[
                    BlockCls(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        **({"n_experts": moe_n_experts} if BlockCls is BlockMoE else {})
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        # ---- Head (MoE decoder) ----
        self.head = DecoderMoE(
            num_classes=num_classes,
            initial_num_features=dims[-1],
            decoder_embedding=768,
            num_of_groups=100,
            num_layers=2,
            activation="gelu",
            n_experts=moe_n_experts,
        )

        self.pos_encoding = Summer(PositionalEncoding2D(dims[-1]))

        self.apply(self._init_weights)

        self.last_aux_loss = None

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):

            
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.MultiheadAttention):
            trunc_normal_(m.in_proj_weight, std=0.02)
            if m.in_proj_bias is not None:
                nn.init.constant_(m.in_proj_bias, 0)
            trunc_normal_(m.out_proj.weight, std=0.02)
            if m.out_proj.bias is not None:
                nn.init.constant_(m.out_proj.bias, 0)

    def forward_features(self, x):
        for i in range(5):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pos_encoding(x)

        logits = self.head(x)

        # aggregate aux losses from ConvNeXt MoE blocks + decoder MoE
        aux = 0.0
        cnt = 0
        for si in (3, 4):
            for blk in self.stages[si]:
                if getattr(blk, "last_aux_loss", None) is not None:
                    aux = aux + blk.last_aux_loss
                    cnt += 1

        if getattr(self.head, "last_aux_loss", None) is not None:
            aux = aux + self.head.last_aux_loss
            cnt += 1

        self.last_aux_loss = aux / max(cnt, 1) if cnt > 0 else torch.tensor(0.0, device=logits.device)

        return logits

# model = ConvNeXt3(
#         depths=[3, 3, 27, 3, 2],
#         dims=[128, 256, 512, 1024, 1024],
#         num_classes=4,
#         moe_n_experts=4,
#     )

# print(model)