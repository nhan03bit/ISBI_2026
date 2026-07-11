# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from ml_decoder import Decoder, Query2Label

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # print("x shape after convnext stages:", x.shape)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")["model"]

        # remove old classifier head keys if present
        for k in list(checkpoint.keys()):
            if "head" in k:
                del checkpoint[k]

        model.load_state_dict(checkpoint, strict=False)
    return model

@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class DecoderLayer(nn.Module):
    """
    One layer: (SelfAttn + FFN) then (CrossAttn + FFN), each with Pre-LN + residual.
    Order: SA -> CA (DETR-style).
    """
    def __init__(self, d_model=1024, nhead=8, dropout=0.1, ffn_mult=4):
        super().__init__()

        # --- Self-attn block (queries attend to queries) ---
        # self.ln_q_sa  = nn.LayerNorm(d_model)
        # self.sa       = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # self.drop_sa  = nn.Dropout(dropout)
        # self.ln_ffn1  = nn.LayerNorm(d_model)
        # self.ffn1     = FFN(d_model, d_model * ffn_mult, dropout=dropout)
        # self.drop_ffn1 = nn.Dropout(dropout)

        # --- Cross-attn block (queries attend to encoder memory) ---
        self.ln_q_ca  = nn.LayerNorm(d_model)
        self.ln_mem   = nn.LayerNorm(d_model)
        self.ca       = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop_ca  = nn.Dropout(dropout)
        self.ln_ffn2  = nn.LayerNorm(d_model)
        self.ffn2     = FFN(d_model, d_model * ffn_mult, dropout=dropout)
        self.drop_ffn2 = nn.Dropout(dropout)

    def forward(self, q, mem):
        # -------------------------
        # 1) Self-attn
        # -------------------------
        # q2 = self.ln_q_sa(q)
        # sa_out, _ = self.sa(q2, q2, q2, need_weights=False)
        # q = q + self.drop_sa(sa_out)
        # q = q + self.drop_ffn1(self.ffn1(self.ln_ffn1(q)))

        # -------------------------
        # 2) Cross-attn
        # -------------------------
        q2 = self.ln_q_ca(q)
        m2 = self.ln_mem(mem)
        ca_out, _ = self.ca(q2, m2, m2, need_weights=False)
        q = q + self.drop_ca(ca_out)
        q = q + self.drop_ffn2(self.ffn2(self.ln_ffn2(q)))


        return q


class LabelQueryHead(nn.Module):
    """
    Label-query decoder with:
    - Learnable query embeddings initialized from scratch (C, D)
    - N stacks of (CA + FFN) + (SA + FFN)
    - Final per-label classifier
    """

    def __init__(
        self,
        num_classes: int,
        d_model: int,
        n_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.1,
        init_std: float = 0.02,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.d_model = d_model

        # -------------------------------------------------
        # 1) Learnable query embeddings from scratch (C, D)
        # -------------------------------------------------
        self.query = nn.Parameter(torch.empty(num_classes, d_model))
        nn.init.trunc_normal_(self.query, std=init_std)

        # -------------------------------------------------
        # 2) Decoder layers (CA + SA), N layers
        # -------------------------------------------------
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dropout=dropout,
                ffn_mult=4,
            )
            for _ in range(n_layers)
        ])

        # -------------------------------------------------
        # 3) Final per-label classifier
        # -------------------------------------------------
        self.cls = nn.Linear(d_model, 1)

    def forward(self, mem: torch.Tensor) -> torch.Tensor:
        """
        mem: (B, S, D) encoder memory tokens
        returns: logits (B, C)
        """
        B = mem.size(0)

        # Expand learnable queries to batch
        q = self.query.unsqueeze(0).expand(B, -1, -1)  # (B, C, D)

        # CA + SA decoder stack
        for layer in self.layers:
            q = layer(q, mem)

        logits = self.cls(q).squeeze(-1)  # (B, C)
        return logits


from positional_encodings.torch_encodings import PositionalEncoding2D, Summer
from einops import rearrange
NORM_EPS = 1e-5

class Block(nn.Module):
    r"""ConvNeXt Block (stride=1, preserves HxW)."""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)  
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)                 # (B,C,H,W)
        x = x.permute(0, 2, 3, 1)          # (B,H,W,C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)          # (B,C,H,W)
        x = residual + self.drop_path(x)
        return x


class ConvNeXt2(nn.Module):
    """
    ConvNeXt2 variant with total spatial reduction /64 (instead of /32).

    Change vs current ConvNeXt:
      - Add 1 extra downsample layer (stride=2)
      - Add 1 extra stage of Blocks after it
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 27, 3, 3],             # NEW: 5 stages
        dims=[128, 256, 512, 1024, 1024],    # NEW: 5 dims (keep last=1024 for checkpoint compatibility)
        drop_path_rate=0.,
        layer_scale_init_value=1e-6,
        head_init_scale=1.,
    ):
        super().__init__()

        assert len(depths) == len(dims) == 5, "For /64 variant, provide 5-stage depths and dims."

        # ---- Downsample layers ----
        # stem: /4
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        # 3 original downsample layers: /2 each (total /32)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # NEW: extra downsample layer for /64
        # dims[3] -> dims[4], stride 2
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
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.head = Decoder(
            num_classes=num_classes,
            initial_num_features=dims[-1],
            decoder_embedding=768,
            num_of_groups=100,
            num_layers=2,
            activation="gelu",
        )
        # self.cls = nn.Linear(dims[-1], num_classes)

        self.pos_encoding = Summer(PositionalEncoding2D(dims[-1]))
        # print("Pos encoding shape: ", self.pos_encoding.shape)

        self.apply(self._init_weights)
        # (Optional scaling, as in original ConvNeXt)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

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
        # Now 5 downsample+stage pairs
        for i in range(5):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x  # (B, dims[-1], H/64, W/64)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pos_encoding(x)   # (B, C, H, W)
        x = self.head(x)
        return x



if __name__ == "__main__":
    model = ConvNeXt2(depths=[3, 3, 27, 3, 2], 
                      dims=[128, 256, 512, 1024, 1024], 
                      num_classes=30)
    
    print(model)

    # url = model_urls['convnext_base_1k']
    # checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")["model"]
    # remove = ['norm.weight', 'norm.bias', 'head.weight', 'head.bias']

    # for k in list(checkpoint.keys()):
    #     if k in remove:
    #         del checkpoint[k]

    # print("Checkpoint keys after removing classifier head and norm:", checkpoint.keys())
    # model.load_state_dict(checkpoint, strict=False)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"MedViT total parameters: {total_params:,}")

    # # Count trainable only
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"MedViT trainable parameters: {trainable_params:,}")

    # x = torch.randn(2, 3, 224, 224)
    # y = model(x)
    # print(y.shape)