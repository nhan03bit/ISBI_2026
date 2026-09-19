from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn.modules.transformer import _get_activation_fn


# def add_ml_decoder_head(model, num_classes=-1, num_of_groups=-1, decoder_embedding=768, zsl=0):
#     if num_classes == -1:
#         num_classes = model.num_classes
#     num_features = model.num_features
#     if hasattr(model, 'global_pool') and hasattr(model, 'fc'):  # resnet50
#         model.global_pool = nn.Identity()
#         del model.fc
#         model.fc = MLDecoder(num_classes=num_classes, initial_num_features=num_features, num_of_groups=num_of_groups,
#                              decoder_embedding=decoder_embedding, zsl=zsl)
#     elif hasattr(model, 'head'):  # tresnet
#         if hasattr(model, 'global_pool'):
#             model.global_pool = nn.Identity()
#         del model.head
#         model.head = MLDecoder(num_classes=num_classes, initial_num_features=num_features, num_of_groups=num_of_groups,
#                                decoder_embedding=decoder_embedding, zsl=zsl)
#     else:
#         print("model is not suited for ml-decoder")
#         exit(-1)

#     return model


class TransformerDecoderLayerOptimal(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5) -> None:
        super(TransformerDecoderLayerOptimal, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayerOptimal, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


# @torch.jit.script
# class ExtrapClasses(object):
#     def __init__(self, num_queries: int, group_size: int):
#         self.num_queries = num_queries
#         self.group_size = group_size
#
#     def __call__(self, h: torch.Tensor, class_embed_w: torch.Tensor, class_embed_b: torch.Tensor, out_extrap:
#     torch.Tensor):
#         # h = h.unsqueeze(-1).expand(-1, -1, -1, self.group_size)
#         h = h[..., None].repeat(1, 1, 1, self.group_size) # torch.Size([bs, 5, 768, groups])
#         w = class_embed_w.view((self.num_queries, h.shape[2], self.group_size))
#         out = (h * w).sum(dim=2) + class_embed_b
#         out = out.view((h.shape[0], self.group_size * self.num_queries))
#         return out

@torch.jit.script
class GroupFC(object):
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            if len(duplicate_pooling.shape)==3:
                w_i = duplicate_pooling[i, :, :]
            else:
                w_i = duplicate_pooling
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)


def _get_activation_module(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class DecoderLayer_SACA(nn.Module):
    """
    One block: SA -> FFN -> CA -> FFN
    (Pre-LN + residual)
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
    ):
        super().__init__()
        self.batch_first = batch_first

        # --- Self-attn ---
        self.ln_q_sa = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.sa = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.drop_sa = nn.Dropout(dropout)

        self.ln_ffn_sa = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn_sa = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            _get_activation_module(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop_ffn_sa = nn.Dropout(dropout)

        # --- Cross-attn ---
        self.ln_q_ca = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ln_mem = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ca = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.drop_ca = nn.Dropout(dropout)

        self.ln_ffn_ca = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn_ca = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            _get_activation_module(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop_ffn_ca = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # ---- SA ----
        q1 = self.ln_q_sa(tgt)
        sa_out = self.sa(
            query=q1,
            key=q1,
            value=q1,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.drop_sa(sa_out)

        # ---- FFN after SA ----
        ffn1 = self.ffn_sa(self.ln_ffn_sa(tgt))
        tgt = tgt + self.drop_ffn_sa(ffn1)

        # ---- CA ----
        q2 = self.ln_q_ca(tgt)
        m = self.ln_mem(memory)
        ca_out = self.ca(
            query=q2,
            key=m,
            value=m,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.drop_ca(ca_out)

        # ---- FFN after CA ----
        ffn2 = self.ffn_ca(self.ln_ffn_ca(tgt))
        tgt = tgt + self.drop_ffn_ca(ffn2)

        return tgt


class DecoderLayer_CASA(nn.Module):
    """
    One block: CA -> FFN -> SA -> FFN
    (CA+SA order, Pre-LN + residual)
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
    ):
        super().__init__()
        self.batch_first = batch_first

        act = _get_activation_module(activation)

        # --- Cross-attn ---
        self.ln_q_ca = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ln_mem  = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ca = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.drop_ca = nn.Dropout(dropout)

        self.ln_ffn_ca = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn_ca = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop_ffn_ca = nn.Dropout(dropout)

        # --- Self-attn ---
        self.ln_q_sa = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.sa = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.drop_sa = nn.Dropout(dropout)

        self.ln_ffn_sa = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn_sa = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act.__class__(),          # create a new instance of same activation module
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop_ffn_sa = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
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

        # ---- FFN after CA ----
        ffn1 = self.ffn_ca(self.ln_ffn_ca(tgt))
        tgt = tgt + self.drop_ffn_ca(ffn1)

        # ---- SA ----
        q2 = self.ln_q_sa(tgt)
        sa_out = self.sa(
            query=q2, key=q2, value=q2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False
        )[0]
        tgt = tgt + self.drop_sa(sa_out)

        # ---- FFN after SA ----
        ffn2 = self.ffn_sa(self.ln_ffn_sa(tgt))
        tgt = tgt + self.drop_ffn_sa(ffn2)

        return tgt




class Decoder(nn.Module):
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
    ):
        super().__init__()

        embed_len_decoder = 100 if num_of_groups < 0 else num_of_groups
        embed_len_decoder = min(embed_len_decoder, num_classes)

        self.decoder_embedding = decoder_embedding
        self.embed_standart = nn.Linear(initial_num_features, decoder_embedding)

        if not zsl:
            self.query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)
            self.query_embed.requires_grad_(False)
        else:
            self.query_embed = None


        self.decoder_layers = nn.ModuleList([
            DecoderLayer_CASA(
                d_model=decoder_embedding,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=False,
            )
            for _ in range(num_layers)
        ])

        self.zsl = zsl
        self.num_classes = num_classes
        self.duplicate_factor = int(num_classes / embed_len_decoder + 0.999)

        self.duplicate_pooling = nn.Parameter(
            torch.Tensor(embed_len_decoder, decoder_embedding, self.duplicate_factor)
        )
        self.duplicate_pooling_bias = nn.Parameter(torch.Tensor(num_classes))

        torch.nn.init.xavier_normal_(self.duplicate_pooling)
        torch.nn.init.constant_(self.duplicate_pooling_bias, 0)

        self.group_fc = GroupFC(embed_len_decoder)

    def forward(self, x, mask=None):
        # x: (B,C,H,W) or (B,S,C)
        if x.dim() == 4:
            embedding_spatial = x.flatten(2).transpose(1, 2)  # (B,S,C)
        else:
            embedding_spatial = x                               # (B,S,C)

        # project memory to decoder_embedding
        embedding_spatial = self.embed_standart(embedding_spatial)  # (B,S,D)
        embedding_spatial = torch.relu(embedding_spatial)

        B = embedding_spatial.size(0)

        # fixed queries
        query_embed = self.query_embed.weight                   # (Q,D)
        tgt = query_embed.unsqueeze(1).expand(-1, B, -1)         # (Q,B,D)

        # memory to (S,B,D) for batch_first=False MHA
        memory = embedding_spatial.transpose(0, 1)               # (S,B,D)

        # stacked CA+SA blocks
        for layer in self.decoder_layers:
            tgt = layer(
                tgt=tgt,
                memory=memory,
                memory_key_padding_mask=mask,   # should be (B,S) or None
                tgt_key_padding_mask=None,
            )

        h = tgt.transpose(0, 1)  # (B,Q,D)

        # Expose the decoded per-class query embeddings for downstream metric
        # learning (train_2_v3.py). This is the (B, Q, D) tensor the entropy
        # paper describes as the triplet embedding source; other callers
        # (train.py stage-1, evaluate.py) simply ignore this attribute.
        # Note: the decoder here uses num_of_groups grouped queries (Q may be
        # < num_classes when duplicate_factor > 1); with num_classes=30 and
        # num_of_groups>=30 this is one query per class.
        self.last_h = h

        out_extrap = torch.zeros(
            B, h.size(1), self.duplicate_factor,
            device=h.device, dtype=h.dtype
        )

        self.group_fc(h, self.duplicate_pooling, out_extrap)

        logits = out_extrap.flatten(1)[:, :self.num_classes]
        logits += self.duplicate_pooling_bias
        return logits, embedding_spatial



#------------------------------------------------Query2Label--------------------------------------------------
class Query2LabelDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
    ):
        super().__init__()
        self.batch_first = batch_first

        act = _get_activation_module(activation)

        # --- Self-attention ---
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.dropout1 = nn.Dropout(dropout)

        # --- Cross-attention ---
        self.norm2_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2_mem = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.dropout2 = nn.Dropout(dropout)

        # --- FFN ---
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # 1) Self-attention
        q = self.norm1(tgt)
        sa_out = self.self_attn(
            query=q,
            key=q,
            value=q,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout1(sa_out)

        # 2) Cross-attention
        q = self.norm2_q(tgt)
        m = self.norm2_mem(memory)
        ca_out = self.cross_attn(
            query=q,
            key=m,
            value=m,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = tgt + self.dropout2(ca_out)

        # 3) FFN
        ffn_out = self.ffn(self.norm3(tgt))
        tgt = tgt + self.dropout3(ffn_out)

        return tgt


class Query2Label(nn.Module):
    """
    A cleaner Query2Label-style head:
      - one learnable query per class
      - no duplicate pooling
      - direct per-class prediction

    Input:
      x: (B, C, H, W) or (B, S, C)

    Output:
      logits: (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        initial_num_features: int = 2048,
        decoder_embedding: int = 768,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.decoder_embedding = decoder_embedding

        # project backbone features to decoder dimension
        self.input_proj = nn.Linear(initial_num_features, decoder_embedding)

        # one learnable query per class
        self.query_embed = nn.Embedding(num_classes, decoder_embedding)

        # stacked decoder layers
        self.decoder_layers = nn.ModuleList([
            Query2LabelDecoderLayer(
                d_model=decoder_embedding,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=False,
            )
            for _ in range(num_layers)
        ])

        # one scalar logit per query/class
        self.classifier = nn.Linear(decoder_embedding, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.constant_(self.input_proj.bias, 0)

        nn.init.xavier_uniform_(self.query_embed.weight)

        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # x: (B, C, H, W) or (B, S, C)
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)  # (B, S, C)

        # project memory features
        memory = self.input_proj(x)           # (B, S, D)
        memory = torch.relu(memory)
        B = memory.size(0)

        # class queries
        tgt = self.query_embed.weight.unsqueeze(1).expand(-1, B, -1)  # (Q, B, D), Q = num_classes

        # convert memory to (S, B, D) for batch_first=False
        memory = memory.transpose(0, 1)  # (S, B, D)

        # decoder
        for layer in self.decoder_layers:
            tgt = layer(
                tgt=tgt,
                memory=memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=mask,
            )

        # (Q, B, D) -> (B, Q, D)
        h = tgt.transpose(0, 1)

        # direct one-logit-per-query/class
        logits = self.classifier(h).squeeze(-1)  # (B, Q) = (B, num_classes)

        return logits
