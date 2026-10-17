"""Class-conditional memory loss for train_2_v5.py --metric-loss nca (2026-10-16).

Why: the shipped Triplet Memory (utils_update.sample_triplets_v13) is near-inactive. Its easy-mined hinge is slack
for ~99% of triplets, it acts on mean_c h_c, which the readout logit_c = h_c . w_c + b_c barely sees, and its rare
large gradients inflate AdamW's second moment (docs/plan/2026-10-15-triplet-memory-debug.md). This module keeps the
two ideas, a cross-batch memory and an entropy gate, and changes what they act on (docs/plan/2026-10-16-beat-asl-plan.md):

  * embedding: each class's own decoded query embedding h_c (the vector the readout reads), L2-normalised;
  * memory: a FIFO of the last `size` images' h (all classes) with labels, plus a per-class queue of the last
    `queue` positives of that class, so a tail class always has positives to compare with (a 4096-image FIFO holds
    ~1.5 Hydropneumothorax positives on average). Entries carry no gradient;
  * loss: soft-neighbour (NCA) classification of every gated (image, class) pair against the memory,
        p_mem[i,c] = sum_{j in P_c} e^{s_ij / tau} / sum_{j in P_c + N_c} e^{s_ij / tau},
    s = cosine of h_ic with the memory's class-c embeddings, P_c = queue c + bank positives of c, N_c = bank negatives
    of c, and loss = BCE(p_mem, y_ic). Dense and smooth (no hinge that switches off), symmetric for positive and
    negative pairs, and probabilistically aligned with the per-class decision the readout makes;
  * entropy gate per (image, class): the classifier's Bernoulli entropy H(sigmoid(z_ic)) >= its mean over the
    micro-batch's pairs ("mean", the multi-label form of the paper's rule), every pair ("all"), or as many random
    pairs ("random", control).
The trainer bounds the term's gradient on h to at most --nca-ratio x the ASL gradient (train_2_v5.py).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class ClassMemory:
    def __init__(self, size: int, queue: int, num_classes: int, dim: int, device):
        self.size, self.queue, self.C = int(size), int(queue), int(num_classes)
        # fp32 (4096 x 30 x 768 = 377 MB): the loss runs outside autocast, so cosines keep full precision
        self.bank = torch.zeros(self.size, self.C, dim, dtype=torch.float32, device=device)
        self.bank_y = torch.zeros(self.size, self.C, dtype=torch.bool, device=device)
        self.q = torch.zeros(self.C, self.queue, dim, dtype=torch.float32, device=device)
        self.q_n = [0] * self.C            # filled entries per class queue (python ints: no GPU syncs)
        self.q_ptr = [0] * self.C
        self.ptr = 0
        self.filled = 0

    @torch.no_grad()
    def update(self, hn: torch.Tensor, y: torch.Tensor) -> None:
        """hn (B, C, D) L2-normalised per-class embeddings, y (B, C) bool labels."""
        B = hn.size(0)
        idx = (torch.arange(B, device=hn.device) + self.ptr) % self.size
        self.bank[idx] = hn.to(self.bank.dtype)
        self.bank_y[idx] = y
        self.ptr = (self.ptr + B) % self.size
        self.filled = min(self.size, self.filled + B)
        for b, c in torch.nonzero(y, as_tuple=False).tolist():
            p = self.q_ptr[c]
            self.q[c, p] = hn[b, c].to(self.q.dtype)
            self.q_ptr[c] = (p + 1) % self.queue
            self.q_n[c] = min(self.queue, self.q_n[c] + 1)


def binary_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-7, 1 - 1e-7)
    return -(p * torch.log(p) + (1 - p) * torch.log1p(-p))


def class_memory_nca(h, y, logits, mem: ClassMemory, tau: float = 0.1, gate: str = "mean", warm: int = 1024,
                     generator=None, stats: dict | None = None):
    """Mean BCE(p_mem, y) over the gated (image, class) pairs; a zero that keeps the graph if nothing is gated.

    h (B, C, D) decoded per-class query embeddings (with grad), y (B, C) bool, logits (B, C) classifier logits
    (only used, detached, for the gate)."""
    B, C, _ = h.shape
    zero = h.sum() * 0.0
    if mem.filled < warm:
        return zero
    with torch.autocast("cuda", enabled=False):
        q = F.normalize(h.float(), dim=-1)
        bank = mem.bank[:mem.filled]                                     # (S, C, D) f32
        by = mem.bank_y[:mem.filled].t()                                 # (C, S)
        s_b = torch.einsum("bcd,scd->bcs", q, bank) / tau               # (B, C, S)
        s_q = torch.einsum("bcd,cqd->bcq", q, mem.q) / tau              # (B, C, Q)
        q_valid = torch.arange(mem.queue, device=h.device)[None, :] < torch.tensor(mem.q_n, device=h.device)[:, None]
        ninf = torch.finfo(s_b.dtype).min
        pos = torch.cat([s_b.masked_fill(~by[None], ninf), s_q.masked_fill(~q_valid[None], ninf)], -1)
        neg = s_b.masked_fill(by[None], ninf)
        lse_pos, lse_neg = torch.logsumexp(pos, -1), torch.logsumexp(neg, -1)      # (B, C)
        lse_all = torch.logaddexp(lse_pos, lse_neg)
    has_pos = (by.any(1) | q_valid.any(1))[None].expand(B, C)
    has_neg = (~by).any(1)[None].expand(B, C)

    with torch.no_grad():
        H = binary_entropy(torch.sigmoid(logits.detach().float()))
        if gate == "all":
            g = torch.ones_like(H, dtype=torch.bool)
        elif gate == "mean":
            g = H >= H.mean()
        elif gate == "random":
            k = int((H >= H.mean()).sum().item())
            perm = torch.randperm(B * C, generator=generator)[:k].to(h.device)
            g = torch.zeros(B * C, dtype=torch.bool, device=h.device)
            g[perm] = True
            g = g.view(B, C)
        else:
            raise ValueError(f"unknown nca gate {gate!r}")
        g = g & has_pos & has_neg
    if stats is not None:
        stats["anchors"] = stats.get("anchors", 0) + int(g.sum().item())
        stats["anchors_pos"] = stats.get("anchors_pos", 0) + int((g & y).sum().item())
        stats["calls"] = stats.get("calls", 0) + 1
    if not g.any():
        return zero
    yf = y.float()
    bce = -(yf * (lse_pos - lse_all) + (1 - yf) * (lse_neg - lse_all))
    loss = bce[g].mean()
    if stats is not None:
        stats["loss_sum"] = stats.get("loss_sum", 0.0) + float(loss.detach())
    return loss
