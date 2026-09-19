import numpy as np
import torch
import random
import torch
import torch.nn.functional as F

random.seed(42)


def scheduled_value(epoch, list_of_vals):
    sched_val = None
    for (st, val) in list_of_vals:
        if st <= epoch:
            sched_val = val
    return sched_val

class CrossBatchMemoryV2:
    def __init__(self, embedding_dim, memory_size=8192, num_classes=30, device='cuda'):
        self.device = device
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # memory bank in VRAM, half precision
        self.embeddings = torch.zeros((memory_size, embedding_dim), dtype=torch.float32, device=device)
        self.labels = torch.zeros((memory_size, num_classes), dtype=torch.float32, device=device)
        self.ptr = 0
        self.filled = 0

    @torch.no_grad()
    def update(self, emb_batch, labels_batch):
        """
        Add a batch of embeddings and labels to memory
        """
        B = emb_batch.size(0)
        emb_batch = emb_batch.to(self.device).half()
        labels_batch = labels_batch.to(self.device)

        if self.ptr + B <= self.memory_size:
            self.embeddings[self.ptr:self.ptr + B] = emb_batch
            self.labels[self.ptr:self.ptr + B] = labels_batch
        else:
            # wrap-around
            first = self.memory_size - self.ptr
            self.embeddings[self.ptr:] = emb_batch[:first]
            self.labels[self.ptr:] = labels_batch[:first]
            remaining = B - first
            self.embeddings[:remaining] = emb_batch[first:]
            self.labels[:remaining] = labels_batch[first:]

        self.ptr = (self.ptr + B) % self.memory_size
        self.filled = min(self.memory_size, self.filled + B)



def calculate_entropy_from_logits(logits, dim=1, mode="softmax"):
    """Per-sample predictive entropy of the classification head.

    mode="softmax" : Shannon entropy of ``softmax(logits)`` over the class axis.
        This is the single-label / mutually-exclusive form. It is what the
        original code and the paper's Eq. (1) used, but it is inconsistent with a
        multi-label task (the 30 findings are not mutually exclusive), so it is
        kept only for the ablation.
    mode="binary"  : sum of per-label Bernoulli entropies of ``sigmoid(logits)``,
        i.e. ``sum_c [-p_c log p_c - (1-p_c) log(1-p_c)]`` with ``p_c = sigma(z_c)``.
        This is the correct multi-label predictive entropy and the new default
        for the training scripts.

    For anchor *ranking* within a fixed-size batch the sum-vs-mean choice is
    monotonic and does not matter; we sum ("total uncertainty").
    """
    if mode == "softmax":
        probs = torch.softmax(logits, dim=dim)
        log_probs = torch.log_softmax(logits, dim=dim)
        return -(probs * log_probs).sum(dim=dim)
    if mode == "binary":
        eps = 1e-7
        p = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
        h = -(p * torch.log(p) + (1.0 - p) * torch.log1p(-p))
        return h.sum(dim=dim)
    raise ValueError(f"unknown entropy mode {mode!r}")


def _anchor_gate(entropy, rule="mean"):
    """Return a boolean [B] mask of which batch samples may be triplet anchors."""
    B = entropy.numel()
    if rule == "all":
        return torch.ones(B, dtype=torch.bool, device=entropy.device)
    if rule == "max":                       # paper Eq. (2): the single most-uncertain sample
        return entropy >= entropy.max()
    if rule == "mean":                      # shipped default: ~the more-uncertain half
        return entropy >= entropy.mean()
    if rule == "random":                    # ablation: entropy-agnostic control
        k = max(1, int((entropy >= entropy.mean()).sum().item()))
        perm = torch.randperm(B, device=entropy.device)[:k]
        m = torch.zeros(B, dtype=torch.bool, device=entropy.device)
        m[perm] = True
        return m
    raise ValueError(f"unknown anchor rule {rule!r}")


def _mine_pair(pos_d, neg_d, margin, mining):
    """Given anchor->positive and anchor->negative L2 distances, return the
    (pos_local_idx, neg_local_idx) to use. Distances are 1D tensors.

    mining="easy"     : closest positive + farthest negative  (shipped behaviour;
                        maximally noise-robust, but not "hard mining").
    mining="hard"     : farthest positive + closest negative  (batch-hard).
    mining="semihard" : farthest positive; negative that is farther than that
                        positive but within the margin (closest such), else the
                        hardest negative.
    """
    if mining == "easy":
        return int(torch.argmin(pos_d)), int(torch.argmax(neg_d))
    if mining == "hard":
        return int(torch.argmax(pos_d)), int(torch.argmin(neg_d))
    if mining == "semihard":
        p_local = int(torch.argmax(pos_d))
        dap = pos_d[p_local]
        band = (neg_d > dap) & (neg_d < dap + margin)
        if band.any():
            cand = torch.where(band)[0]
            n_local = int(cand[torch.argmin(neg_d[cand])])
        else:
            n_local = int(torch.argmin(neg_d))
        return p_local, n_local
    raise ValueError(f"unknown mining {mining!r}")


def sample_triplets_v12(batch_embeddings, logits, batch_labels, memory, margin=0.5,
                        temperature=0.05, *, anchor_rule="mean", pos_min_shared=1,
                        mining="easy", entropy_mode="softmax"):
    """
    Sample triplets using Cross-Batch Memory safely on GPU.

    Args:
        batch_embeddings: Tensor of shape [B, D]
        batch_labels: Tensor of shape [B, C] (multi-label)
        memory: object storing memory embeddings and labels
        margin: float, triplet margin
        anchor_rule: "mean" (shipped) | "max" (paper Eq. 2) | "all" | "random"
        pos_min_shared: min shared labels for a positive (paper Eq. 3 uses > 1,
            i.e. pos_min_shared=2; shipped code uses 1)
        mining: "easy" (shipped) | "hard" | "semihard"
        entropy_mode: "softmax" (paper Eq. 1 / shipped) | "binary" (multi-label)

    Returns:
        (anchors, positives, negatives) stacked tensors, or None.
    """
    device = batch_embeddings.device
    B = batch_embeddings.size(0)
    logits = logits.to(device)

    # Get memory embeddings and labels. Use `filled` (true occupancy) rather than
    # `ptr` (write cursor): ptr wraps modulo memory_size, so slicing [:ptr] once
    # the ring buffer is full exposes only 0..(size-B) entries and is empty on the
    # step where ptr wraps back to 0.
    mem_count = min(getattr(memory, "filled", memory.ptr), memory.memory_size)
    if mem_count == 0:
        return None
    mem_emb = memory.embeddings[:mem_count].to(device)
    mem_lbl = memory.labels[:mem_count].to(device)

    triplets = []
    dist = torch.cdist(batch_embeddings.float(), mem_emb.float(), p=2)

    entropy = calculate_entropy_from_logits(logits, mode=entropy_mode)
    if isinstance(entropy, (list, tuple)):
        entropy = torch.stack(entropy)
    gate = _anchor_gate(entropy, rule=anchor_rule)

    for i in range(B):
        if not gate[i]:
            continue

        anchor = batch_embeddings[i]
        anchor_label = batch_labels[i]

        # Positive = shares >= pos_min_shared active labels with the anchor.
        # Negative = shares zero labels with the anchor.
        shared = (mem_lbl * anchor_label).sum(dim=1)
        pos_mask = shared >= pos_min_shared
        neg_mask = shared == 0

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        pos_idx = torch.where(pos_mask)[0]
        neg_idx = torch.where(neg_mask)[0]

        pos_d = dist[i][pos_idx]
        neg_d = dist[i][neg_idx]
        if pos_d.numel() == 0 or neg_d.numel() == 0:
            continue

        p_local, n_local = _mine_pair(pos_d, neg_d, margin, mining)
        triplets.append((anchor, mem_emb[pos_idx[p_local]], mem_emb[neg_idx[n_local]]))

    if len(triplets) == 0:
        return None

    anc, pos, neg = zip(*triplets)
    return (torch.stack(anc), torch.stack(pos), torch.stack(neg))


def sample_triplets_v13(
    batch_embeddings,
    logits,
    batch_labels,
    memory,
    margin=0.5,
    temperature=0.05,
    head_class_ids=None,
    pos_min_shared=1,
):
    """
    train_2_v3 triplet sampler. Same entropy-gated / hardest-mined structure as
    sample_triplets_v12, with one change: the positive label-overlap test
    ignores the most frequent ("head") classes, so a positive pair must share a
    genuine non-head finding rather than merely both being "Normal" / an
    otherwise-dominant label. The negative test is unchanged (shares zero labels
    of any kind), so negatives stay genuinely dissimilar.

    Args:
        head_class_ids: 1D LongTensor / list of class indices to exclude from the
            positive-overlap computation. None -> behaves like v12 with
            pos_mask = (shared >= pos_min_shared).
        pos_min_shared: minimum shared non-head labels for a positive (default 1).
    """
    device = batch_embeddings.device
    B = batch_embeddings.size(0)
    logits = logits.to(device)

    mem_count = min(getattr(memory, "filled", memory.ptr), memory.memory_size)
    mem_emb = memory.embeddings[:mem_count].to(device)
    mem_lbl = memory.labels[:mem_count].to(device)

    if mem_count == 0:
        return None

    # Non-head label mask over classes: 1 for kept (tail/mid) classes, 0 for head.
    C = batch_labels.size(1)
    keep_mask = torch.ones(C, device=device)
    if head_class_ids is not None and len(head_class_ids) > 0:
        idx = torch.as_tensor(head_class_ids, device=device, dtype=torch.long).flatten()
        keep_mask[idx] = 0.0

    triplets = []
    dist = torch.cdist(batch_embeddings.float(), mem_emb.float(), p=2)

    entropy = calculate_entropy_from_logits(logits)
    if isinstance(entropy, (list, tuple)):
        entropy = torch.stack(entropy)
    entropy_thr = entropy.mean()

    for i in range(B):
        if entropy[i] < entropy_thr:
            continue

        anchor = batch_embeddings[i]
        anchor_label = batch_labels[i]

        # Positive: shares >= pos_min_shared NON-head labels with the anchor.
        shared_pos = ((mem_lbl * keep_mask) * (anchor_label * keep_mask)).sum(dim=1)
        pos_mask = shared_pos >= pos_min_shared
        # Negative: shares zero labels of any kind with the anchor.
        shared_any = (mem_lbl * anchor_label).sum(dim=1)
        neg_mask = shared_any == 0

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        pos_idx = torch.where(pos_mask)[0]
        neg_idx = torch.where(neg_mask)[0]

        pos_d = dist[i][pos_idx]
        neg_d = dist[i][neg_idx]
        if pos_d.numel() == 0 or neg_d.numel() == 0:
            continue

        hardest_pos = mem_emb[pos_idx[torch.argmin(pos_d)]]
        hardest_neg = mem_emb[neg_idx[torch.argmax(neg_d)]]
        triplets.append((anchor, hardest_pos, hardest_neg))

    if len(triplets) == 0:
        return None

    anc, pos, neg = zip(*triplets)
    return (torch.stack(anc), torch.stack(pos), torch.stack(neg))










