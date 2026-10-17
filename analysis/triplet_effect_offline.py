"""Offline (CPU) checks of how far the Triplet Memory term moved the trained model, vs noise references.

Plan: docs/plan/2026-10-15-triplet-memory-debug-plan.md (follow-up: why is the triplet gradient mitigated).
  1. weight space: per-module ||theta_X - theta_Y|| for seeded ASL / Entropy c / probe p, and the
     data-order reference (ASL seeded vs ASL unseeded, same seed 42, different sample stream)
  2. prediction space: per-class logit correlation / mean |delta logit| on the internal val split
     (flip-TTA SWA dumps), same references
  3. the frozen ML-Decoder query embedding: its mean over the 30 queries is an image-independent
     vector that every mean-pooled triplet embedding contains
"""
import os
import sys
from collections import OrderedDict

import numpy as np
import torch

ROOT = "/home/psytp7/ISBI_2026"
T = f"{ROOT}/train"
CK = OrderedDict([
    ("stage1", f"{T}/checkpoint3/Model_20260119_062652/model_best.pth"),
    ("asl_s", f"{T}/checkpoint_ASL/v0_768_seeded/seed_42/Model_run/model_swa.pth"),
    ("c_s", f"{T}/checkpoint_Triplet_3/unified768_seeded/c/seed_42/Model_run/model_swa.pth"),
    ("p_s", f"{T}/checkpoint_Triplet_3/unified768_seeded/p/seed_42/Model_run/model_swa.pth"),
    ("asl_u", f"{T}/checkpoint_ASL/v0_768/seed_42/Model_run/model_swa.pth"),
    ("asl_u123", f"{T}/checkpoint_ASL/v0_768/seed_123/Model_run/model_swa.pth"),
    ("c_u", f"{T}/checkpoint_Triplet_3/unified768/c/seed_42/Model_run/model_swa.pth"),
])
O = f"{ROOT}/analysis/out"
DUMPS = OrderedDict([
    ("asl_s", f"{O}/probs_v0_flip_seeded/v0_768_seeded__seed_42__model_swa@768_flip.npy"),
    ("c_s", f"{O}/probs_unified_flip_seeded/c__seed_42__model_swa@768_flip.npy"),
    ("p_s", f"{O}/probs_unified_flip_seeded/p__seed_42__model_swa@768_flip.npy"),
    ("asl_u", f"{O}/probs_v0_flip/v0_768__seed_42__model_swa@768_flip.npy"),
    ("asl_u123", f"{O}/probs_v0_flip/v0_768__seed_123__model_swa@768_flip.npy"),
    ("c_u", f"{O}/probs_unified_flip/c__seed_42__model_swa@768_flip.npy"),
])
KEYS = {"asl_s": f"{O}/probs_v0_flip_seeded/keys.txt", "c_s": f"{O}/probs_unified_flip_seeded/keys.txt",
        "p_s": f"{O}/probs_unified_flip_seeded/keys.txt", "asl_u": f"{O}/probs_v0_flip/keys.txt",
        "asl_u123": f"{O}/probs_v0_flip/keys.txt", "c_u": f"{O}/probs_unified_flip/keys.txt"}
PAIRS = [("asl_s", "c_s", "triplet lam 0.1, same stream"),
         ("asl_s", "p_s", "triplet lam 1.0, same stream"),
         ("c_s", "p_s", "lam 0.1 vs 1.0, same stream"),
         ("asl_s", "asl_u", "REF data order only (ASL, seed 42)"),
         ("asl_u", "asl_u123", "REF seed 42 vs 123 (ASL, unseeded)"),
         ("asl_u", "c_u", "triplet lam 0.1, unseeded s42 pair")]


def module_of(k):
    if k.startswith("head."):
        parts = k.split(".")
        return "head." + (parts[1] if parts[1] != "decoder_layers" else f"decoder_layers.{parts[2]}")
    if k.startswith(("stages.", "downsample_layers.")):
        return "backbone.s" + k.split(".")[1]
    return "other"


def weight_space():
    sd = {}
    for n, p in CK.items():
        if os.path.exists(p):
            sd[n] = {k: v.float() for k, v in torch.load(p, map_location="cpu").items() if v.is_floating_point()}
        else:
            print(f"[skip] {n}: {p} missing")
    s1 = sd["stage1"]
    keys = [k for k in sd["asl_s"] if k in s1]
    mods = sorted({module_of(k) for k in keys})

    def dist(a, b, ks):
        return float(sum(((sd[a][k] - sd[b][k]) ** 2).sum() for k in ks)) ** 0.5

    def dist_s1(a, ks):
        return float(sum(((sd[a][k] - s1[k]) ** 2).sum() for k in ks)) ** 0.5

    print("\n## Weight space (SWA weights). ratio = ||X - Y|| / ||ASL_seeded - Stage1|| per module")
    hdr = "| module | ||asl_s-stage1|| | " + " | ".join(f"{a} vs {b}" for a, b, _ in PAIRS if a in sd and b in sd) + " |"
    print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
    for m in ["ALL"] + mods:
        ks = keys if m == "ALL" else [k for k in keys if module_of(k) == m]
        base = dist_s1("asl_s", ks)
        cells = [f"{dist(a, b, ks) / max(base, 1e-12):.3f}" for a, b, _ in PAIRS if a in sd and b in sd]
        print(f"| {m} | {base:.3f} | " + " | ".join(cells) + " |")
    print("pairs:", "; ".join(f"{a} vs {b} = {d}" for a, b, d in PAIRS))

    # Frozen ML-Decoder queries: identical in every checkpoint (requires_grad False)?
    q = {n: s["head.query_embed.weight"] for n, s in sd.items() if "head.query_embed.weight" in s}
    print("\n## Frozen query embedding")
    for n, w in q.items():
        print(f"{n}: max|q - q_stage1| = {float((w - q['stage1']).abs().max()):.2e}")
    w = q["stage1"]
    mean_q = w.mean(0)
    print(f"Q={w.shape[0]} D={w.shape[1]} ||mean_c q_c|| = {float(mean_q.norm()):.3f}, "
          f"mean ||q_c|| = {float(w.norm(dim=1).mean()):.3f}, "
          f"mean pairwise cos(q_c, q_c') = {float((torch.nn.functional.normalize(w, dim=1) @ torch.nn.functional.normalize(w, dim=1).T).fill_diagonal_(0).sum() / (w.shape[0] * (w.shape[0] - 1))):.3f}")


def logit(p):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def pred_space():
    P, K = {}, {}
    for n, p in DUMPS.items():
        if os.path.exists(p):
            P[n] = logit(np.load(p)); K[n] = open(KEYS[n]).read().split()
        else:
            print(f"[skip] {n}: {p} missing")
    ref = next(iter(K.values()))
    for n in K:
        if K[n] != ref:
            order = {k: i for i, k in enumerate(K[n])}
            P[n] = P[n][[order[k] for k in ref]]
    L = np.load(f"{O}/probs_v0_flip_seeded/labels.npy")
    print("\n## Prediction space (val, flip-TTA SWA). per-class means over 30 classes")
    print("| pair | meaning | logit corr | mean |dlogit| | |dlogit| / logit sd | top-1 agree | pos-rank corr (tail) |")
    print("|---|---|---|---|---|---|---|")
    from scipy.stats import spearmanr
    for a, b, d in PAIRS:
        if a not in P or b not in P:
            continue
        x, y = P[a], P[b]
        corr = np.mean([np.corrcoef(x[:, c], y[:, c])[0, 1] for c in range(30)])
        dl = np.abs(x - y).mean()
        rel = np.mean([np.abs(x[:, c] - y[:, c]).mean() / x[:, c].std() for c in range(30)])
        top1 = float((x.argmax(1) == y.argmax(1)).mean())
        # ranking of positives among all images, averaged over classes with < 200 positives
        rc = [spearmanr(x[:, c], y[:, c]).correlation for c in range(30) if 0 < L[:, c].sum() < 200]
        print(f"| {a} vs {b} | {d} | {corr:.4f} | {dl:.3f} | {rel:.3f} | {top1:.3f} | {np.mean(rc):.4f} |")


if __name__ == "__main__":
    weight_space()
    pred_space()
