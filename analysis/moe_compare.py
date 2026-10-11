"""Matched-recipe comparison: Triplet Memory vs rebuilt Mixture of Experts vs ASL-only, with and
without the patient-history Markov prior (plan: ~/.claude/plans/purring-wandering-quiche.md).

Reads flip-TTA probability dumps (one row order, checked through labels.npy and keys.txt):
  Triplet Memory      analysis/out/probs_w01_flip (push / res_sweep), probs_v5_{orig_recheck,rerun}_flip
  MoE / ASL / MoE+M   analysis/out/probs_moe_flip (scripts/evaluate_tta_v5.sh --dump-probs)
Aggregates mirror Table III with FIXED members (no re-selection on val):
  2-seed 768 px (s42 + s1024), fixed 4-model (768 s42, 768 s1024, 896 s86, 1024 s86),
  greedy-member trio (768 s42 + 1024 s86 + 640 s86 best epoch), Markov 2-seed with / zero.
Point metrics use torchmetrics with the evaluate.py definitions (macro mAP, macro AUROC, macro F1
at 0.5, 15-bin L1 ECE on flattened probs). Paired image-level bootstrap CIs use the fast weighted
re-implementation from analysis/moe_anchor_boot.py, checked against torchmetrics on every arm.

    python -I analysis/moe_compare.py [B=2000] [out.json]
Missing dumps are reported and their arms skipped, so the script can run as results arrive.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchmetrics.classification import (BinaryCalibrationError, MultilabelAUROC,
                                         MultilabelAveragePrecision, MultilabelF1Score)

O = Path("/home/psytp7/ISBI_2026/analysis/out")
W01, OR, RR, MOE = (O / "probs_w01_flip", O / "probs_v5_orig_recheck_flip",
                    O / "probs_v5_rerun_flip", O / "probs_moe_flip")
B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else O / "moe_compare.json"

L = np.load(W01 / "labels.npy").astype(np.uint8)
N, C = L.shape
for d in (OR, RR, MOE):
    if (d / "labels.npy").exists():
        assert (np.load(d / "labels.npy") == L).all(), f"row order differs: {d}"
if (MOE / "keys.txt").exists() and (OR / "keys.txt").exists():
    assert (MOE / "keys.txt").read_text() == (OR / "keys.txt").read_text(), "keys.txt differs"

# ---- member files ------------------------------------------------------------------------
def tm(f): return W01 / f
def mo(run, px, seed, stem="model_swa", tag=""):
    return MOE / f"checkpoint_MoE__{run}_img{px}_seed{seed}__{stem}@{px}_flip{tag}.npy"
def v5(d, sfx, seed, tag):
    return d / f"prior__img768_qprior_seed{seed}{sfx}__model_swa@768_flip_prior-{tag}.npy"

FILES = {
    "TM 768 s42": tm("push__img768_dp01_seed42__model_swa@768_flip.npy"),
    "TM 768 s1024": tm("push__img768_dp01_seed1024__model_swa@768_flip.npy"),
    "TM 896 s86": tm("push__img896_dp01_seed86__model_swa@896_flip.npy"),
    "TM 1024 s86": tm("push__img1024_dp01_seed86__model_swa@1024_flip.npy"),
    "TM 640 s86 (best)": tm("res_sweep__img640_seed86__model_best@640_flip.npy"),
    "MoE 768 s42": mo("moe", 768, 42), "MoE 768 s1024": mo("moe", 768, 1024),
    "MoE 896 s86": mo("moe", 896, 86), "MoE 1024 s86": mo("moe", 1024, 86),
    "MoE 640 s86 (best)": mo("moe", 640, 86, stem="model_best"),
    "ASL 768 s42": mo("asl", 768, 42), "ASL 768 s1024": mo("asl", 768, 1024),
}
for tag in ("with", "zero"):
    FILES[f"TM+M orig 768 s42 {tag}"] = v5(OR, "", 42, tag)
    FILES[f"TM+M orig 768 s1024 {tag}"] = v5(OR, "", 1024, tag)
    FILES[f"TM+M rerun 768 s42 {tag}"] = v5(RR, "_rerun", 42, tag)
    FILES[f"TM+M rerun 768 s1024 {tag}"] = v5(RR, "_rerun", 1024, tag)
    FILES[f"MoE+M 768 s42 {tag}"] = mo("moeprior", 768, 42, tag=f"_prior-{tag}")
    FILES[f"MoE+M 768 s1024 {tag}"] = mo("moeprior", 768, 1024, tag=f"_prior-{tag}")

P = {}
for k, f in FILES.items():
    if f.exists():
        P[k] = np.load(f).astype(np.float32)
    else:
        print(f"missing: {k:28s} {f.name}")

ENS = {
    "TM 2-seed 768": ["TM 768 s42", "TM 768 s1024"],
    "MoE 2-seed 768": ["MoE 768 s42", "MoE 768 s1024"],
    "ASL 2-seed 768": ["ASL 768 s42", "ASL 768 s1024"],
    "TM fixed 4-model": ["TM 768 s42", "TM 768 s1024", "TM 896 s86", "TM 1024 s86"],
    "MoE fixed 4-model": ["MoE 768 s42", "MoE 768 s1024", "MoE 896 s86", "MoE 1024 s86"],
    "TM trio (greedy members)": ["TM 768 s42", "TM 1024 s86", "TM 640 s86 (best)"],
    "MoE trio (same members)": ["MoE 768 s42", "MoE 1024 s86", "MoE 640 s86 (best)"],
}
for tag in ("with", "zero"):
    ENS[f"TM+M orig 2-seed {tag}"] = [f"TM+M orig 768 s42 {tag}", f"TM+M orig 768 s1024 {tag}"]
    ENS[f"TM+M rerun 2-seed {tag}"] = [f"TM+M rerun 768 s42 {tag}", f"TM+M rerun 768 s1024 {tag}"]
    ENS[f"MoE+M 2-seed {tag}"] = [f"MoE+M 768 s42 {tag}", f"MoE+M 768 s1024 {tag}"]

ARMS = dict(P)
for k, members in ENS.items():
    if all(m in P for m in members):
        ARMS[k] = np.mean([P[m] for m in members], axis=0)

# ---- point metrics (torchmetrics) ------------------------------------------------------------
Lt = torch.from_numpy(L).long()
def tm_metrics(p):
    pt = torch.from_numpy(np.ascontiguousarray(p))
    return dict(mAP=float(MultilabelAveragePrecision(num_labels=C)(pt, Lt)),
                mAUC=float(MultilabelAUROC(num_labels=C)(pt, Lt)),
                mF1=float(MultilabelF1Score(num_labels=C, average="macro", threshold=0.5)(pt, Lt)),
                mECE=float(BinaryCalibrationError(n_bins=15, norm="l1")(pt.flatten(), Lt.flatten())))

point = {k: tm_metrics(p) for k, p in ARMS.items()}
print(f"\n{'arm':32s} {'mAP':>7s} {'mAUC':>7s} {'mF1':>7s} {'mECE':>7s}")
for k, m in point.items():
    print(f"{k:32s} {m['mAP']:7.4f} {m['mAUC']:7.4f} {m['mF1']:7.4f} {m['mECE']:7.4f}")

# ---- fast weighted metrics for the bootstrap (from analysis/moe_anchor_boot.py) ---------------
class Pre:
    def __init__(self, p):
        self.P = p
        self.order = np.argsort(p, axis=0, kind="stable")
        self.grp = []
        for c in range(C):
            ps = p[self.order[:, c], c]
            self.grp.append(np.cumsum(np.r_[True, ps[1:] != ps[:-1]]) - 1)
        self.pred = p >= 0.5
        self.bin = np.clip(np.searchsorted(np.linspace(0, 1, 16), p, side="left") - 1, 0, 14)

def weighted(pre, w):
    auc, ap, f1 = np.full(C, np.nan), np.full(C, np.nan), np.zeros(C)
    for c in range(C):
        o = pre.order[:, c]; y = L[o, c]; ww = w[o]
        wp = ww * y; wn = ww - wp; Wp, Wn = wp.sum(), wn.sum()
        if Wp > 0 and Wn > 0:
            g = pre.grp[c]; ng = g[-1] + 1
            pg = np.bincount(g, wp, ng); ngr = np.bincount(g, wn, ng)
            auc[c] = (pg * (np.cumsum(ngr) - ngr + 0.5 * ngr)).sum() / (Wp * Wn)
            pr, nr = pg[::-1], ngr[::-1]
            tp, fp = np.cumsum(pr), np.cumsum(nr)
            ap[c] = (pr * tp / np.maximum(tp + fp, 1e-12)).sum() / Wp
        prd = pre.pred[:, c]; yy = L[:, c].astype(bool)
        tp_, fp_, fn_ = (w * (prd & yy)).sum(), (w * (prd & ~yy)).sum(), (w * (~prd & yy)).sum()
        d = 2 * tp_ + fp_ + fn_
        f1[c] = 2 * tp_ / d if d > 0 else 0.0
    ww = np.repeat(w[:, None], C, 1).ravel()
    b = pre.bin.ravel()
    ece = np.abs(np.bincount(b, ww * L.ravel(), 15) - np.bincount(b, ww * pre.P.ravel(), 15)).sum() / (w.sum() * C)
    return dict(mAP=np.nanmean(ap), mAUC=np.nanmean(auc), mF1=f1.mean(), mECE=ece)

PAIRS = [
    ("TM 2-seed 768", "MoE 2-seed 768"), ("TM 2-seed 768", "ASL 2-seed 768"),
    ("MoE 2-seed 768", "ASL 2-seed 768"),
    ("TM fixed 4-model", "MoE fixed 4-model"), ("TM trio (greedy members)", "MoE trio (same members)"),
    ("MoE+M 2-seed with", "MoE 2-seed 768"), ("MoE+M 2-seed zero", "MoE 2-seed 768"),
    ("TM+M orig 2-seed with", "TM 2-seed 768"), ("TM+M orig 2-seed zero", "TM 2-seed 768"),
    ("TM+M rerun 2-seed with", "TM 2-seed 768"), ("TM+M rerun 2-seed zero", "TM 2-seed 768"),
    ("TM+M orig 2-seed with", "MoE+M 2-seed with"), ("TM+M orig 2-seed zero", "MoE+M 2-seed zero"),
    ("TM+M rerun 2-seed with", "MoE+M 2-seed with"), ("TM+M rerun 2-seed zero", "MoE+M 2-seed zero"),
]
PAIRS = [(a, b) for a, b in PAIRS if a in ARMS and b in ARMS]
need = sorted({k for ab in PAIRS for k in ab})
pre = {k: Pre(ARMS[k]) for k in need}
ones = np.ones(N)
for k in need:  # the fast metrics must reproduce torchmetrics before they are trusted
    m = weighted(pre[k], ones)
    gap = max(abs(m[a] - point[k][a]) for a in m)
    assert gap < 1e-4, f"fast metrics disagree with torchmetrics on {k}: {gap}"

rng = np.random.default_rng(2026)
W = [np.bincount(rng.integers(0, N, N), minlength=N).astype(np.float64) for _ in range(B)]
cache, out = {}, {"B": B, "point": point, "missing": [k for k in FILES if k not in P], "pairs": {}}
t0 = time.time()
print(f"\npaired bootstrap, B={B}, 95% percentile CI")
for a, b in PAIRS:
    for k in (a, b):
        if k not in cache:
            cache[k] = [weighted(pre[k], w) for w in W]
    r = {}
    for met in ("mAP", "mAUC", "mF1", "mECE"):
        d = np.array([x[met] - y[met] for x, y in zip(cache[a], cache[b])])
        r[met] = dict(point=point[a][met] - point[b][met],
                      ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                      frac_le0=float((d <= 0).mean()))
    out["pairs"][f"{a} - {b}"] = r
    print(f"{a} - {b}  ({time.time() - t0:.0f}s)")
    for met, v in r.items():
        print(f"    {met}: {v['point']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
OUT.write_text(json.dumps(out, indent=1))
print(f"\nwrote {OUT}")
