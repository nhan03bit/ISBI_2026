"""Shared helpers for analysis/entropy_best_config.py and analysis/entropy_edge_cases.py (2026-10-17).

Read-only. Inputs: cached flip-TTA probability dumps (all share the row order of
analysis/out/probs_unified_flip/keys.txt), labels.npy, and the PadChest columns ImageID / PatientID / Projection of
internal train/val-fold images (never Report / Labels, never dev / test images).

  FastAPM        tie-grouped weighted AP for several weight columns at once (equal to
                 analysis/entropy_ablation.FastAP and torchmetrics at 0/1 weights; checked by the callers)
  load_context   val keys, labels, class meta, and the image masks used by both analyses
  boot_replicates  paired image / patient-cluster bootstrap, per-class AP for every run and mask, multiprocessing
  holm, boot_summary, mde            small statistics helpers
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entropy_ablation as ea  # noqa: E402  (FastAP, tm_metrics, paired, summarize, class_groups)

warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmean of an all-NaN slice (class absent from a resample)

ROOT = ea.ROOT
O = ROOT / "analysis/out"
PADCHEST_CSV = ROOT / "data/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
FRONTAL = ("PA", "AP", "AP_horizontal")
# weight columns shared by every bootstrap: the image masks of both analyses
MASK_NAMES = ("all", "frontal", "frontal_unseen", "lateral", "aph", "abnormal", "ge3")
MIDX = {m: i for i, m in enumerate(MASK_NAMES)}


class FastAPM:
    """Per-class weighted AP for M weight columns at once. Same formula as entropy_ablation.FastAP: images sorted by
    score (descending), ties grouped, AP = sum_g pos_g * TP_g / (TP_g + FP_g) / total_pos with cumulative weights."""

    def __init__(self, P: np.ndarray, L: np.ndarray):
        self.N, self.C = P.shape
        self.o, self.y, self.starts = [], [], []
        for c in range(self.C):
            o = np.argsort(-P[:, c], kind="stable")
            ps = P[o, c]
            self.o.append(o)
            self.y.append(L[o, c].astype(np.float64))
            self.starts.append(np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]]))

    def __call__(self, W: np.ndarray) -> np.ndarray:
        """W (N, M) non-negative weights -> (M, C) AP, NaN where a class has no positive or no negative weight."""
        out = np.full((W.shape[1], self.C), np.nan)
        for c in range(self.C):
            ww = W[self.o[c]]
            wp = ww * self.y[c][:, None]
            pg = np.add.reduceat(wp, self.starts[c], axis=0)
            tg = np.add.reduceat(ww, self.starts[c], axis=0)
            tp = np.cumsum(pg, axis=0)
            allw = np.cumsum(tg, axis=0)
            Wp = tp[-1]
            Wn = allw[-1] - Wp
            ap = (pg * tp / np.maximum(allw, 1e-12)).sum(0) / np.maximum(Wp, 1e-12)
            out[:, c] = np.where((Wp > 0) & (Wn > 0), ap, np.nan)
        return out


def load_context() -> dict:
    """Val keys, labels, class metadata and the image masks."""
    keys = (O / "probs_unified_flip/keys.txt").read_text().split("\n")
    L = np.load(O / "probs_unified_flip/labels.npy").astype(np.int64)
    assert len(keys) == len(L) == 17270
    meta = ea.class_groups.build()
    names = list(meta["class_names"])
    prev = meta["train_prevalence"]
    groups = {g: np.where(meta["groups"][g])[0] for g in ("head", "medium", "tail", "extreme_tail")}
    groups["rare10"] = np.argsort(prev)[:10]
    # tail classes rarest first (the order the per-class tables use)
    tail_sorted = [int(i) for i in np.argsort(prev) if meta["groups"]["tail"][i]]

    pc = pd.read_csv(PADCHEST_CSV, usecols=["ImageID", "PatientID", "Projection"], dtype=str, low_memory=False) \
        .drop_duplicates("ImageID").set_index("ImageID")
    proj = pc["Projection"].reindex(keys)
    pid = pc["PatientID"].reindex(keys)
    assert proj.notna().all() and pid.notna().all(), "val image missing from the PadChest metadata"
    tr = pd.read_csv(ROOT / "data/train_fold_1.csv")
    tr_pid = pc["PatientID"].reindex(tr["filename"])
    assert tr_pid.notna().all()
    unseen = ~pid.isin(set(tr_pid)).values
    fr = proj.isin(FRONTAL).values
    ni = names.index("Normal")
    non_normal = np.delete(L, ni, axis=1).sum(1)
    masks = {
        "all": np.ones(len(L), bool),
        "frontal": fr,
        "frontal_unseen": fr & unseen,
        "lateral": (proj == "L").values,
        "aph": (proj == "AP_horizontal").values,
        "abnormal": non_normal >= 1,
        "ge3": non_normal >= 3,
    }
    codes, uniq = pd.factorize(pid.values)
    return dict(keys=keys, L=L, meta=meta, names=names, prev=prev, groups=groups, tail_sorted=tail_sorted,
                masks=masks, mmat=np.stack([masks[m] for m in MASK_NAMES], 1).astype(np.float64),
                normal_idx=ni, unseen=unseen, proj=proj.values, pid_codes=codes, n_patients=len(uniq))


def subset_values(ap: np.ndarray, ctx: dict) -> dict:
    """The metrics of both analyses from one run's (M, C) per-class AP matrix (M = len(MASK_NAMES))."""
    nm, ni, g = np.nanmean, ctx["normal_idx"], ctx["groups"]
    a = ap[MIDX["abnormal"]].copy()
    a[ni] = np.nan
    t = ap[MIDX["ge3"]].copy()
    t[ni] = np.nan
    with np.errstate(all="ignore"):
        return {
            "ALL": nm(ap[MIDX["all"]]), "S1": nm(ap[MIDX["frontal"]]), "S2": nm(ap[MIDX["frontal_unseen"]]),
            "S3": nm(ap[MIDX["lateral"]]), "S4": nm(ap[MIDX["aph"]]), "S5": nm(a), "S6": nm(t),
            "S7": nm(ap[MIDX["all"]][g["tail"]]), "S8": nm(ap[MIDX["frontal"]][g["tail"]]),
            "S9": nm(ap[MIDX["all"]][g["rare10"]]), "S10": nm(ap[MIDX["all"]][g["extreme_tail"]]),
            **{f"C:{ctx['names'][i]}": ap[MIDX["all"]][i] for i in ctx["tail_sorted"]},
        }


# ---- bootstrap ---------------------------------------------------------------------------------------------
_G: dict = {}


def _rep_chunk(args):
    b0, b1 = args
    fapm, mmat, N = _G["fapm"], _G["mmat"], _G["N"]
    out = np.empty((b1 - b0, len(fapm), mmat.shape[1], fapm[0].C), np.float32)
    for k, b in enumerate(range(b0, b1)):
        rng = np.random.default_rng([_G["seed"], b])
        if _G["cluster"] is None:
            w = np.bincount(rng.integers(0, N, N), minlength=N).astype(np.float64)
        else:
            codes, P = _G["cluster"]
            w = np.bincount(rng.integers(0, P, P), minlength=P).astype(np.float64)[codes]
        W = w[:, None] * mmat
        for r, fa in enumerate(fapm):
            out[k, r] = fa(W)
    return out


def boot_replicates(fapm: list, mmat: np.ndarray, B: int, seed: int, workers: int, cluster=None) -> np.ndarray:
    """(B, n_runs, M, C) per-class AP of every run under B shared bootstrap resamples (images, or patients if
    `cluster` = (codes, n_clusters)). Replicate b always uses default_rng([seed, b]), so the result does not depend
    on the number of workers."""
    _G.update(fapm=fapm, mmat=mmat, N=mmat.shape[0], seed=seed, cluster=cluster)
    chunk = max(1, B // (workers * 4))
    jobs = [(b, min(B, b + chunk)) for b in range(0, B, chunk)]
    if workers <= 1:
        res = [_rep_chunk(j) for j in jobs]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_rep_chunk, jobs)
    return np.concatenate(res, 0)


# ---- statistics --------------------------------------------------------------------------------------------
def holm(p: dict) -> dict:
    """Holm step-down adjusted p-values for {name: p}."""
    names = sorted(p, key=lambda k: p[k])
    m, adj, run = len(names), {}, 0.0
    for i, k in enumerate(names):
        run = max(run, min(1.0, (m - i) * p[k]))
        adj[k] = run
    return adj


def boot_summary(dstar: np.ndarray) -> dict:
    """Percentile 95% interval and two-sided bootstrap p (null 0, +1 correction) of bootstrap draws (NaN dropped)."""
    d = np.asarray(dstar, float)
    d = d[~np.isnan(d)]
    B = len(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((np.sum(d <= 0) + 1) / (B + 1), (np.sum(d >= 0) + 1) / (B + 1)))
    return {"ci": [float(lo), float(hi)], "p": float(p), "B": int(B), "sd": float(d.std(ddof=1))}


def mde80(sd: float, n: int) -> float:
    """Smallest true paired mean difference a two-sided 5% paired t-test detects with 80% power at this sd."""
    return float((stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * sd / np.sqrt(n))


def ece_l1(P: np.ndarray, L: np.ndarray, w: np.ndarray | None = None, nb: int = 15) -> float:
    """15-bin L1 calibration error on all probabilities (torchmetrics BinaryCalibrationError(norm='l1') for unit
    weights), with optional per-image weights."""
    b = np.minimum((P * nb).astype(np.int64), nb - 1).ravel()
    ww = np.ones(P.size) if w is None else np.repeat(w, P.shape[1])
    sp = np.bincount(b, ww * P.ravel(), nb)
    sy = np.bincount(b, ww * L.ravel(), nb)
    return float(np.abs(sp - sy).sum() / ww.sum())
