"""Applicability checks for Markov-chain ideas on CXR-LT Task 1 (2026-09-26).

1. Label-graph Markov smoothing: blend cached val probs with one random-walk step
   over a row-stochastic label co-occurrence matrix (train fold), q = (1-a)p + a*(p_hat P)*sum(p),
   and report macro mAP vs the unsmoothed ensemble.
2. Patient structure of the internal split, using the ORIGINAL PadChest metadata read
   with dtype=str (data/CXRLT_2026_training_filtered.csv stores PatientID as a lossy float).
3. Normal-gating (CXR-LT 2026 Task 1 winner, arXiv 2602.13430):
   p_c <- p_c * (1 - p_Normal)^alpha for every abnormal class c.
4. Val subsets closer to the leaderboard test set (frontal only, patients unseen in train),
   using the reconstructed row order of the dumped probs (analysis/out/val_key_order.npy).

Report: docs/plan/2026-09-26-markov-applicability.md

Usage
-----
    python analysis/markov_check.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ens_select import macro_map  # noqa: E402
from torchmetrics.functional.classification import multilabel_average_precision  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROBS = ROOT / "analysis/out/probs_w01_flip"
ENSEMBLE = [  # greedy-3 from analysis/out/ens_select_w01_flip.json
    "push__img768_dp01_seed42__model_swa@768_flip.npy",
    "push__img1024_dp01_seed86__model_swa@1024_flip.npy",
    "res_sweep__img640_seed86__model_best@640_flip.npy",
]
LABEL_INFO = "/data/psytp7/wds_shards_train_raw/label_info.pt"  # shard/logit class order
PADCHEST = "/data/psytp7/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
KEY_ORDER = ROOT / "analysis/out/val_key_order.npy"
FRONTAL = ("PA", "AP", "AP_horizontal")
SUBSET_MODELS = {
    "greedy-3 ens": ENSEMBLE,
    "img1024 SWA": ["push__img1024_dp01_seed86__model_swa@1024_flip.npy"],
    "img768_s42 SWA": ["push__img768_dp01_seed42__model_swa@768_flip.npy"],
    "img896 SWA": ["push__img896_dp01_seed86__model_swa@896_flip.npy"],
    "res img768 s86": ["res_sweep__img768_seed86__model_best@768_flip.npy"],
    "img640 s86": ["res_sweep__img640_seed86__model_best@640_flip.npy"],
    "512 seed1024": ["seedsweep__seed_1024__model_best@512_flip.npy"],
    "512 dp01 s86": ["dp01__seed_86__model_best@512_flip.npy"],
}


def train_label_matrix(classes: list[str]) -> np.ndarray:
    tr = pd.read_csv(ROOT / "data/train_fold_1.csv")
    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(tr), len(classes)), np.float32)
    for r, labels in enumerate(tr["label_list"]):
        for c in ast.literal_eval(labels):
            if c in idx:
                Y[r, idx[c]] = 1
    return Y


def label_graph_smoothing() -> None:
    classes = list(torch.load(LABEL_INFO, weights_only=False)["class_names"])
    Y = train_label_matrix(classes)
    C = Y.T @ Y
    P = C / np.maximum(C.diagonal()[:, None], 1)  # P[i, j] = P(j | i)
    np.fill_diagonal(P, 0)
    P = P / np.maximum(P.sum(1, keepdims=True), 1e-9)  # row-stochastic transition matrix

    labels = torch.from_numpy(np.load(PROBS / "labels.npy")).long()
    p = np.mean([np.load(PROBS / n) for n in ENSEMBLE], 0).astype(np.float32)
    base = macro_map(p, labels)
    print(f"[1] label-graph smoothing | base ensemble mAP {base:.4f}")

    p_hat = p / np.maximum(p.sum(1, keepdims=True), 1e-9)
    step = (p_hat @ P) * p.sum(1, keepdims=True)
    for a in (0.02, 0.05, 0.10, 0.20):
        q = ((1 - a) * p + a * step).astype(np.float32)
        m = macro_map(q, labels)
        print(f"    alpha {a:.2f}  mAP {m:.4f}  ({m - base:+.4f})")

    q = ((1 - 0.05) * p + 0.05 * step).astype(np.float32)
    per = lambda z: multilabel_average_precision(  # noqa: E731
        torch.from_numpy(z), labels, num_labels=labels.shape[1], average="none").numpy()
    d = per(q) - per(p)
    order = np.argsort(Y.sum(0))  # by train frequency
    print(f"    alpha 0.05 per-class dAP: tail-10 {d[order[:10]].mean():+.4f} | "
          f"head-10 {d[order[-10:]].mean():+.4f}")


def patient_structure() -> None:
    meta = pd.read_csv(PADCHEST, usecols=["ImageID", "PatientID", "StudyID"],
                       dtype=str, low_memory=False).set_index("ImageID")
    folds = {k: pd.read_csv(ROOT / f"data/{k}_fold_1.csv") for k in ("train", "val")}
    pid = {k: meta["PatientID"].reindex(df["filename"]) for k, df in folds.items()}
    shared = set(pid["train"].dropna()) & set(pid["val"].dropna())
    print(f"[2] split | train pts {pid['train'].nunique()} val pts {pid['val'].nunique()} | "
          f"shared pts {len(shared)} | val images with a train-set patient "
          f"{pid['val'].isin(shared).mean():.3f}")
    for k, df in folds.items():
        st = meta["StudyID"].reindex(df["filename"])
        n = pd.DataFrame({"p": pid[k], "s": st}).dropna().groupby("p")["s"].nunique()
        multi = n[n >= 2].index
        print(f"    {k}: patients with >=2 studies {len(multi)}/{len(n)} | "
              f"images in them {pid[k].isin(multi).mean():.3f}")


def normal_gating() -> None:
    classes = list(torch.load(LABEL_INFO, weights_only=False)["class_names"])
    n0 = classes.index("Normal")
    labels = torch.from_numpy(np.load(PROBS / "labels.npy")).long()
    p = np.mean([np.load(PROBS / n) for n in ENSEMBLE], 0).astype(np.float32)
    base = macro_map(p, labels)
    print(f"[3] normal-gating | base ensemble mAP {base:.4f}")
    abnormal = np.arange(p.shape[1]) != n0
    for a in (0.25, 0.5, 1.0):
        q = p.copy()
        q[:, abnormal] = p[:, abnormal] * (1 - p[:, [n0]]) ** a
        m = macro_map(q, labels)
        print(f"    alpha {a:.2f}  mAP {m:.4f}  ({m - base:+.4f})")


def build_val_key_order(batch_size: int = 8, shard_size: int = 10000) -> np.ndarray:
    """Row of data/val_fold_1.csv behind each row of the dumped probs.

    evaluate_tta.py reads the 2 val shards with 4 DataLoader workers: worker 0 gets shard 0
    (CSV rows 0-9999), worker 1 gets shard 1 (rows 10000-), and batches alternate w0, w1.
    Rows missing from the shards (images absent at convert time) are unknown, so every split
    of the missing count between the shards is tried; exactly one reproduces labels.npy.
    """
    classes = list(torch.load(LABEL_INFO, weights_only=False)["class_names"])
    idx = {c: i for i, c in enumerate(classes)}
    va = pd.read_csv(ROOT / "data/val_fold_1.csv")
    Y = np.zeros((len(va), len(classes)), np.uint8)
    for r, labels in enumerate(va["label_list"]):
        for c in ast.literal_eval(labels):
            Y[r, idx[c]] = 1
    L = np.load(PROBS / "labels.npy")
    n_missing = len(Y) - len(L)

    def positions(n0, n1):
        pos0, pos1, i0, i1, p = [], [], 0, 0, 0
        while i0 < n0 or i1 < n1:
            for which, (i, n) in enumerate(((i0, n0), (i1, n1))):
                if i < n:
                    k = min(batch_size, n - i)
                    (pos0 if which == 0 else pos1).extend(range(p, p + k))
                    p += k
                    if which == 0:
                        i0 += k
                    else:
                        i1 += k
        return np.array(pos0), np.array(pos1)

    def align(rows, labs, max_skip):
        out, i, skipped = [], 0, 0
        for lab in labs:
            while i < len(rows) and not (Y[rows[i]] == lab).all():
                i, skipped = i + 1, skipped + 1
                if skipped > max_skip:
                    return None
            if i >= len(rows):
                return None
            out.append(rows[i])
            i += 1
        return np.array(out)

    fits = []
    for m0 in range(n_missing + 1):
        p0, p1 = positions(shard_size - m0, len(Y) - shard_size - (n_missing - m0))
        a0 = align(np.arange(0, shard_size), L[p0], m0)
        a1 = align(np.arange(shard_size, len(Y)), L[p1], n_missing - m0)
        if a0 is not None and a1 is not None:
            order = np.empty(len(L), int)
            order[p0], order[p1] = a0, a1
            if (Y[order] == L).all():
                fits.append(order)
    assert len(fits) == 1, f"expected exactly one consistent row order, got {len(fits)}"
    return fits[0]


def val_subsets() -> None:
    if KEY_ORDER.exists():
        order = np.load(KEY_ORDER)
    else:
        order = build_val_key_order()
        np.save(KEY_ORDER, order)
    va = pd.read_csv(ROOT / "data/val_fold_1.csv").iloc[order].reset_index(drop=True)
    meta = pd.read_csv(PADCHEST, usecols=["ImageID", "PatientID", "Projection"],
                       dtype=str, low_memory=False).set_index("ImageID")
    proj = meta["Projection"].reindex(va["filename"]).values
    pid = meta["PatientID"].reindex(va["filename"])
    tr = pd.read_csv(ROOT / "data/train_fold_1.csv")
    unseen = ~pid.isin(set(meta["PatientID"].reindex(tr["filename"]).dropna())).values
    front = np.isin(proj, FRONTAL)
    L = np.load(PROBS / "labels.npy")
    subsets = {"all": np.ones(len(L), bool), "frontal": front, "lateral": proj == "L",
               "frontal&seen": front & ~unseen, "frontal&unseen": front & unseen}

    def subset_map(P, mask):  # macro over classes with >=1 positive in the subset
        y = torch.from_numpy(L[mask]).long()
        keep = (y.sum(0) > 0).numpy()
        return macro_map(P[mask][:, keep], y[:, keep])

    print("[4] val subsets | n = " + ", ".join(f"{k} {int(m.sum())}" for k, m in subsets.items()))
    print(f"    {'model':16s}" + "".join(f"{k:>16s}" for k in subsets))
    for name, files in SUBSET_MODELS.items():
        P = np.mean([np.load(PROBS / f) for f in files], 0).astype(np.float32)
        print(f"    {name:16s}" + "".join(f"{subset_map(P, m):16.4f}" for m in subsets.values()))


if __name__ == "__main__":
    label_graph_smoothing()
    patient_structure()
    normal_gating()
    val_subsets()
