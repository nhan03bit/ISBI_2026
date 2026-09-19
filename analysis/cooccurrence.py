"""Co-occurrence-conditioned analysis for tail classes.

Answers the question the paper currently only asserts: does the proposed Stage-2
help *specifically* on rare findings that co-occur with common ones?

Two parts:
  1. Training co-occurrence structure -- for every non-head class, the fraction of
     its positive studies that also carry >=1 head finding (Normal / cardiomegaly /
     pleural effusion). Printed + saved; also a rare x head co-occurrence matrix.
  2. Conditioned ranking quality on the val set -- for each tail class, split its
     positive val samples into "co-occurs with a head finding" vs "isolated", and
     for each subset report (a) AP with only that subset marked positive (all
     negatives kept) and (b) the mean percentile rank of those positives. Compared
     across conditions (e.g. stage1 vs proposed) with deltas.

Usage
-----
    python analysis/cooccurrence.py \
        --cond stage1   train/checkpoint3/Model_20260119_062652/preds.res512.npz \
        --cond proposed train/checkpoint_Triplet_2/accum_6/seed_86/Model_2026082*/preds.npz \
        --out analysis/out/cooccurrence
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "evaluate"))
import class_groups  # noqa: E402

try:
    import torch
    from torchmetrics.classification import BinaryAveragePrecision
    _HAVE_TM = True
except Exception:  # noqa: BLE001
    _HAVE_TM = False


def _binary_ap(scores: np.ndarray, y: np.ndarray) -> float:
    if y.sum() == 0:
        return float("nan")
    if _HAVE_TM:
        m = BinaryAveragePrecision()
        m.update(torch.from_numpy(scores).float(), torch.from_numpy(y).int())
        return float(m.compute())
    order = np.argsort(-scores, kind="mergesort")
    ys = y[order].astype(bool)
    tp = np.cumsum(ys)
    fp = np.cumsum(~ys)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y.sum()
    rec_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - rec_prev) * precision))


def _percentile_rank(scores: np.ndarray, mask: np.ndarray) -> float:
    """Mean percentile rank (1 = top) of the masked samples within `scores`."""
    if mask.sum() == 0:
        return float("nan")
    ranks = scores.argsort().argsort()  # 0 = lowest score
    pct = ranks / (len(scores) - 1)
    return float(pct[mask].mean())


def _load(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    return d["probs"].astype(np.float64), d["labels"].astype(np.int64), list(d["class_names"])


def train_cooccurrence(names, groups):
    import pandas as pd

    df = pd.read_csv(class_groups.DEFAULT_TRAIN_CSV)
    mat = df[list(names)].to_numpy(dtype=np.int64)  # [N, C] shard order
    head_idx = np.where(groups["head"])[0]
    has_head = (mat[:, head_idx].sum(axis=1) > 0).astype(np.int64)

    rows = {}
    for i, nm in enumerate(names):
        if groups["head"][i]:
            continue
        pos = mat[:, i] == 1
        n_pos = int(pos.sum())
        if n_pos == 0:
            continue
        co = int((pos & (has_head == 1)).sum())
        rows[nm] = {
            "group": class_groups.group_of(mat[:, i].mean()),
            "train_pos": n_pos,
            "co_with_head": co,
            "co_rate": co / n_pos,
        }
    # rare x head matrix (tail rows, head cols): P(head present | tail present)
    tail_idx = np.where(groups["tail"] | groups["medium"])[0]
    mtx = np.zeros((len(tail_idx), len(head_idx)))
    for r, ti in enumerate(tail_idx):
        pos = mat[:, ti] == 1
        if pos.sum() == 0:
            continue
        for cix, hi in enumerate(head_idx):
            mtx[r, cix] = (mat[pos, hi] == 1).mean()
    matrix = {
        "rows": [names[j] for j in tail_idx],
        "cols": [names[j] for j in head_idx],
        "p_head_given_row": mtx.tolist(),
    }
    return rows, matrix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", action="append", metavar=("NAME", "NPZ"),
                    required=True, help="condition name + npz (single seed; pass the mean-seed npz or one seed)")
    ap.add_argument("--groups", nargs="+", default=["tail", "extreme_tail", "medium"],
                    help="which frequency groups to run the conditioned analysis on")
    ap.add_argument("--out", default="analysis/out/cooccurrence")
    args = ap.parse_args()

    meta = class_groups.build()
    names = meta["class_names"]
    groups = meta["groups"]
    prev = meta["train_prevalence"]
    head_idx = np.where(groups["head"])[0]
    normal_idx = names.index("Normal") if "Normal" in names else -1

    co_rows, co_matrix = train_cooccurrence(names, groups)
    print("train co-occurrence with a head finding (Normal/cardiomegaly/pleural effusion):")
    for nm, r in sorted(co_rows.items(), key=lambda kv: kv[1]["co_rate"], reverse=True):
        print(f"  {nm:<32} {r['group']:<7} pos={r['train_pos']:>6}  co_rate={r['co_rate']:.3f}")

    report = {"train_cooccurrence": co_rows, "train_cooccurrence_matrix": co_matrix,
              "conditions": {}}

    target_idx = sorted(set(
        i for g in args.groups for i in np.where(groups[g])[0]
    ))

    for entry in args.cond:
        name, paths = entry[0], [Path(p) for p in entry[1:]]
        probs, labels, npz_names = _load(paths[0])
        if npz_names != names:
            raise SystemExit(f"{paths[0]}: class order mismatch")
        val_has_head = (labels[:, head_idx].sum(axis=1) > 0)
        # count of *other* findings present (excluding Normal and the class itself)
        finding_cols = [j for j in range(len(names)) if j != normal_idx]
        n_findings = labels[:, finding_cols].sum(axis=1)

        per_class = {}
        for i in target_idx:
            nm = names[i]
            pos = labels[:, i] == 1
            self_pos_in_findings = 1 if i != normal_idx else 0
            n_other = n_findings - (pos.astype(int) * self_pos_in_findings)
            co_mask = pos & (n_other > 0)          # co-occurs with >=1 other finding
            iso_mask = pos & (n_other == 0)        # this finding only
            cohead_mask = pos & val_has_head       # co-occurs with a head finding
            scores = probs[:, i]

            def subset_ap(sub_mask):
                y = np.zeros(len(scores), dtype=np.int64)
                y[sub_mask] = 1
                return _binary_ap(scores, y)

            per_class[nm] = {
                "val_pos": int(pos.sum()),
                "val_pos_co": int(co_mask.sum()),
                "val_pos_isolated": int(iso_mask.sum()),
                "val_pos_co_head": int(cohead_mask.sum()),
                "ap_all": subset_ap(pos),
                "ap_co": subset_ap(co_mask),
                "ap_isolated": subset_ap(iso_mask),
                "ap_co_head": subset_ap(cohead_mask),
                "pctrank_co": _percentile_rank(scores, co_mask),
                "pctrank_isolated": _percentile_rank(scores, iso_mask),
            }
        report["conditions"][name] = {"npz": str(paths[0]), "per_class": per_class}

        print(f"\n=== {name} ===  (co = co-occurs with >=1 other finding; iso = finding alone)")
        print(f"{'class':<30} {'n_co':>5} {'n_iso':>5} {'AP_co':>7} {'AP_iso':>7} {'pr_co':>6} {'pr_iso':>6}")
        for nm, d in per_class.items():
            print(f"{nm:<30} {d['val_pos_co']:>5} {d['val_pos_isolated']:>5} "
                  f"{d['ap_co']:>7.3f} {d['ap_isolated']:>7.3f} "
                  f"{d['pctrank_co']:>6.3f} {d['pctrank_isolated']:>6.3f}")

    conds = list(report["conditions"])
    if len(conds) >= 2:
        base, other = conds[0], conds[1]
        print(f"\n=== delta  {other} - {base}  (AP on co-occurring vs isolated positives) ===")
        deltas = {}
        for nm in report["conditions"][base]["per_class"]:
            b = report["conditions"][base]["per_class"][nm]
            o = report["conditions"][other]["per_class"][nm]
            deltas[nm] = {
                "d_ap_co": o["ap_co"] - b["ap_co"],
                "d_ap_isolated": o["ap_isolated"] - b["ap_isolated"],
                "d_ap_co_head": o["ap_co_head"] - b["ap_co_head"],
                "d_ap_all": o["ap_all"] - b["ap_all"],
            }
            print(f"  {nm:<30} co {deltas[nm]['d_ap_co']:+.3f}   "
                  f"iso {deltas[nm]['d_ap_isolated']:+.3f}")
        report["delta"] = {"baseline": base, "other": other, "per_class": deltas}
        for key in ("d_ap_co", "d_ap_isolated", "d_ap_co_head", "d_ap_all"):
            vals = [d[key] for d in deltas.values() if np.isfinite(d[key])]
            report["delta"][f"mean_{key}"] = float(np.mean(vals)) if vals else float("nan")
        print(f"\n  mean delta AP  co-occurring: {report['delta']['mean_d_ap_co']:+.4f}   "
              f"isolated: {report['delta']['mean_d_ap_isolated']:+.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
