"""Per-class AP + head/medium/tail breakdown from saved eval predictions.

Consumes the ``preds*.npz`` files written by ``evaluate/evaluate.py --save-preds``
(keys: ``probs`` [N,C] float, ``labels`` [N,C] int8, ``class_names`` [C] str).
Each "condition" may point at several npz files (one per seed); the script reports
per-seed metrics and their mean +/- sample std, plus a bootstrap CI over
validation samples for the first seed (the tail-measurability check).

Metric = torchmetrics ``MultilabelAveragePrecision`` (same as training/eval), so
the macro number here reproduces the paper's mAP exactly.

Example
-------
    python analysis/per_class_report.py \
        --cond stage1   train/checkpoint3/Model_20260119_062652/preds.res512.npz \
        --cond proposed train/checkpoint_Triplet_2/accum_6/seed_*/Model_2026082*/preds.npz \
        --baseline stage1 \
        --out analysis/out/perclass
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
    from torchmetrics.classification import MultilabelAveragePrecision, MultilabelAUROC
    _HAVE_TM = True
except Exception:  # noqa: BLE001
    _HAVE_TM = False


def _ap_per_class(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """[C] per-class average precision."""
    c = probs.shape[1]
    if _HAVE_TM:
        m = MultilabelAveragePrecision(num_labels=c, average=None)
        m.update(torch.from_numpy(probs).float(), torch.from_numpy(labels).int())
        return m.compute().numpy()
    # numpy fallback: exact step-function AP (sklearn-equivalent)
    out = np.full(c, np.nan)
    for j in range(c):
        y = labels[:, j].astype(bool)
        if y.sum() == 0:
            continue
        order = np.argsort(-probs[:, j], kind="mergesort")
        y_sorted = y[order]
        tp = np.cumsum(y_sorted)
        fp = np.cumsum(~y_sorted)
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / y.sum()
        rec_prev = np.concatenate([[0.0], recall[:-1]])
        out[j] = np.sum((recall - rec_prev) * precision)
    return out


def _auc_per_class(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    c = probs.shape[1]
    if _HAVE_TM:
        m = MultilabelAUROC(num_labels=c, average=None)
        m.update(torch.from_numpy(probs).float(), torch.from_numpy(labels).int())
        return m.compute().numpy()
    return np.full(c, np.nan)


def _group_means(ap: np.ndarray, groups: dict[str, np.ndarray]) -> dict[str, float]:
    out = {g: float(np.nanmean(ap[np.where(groups[g])[0]])) for g in
           ("head", "medium", "tail", "extreme_tail")}
    out["macro"] = float(np.nanmean(ap))
    out["category_avg"] = float(np.mean([out["head"], out["medium"], out["tail"]]))
    return out


def _load(npz_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    d = np.load(npz_path, allow_pickle=True)
    return d["probs"].astype(np.float64), d["labels"].astype(np.int64), list(d["class_names"])


def _bootstrap_group_ci(probs, labels, groups, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = probs.shape[0]
    keys = ("head", "medium", "tail", "extreme_tail", "macro", "category_avg")
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        gm = _group_means(_ap_per_class(probs[idx], labels[idx]), groups)
        for k in keys:
            samples[k].append(gm[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in samples.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", action="append", metavar=("NAME", "NPZ"),
                    required=True, help="condition name followed by 1+ npz paths (seeds)")
    ap.add_argument("--baseline", default=None, help="condition name to diff others against")
    ap.add_argument("--bootstrap", type=int, default=500, help="0 to disable")
    ap.add_argument("--out", default="analysis/out/perclass", help="output path prefix")
    args = ap.parse_args()

    meta = class_groups.build()
    names = meta["class_names"]
    groups = meta["groups"]
    prev = meta["train_prevalence"]
    c = len(names)

    report: dict = {"class_names": names,
                    "train_prevalence": prev.tolist(),
                    "group_membership": {g: [names[i] for i in np.where(groups[g])[0]]
                                         for g in ("head", "medium", "tail", "extreme_tail")},
                    "conditions": {}}

    per_cond_ap_mean: dict[str, np.ndarray] = {}

    for entry in args.cond:
        name, paths = entry[0], [Path(p) for p in entry[1:]]
        if not paths:
            raise SystemExit(f"--cond {name} has no npz paths")
        seed_ap, seed_auc, seed_groups, seed_valpos = [], [], [], []
        for p in paths:
            probs, labels, npz_names = _load(p)
            if npz_names != names:
                raise SystemExit(f"{p}: class order != label_info.pt order")
            a = _ap_per_class(probs, labels)
            seed_ap.append(a)
            seed_auc.append(_auc_per_class(probs, labels))
            seed_groups.append(_group_means(a, groups))
            seed_valpos.append(labels.sum(axis=0))
        seed_ap = np.vstack(seed_ap)              # [S, C]
        seed_auc = np.vstack(seed_auc)
        valpos = np.vstack(seed_valpos)[0]        # identical across seeds
        gm_keys = ("head", "medium", "tail", "extreme_tail", "macro", "category_avg")
        gmat = {k: np.array([g[k] for g in seed_groups]) for k in gm_keys}

        per_cond_ap_mean[name] = np.nanmean(seed_ap, axis=0)

        cond = {
            "npz": [str(p) for p in paths],
            "n_seeds": len(paths),
            "val_pos": valpos.tolist(),
            "per_class_ap_mean": np.nanmean(seed_ap, axis=0).tolist(),
            "per_class_ap_std": np.nanstd(seed_ap, axis=0, ddof=1).tolist() if len(paths) > 1
                                else [0.0] * c,
            "per_class_auc_mean": np.nanmean(seed_auc, axis=0).tolist(),
            "group_mAP_mean": {k: float(np.mean(gmat[k])) for k in gm_keys},
            "group_mAP_std": {k: float(np.std(gmat[k], ddof=1)) if len(paths) > 1 else 0.0
                              for k in gm_keys},
        }
        if args.bootstrap and len(paths) >= 1:
            probs0, labels0, _ = _load(paths[0])
            cond["group_mAP_bootstrap95_seed0"] = _bootstrap_group_ci(
                probs0, labels0, groups, n_boot=args.bootstrap)
        report["conditions"][name] = cond

        print(f"\n=== {name}  ({len(paths)} seed(s)) ===")
        for k in ("macro", "head", "medium", "tail", "extreme_tail", "category_avg"):
            s = cond["group_mAP_std"][k]
            print(f"  {k:<13} mAP = {cond['group_mAP_mean'][k]:.4f}" +
                  (f" +/- {s:.4f}" if s else ""))

    if args.baseline and args.baseline in per_cond_ap_mean:
        base = per_cond_ap_mean[args.baseline]
        report["deltas_vs_baseline"] = {"baseline": args.baseline, "per_condition": {}}
        for name, apm in per_cond_ap_mean.items():
            if name == args.baseline:
                continue
            d = apm - base
            dg = {g: float(np.nanmean(d[np.where(groups[g])[0]])) for g in
                  ("head", "medium", "tail", "extreme_tail")}
            dg["macro"] = float(np.nanmean(d))
            report["deltas_vs_baseline"]["per_condition"][name] = {
                "group_delta": dg,
                "per_class_delta": d.tolist(),
            }
            print(f"\n=== delta  {name} - {args.baseline} ===")
            for g in ("macro", "head", "medium", "tail", "extreme_tail"):
                print(f"  {g:<13} {dg[g]:+.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.with_suffix(".json")).write_text(json.dumps(report, indent=2))
    _write_latex(report, out.with_suffix(".tex"), names, prev)
    print(f"\nwrote {out.with_suffix('.json')} and {out.with_suffix('.tex')}")


def _write_latex(report, path, names, prev):
    conds = list(report["conditions"])
    lines = []
    # frequency-group table
    lines.append("% frequency-group mAP (mean +/- std over seeds)")
    lines.append("\\begin{tabular}{l" + "c" * 5 + "}")
    lines.append("\\hline")
    lines.append("Method & Macro & Head & Medium & Tail & Cat.\\ avg \\\\")
    lines.append("\\hline")
    for cn in conds:
        gm = report["conditions"][cn]["group_mAP_mean"]
        gs = report["conditions"][cn]["group_mAP_std"]
        cells = []
        for k in ("macro", "head", "medium", "tail", "category_avg"):
            cells.append(f"{gm[k]:.3f}" + (f"$\\pm${gs[k]:.3f}" if gs[k] else ""))
        lines.append(f"{cn} & " + " & ".join(cells) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("")
    # full per-class table
    lines.append("% per-class AP")
    lines.append("\\begin{tabular}{lr" + "c" * len(conds) + "}")
    lines.append("\\hline")
    lines.append("Class & Prev.\\ (\\%) & " + " & ".join(conds) + " \\\\")
    lines.append("\\hline")
    order = np.argsort(prev)  # rare first
    for i in order:
        row = [names[i].replace("_", "\\_"), f"{100*prev[i]:.2f}"]
        for cn in conds:
            m = report["conditions"][cn]["per_class_ap_mean"][i]
            s = report["conditions"][cn]["per_class_ap_std"][i]
            row.append(f"{m:.3f}" + (f"$\\pm${s:.3f}" if s else ""))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
