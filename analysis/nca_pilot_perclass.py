"""Class-conditional memory loss pilot (job 53160, --metric-loss nca, seed 42): per-class AP and calibration breakdown
against the seeded ASL (v0) and seeded Entropy (arm c) runs of the same seed (2026-10-17).

Aggregates, groups and the paired image bootstrap come from analysis/eval_interp_table.py (analysis/out/
eval_interp_table.json, "arms" -> "N vs A (area)"); this script adds what that table does not print: every class, the
share of classes that got worse, and why the mECE moved (probability mass on negatives, reliability bins).
Flip-TTA SWA dumps, area eval, same row order (keys.txt).

    python -I analysis/nca_pilot_perclass.py  -> analysis/out/nca_pilot_perclass.json + printed tables
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edge_common as ec  # noqa: E402

ea, O = ec.ea, ec.O


def main() -> None:
    ctx = ec.load_context()
    L, names, meta = ctx["L"], ctx["names"], ctx["meta"]
    files = {"A": O / "probs_v0_flip_seeded_area/v0_768_seeded__seed_42__model_swa@768_flip_area.npy",
             "C": O / "probs_unified_flip_seeded_area/c__seed_42__model_swa@768_flip_area.npy",
             "N": O / "probs_unified_flip_seeded/n__seed_42__model_swa@768_flip_area.npy"}
    P = {k: np.load(f).astype(np.float64) for k, f in files.items()}
    ap = {k: ea.tm_metrics(p, L) for k, p in P.items()}
    out = {"files": {k: str(f.relative_to(O)) for k, f in files.items()},
           "macro": {k: {m: ap[k][m] for m in ("mAP", "mAUC", "mF1", "mECE")} for k in P}}

    prev = L.mean()
    cal = {}
    for k, p in P.items():
        b = np.minimum((p * 15).astype(int), 14)
        bins = []
        for i in range(15):
            m = b == i
            if m.any():
                bins.append({"bin": i, "n": int(m.sum()), "conf": float(p[m].mean()), "acc": float(L[m].mean())})
        cal[k] = {"mean_prob": float(p.mean()), "ratio_to_prevalence": float(p.mean() / prev),
                  "mean_prob_positives": float(p[L == 1].mean()), "mean_prob_negatives": float(p[L == 0].mean()),
                  "share_entries_below_0.5": float((p < 0.5).mean()),
                  "bins": bins}
    out["calibration"] = cal
    out["label_prevalence_mean"] = float(prev)

    rows = []
    d = ap["N"]["ap"] - ap["A"]["ap"]
    dc = ap["N"]["ap"] - ap["C"]["ap"]
    for i in np.argsort(ctx["prev"]):
        rows.append({"class": names[i], "group": ea.class_groups.group_of(ctx["prev"][i]),
                     "train_pos": int(meta["train_counts"][i]), "val_pos": int(L[:, i].sum()),
                     "A": float(ap["A"]["ap"][i]), "C": float(ap["C"]["ap"][i]), "N": float(ap["N"]["ap"][i]),
                     "N-A": float(d[i]), "N-C": float(dc[i])})
    out["per_class"] = rows
    g = ctx["groups"]
    tail = g["tail"]
    two_noisy = [i for i in tail if names[i] in ("Hydropneumothorax", "pneumoperitoneo")]
    out["summary"] = {
        "classes_worse_than_A": int((d < 0).sum()), "classes_better_than_A": int((d > 0).sum()),
        "tail_worse": int((d[tail] < 0).sum()), "tail_n": int(len(tail)),
        "tail_mean_N-A": float(d[tail].mean()),
        "tail_mean_N-A_without_two_rarest": float(np.mean([d[i] for i in tail if i not in two_noisy])),
        "medium_mean_N-A": float(d[g["medium"]].mean()), "medium_worse": int((d[g["medium"]] < 0).sum()),
        "head_mean_N-A": float(d[g["head"]].mean()),
    }
    Path(O / "nca_pilot_perclass.json").write_text(json.dumps(out, indent=1))

    print("macro:", json.dumps(out["macro"], indent=0))
    for k in ("A", "C", "N"):
        c = cal[k]
        print(f"{k}: mean p {c['mean_prob']:.4f} ({c['ratio_to_prevalence']:.2f}x prevalence {prev:.4f}) | "
              f"mean p on positives {c['mean_prob_positives']:.4f}, on negatives {c['mean_prob_negatives']:.4f}")
    print("\n| class | group | train+ | val+ | ASL | Entropy | NCA | NCA - ASL | NCA - Entropy |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['class']} | {r['group']} | {r['train_pos']} | {r['val_pos']} | {r['A']:.4f} | {r['C']:.4f} | {r['N']:.4f} | "
              f"{r['N-A']:+.4f} | {r['N-C']:+.4f} |")
    print("\nsummary:", json.dumps(out["summary"], indent=1))
    print(f"wrote {O / 'nca_pilot_perclass.json'}")


if __name__ == "__main__":
    main()
