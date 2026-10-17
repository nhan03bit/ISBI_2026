"""Where do the Table 1 arms differ? Per-class and per-frequency-group AP, tail first (2026-10-16).

Macro mAP over 30 classes is dominated by the 20 medium classes, so an effect confined to the 7 tail classes
(train prevalence < 1%, evaluate/class_groups.py) moves the headline by at most 7/30 of its size. This script
reads the saved flip-TTA SWA lanczos dumps of the 5-seed unseeded Table 1 runs (A = ASL v0, C = Entropy,
Dwith / Dzero = Markov with / withheld history; file layout from analysis/eval_interp_table.runs) and reports:

  * per-class AP (seed mean) with train / val positive counts, and the paired per-seed difference vs ASL
    (wins out of 5);
  * group means (head / medium / tail / extreme_tail, plus rare10 = the 10 rarest classes) on all views and
    frontal views (PA / AP / AP_horizontal; the leaderboard test set is frontal-only);
  * two intervals for each group contrast x - y:
      seeds  paired t-CI over the 5 seeds (training noise; analysis/entropy_ablation.paired)
      images 95% bootstrap over val images of the seed-mean difference (val sampling noise; with only a few
             val positives per tail class this one is usually the wider).
AP and AUROC in the bootstrap come from entropy_ablation.FastAP (tie-grouped weighted AP, equal to torchmetrics
at unit weights; asserted below).

Patient history (the internal val is not patient-disjoint: 77% of val images share a patient with train fold 1):
  * unseen  the group contrasts on val images whose patient has no train-fold image;
  * hist+ / hist-  the tail and rare10 contrasts with each class's positives split by whether the patient has a
    train-fold image positive for the same class (negatives = all val images). The Markov "with history" arm
    sees the patient's earlier labels, so a gain confined to hist+ positives is label recall, not image reading.
PatientID / Projection come from the PadChest CSV for internal-val and train-fold images only.

    python -I analysis/tail_breakdown.py [--boot 500]  -> analysis/out/tail_breakdown.json + printed tables
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entropy_ablation as ea  # noqa: E402
import eval_interp_table as eit  # noqa: E402

O = eit.O
ARMS = ("A", "C", "Dwith", "Dzero")
NAMES = {"A": "ASL", "C": "Entropy", "Dwith": "Markov+hist", "Dzero": "Markov-wh"}
CONTRASTS = [("C-A", "C", "A"), ("Dwith-A", "Dwith", "A"), ("Dzero-A", "Dzero", "A"), ("Dwith-Dzero", "Dwith", "Dzero")]
GROUPS = ("head", "medium", "tail", "extreme_tail", "rare10")
SUBSETS = ("all", "frontal")
HIST_GROUPS = ("tail", "rare10")


def ap_cols(fa: ea.FastAP, W: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """AP of classes `cols` with per-class sample weights W[:, j] (the FastAP tie-grouped AP, one weight
    vector per class instead of one shared vector)."""
    out = np.full(len(cols), np.nan)
    for j, c in enumerate(cols):
        o = fa.order[:, c]
        y = fa.L[o, c]
        wp = W[o, j] * y
        wn = W[o, j] - wp
        if wp.sum() > 0 and wn.sum() > 0:
            g = fa.grp[c]
            ng = g[-1] + 1
            pr, nr = np.bincount(g, wp, ng)[::-1], np.bincount(g, wn, ng)[::-1]
            tp, fp = np.cumsum(pr), np.cumsum(nr)
            out[j] = (pr * tp / np.maximum(tp + fp, 1e-12)).sum() / wp.sum()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=500, help="image bootstrap resamples")
    ap.add_argument("--out", type=Path, default=O / "tail_breakdown.json")
    args = ap.parse_args()

    keys = (O / "probs_unified_flip/keys.txt").read_text().split("\n")
    L = np.load(O / "probs_unified_flip/labels.npy").astype(np.int64)
    meta = ea.class_groups.build()
    names, prev, tr_pos = meta["class_names"], meta["train_prevalence"], meta["train_counts"]
    C = len(names)
    groups = {g: np.where(meta["groups"][g])[0] for g in ("head", "medium", "tail", "extreme_tail")}
    groups["rare10"] = np.argsort(prev)[:10]
    pc = pd.read_csv(eit.PADCHEST_CSV, usecols=["ImageID", "PatientID", "Projection"], dtype=str,
                     low_memory=False).drop_duplicates("ImageID").set_index("ImageID")
    proj = pc["Projection"].reindex(keys)
    assert proj.notna().all()
    masks = {"all": np.ones(len(L), bool), "frontal": proj.isin(eit.FRONTAL).values}

    # patient history from train fold 1: seen = the patient has a train image; H[i, c] = the patient has a train
    # image positive for class c
    pid = pc["PatientID"].reindex(keys).values
    tr = pd.read_csv(eit.ROOT / "data/train_fold_1.csv")
    tr_pid = pc["PatientID"].reindex(tr["filename"]).values
    cidx = {c: i for i, c in enumerate(names)}
    Ytr = np.zeros((len(tr), C), bool)
    for r, labs in enumerate(tr["label_list"]):
        for c in ast.literal_eval(labs):
            Ytr[r, cidx[c]] = True
    hpos = pd.DataFrame(Ytr, columns=range(C)).assign(pid=tr_pid).dropna(subset=["pid"]).groupby("pid").any()
    H = hpos.reindex(pid).fillna(False).to_numpy(bool)
    seen = np.isin(pid, hpos.index)

    files = {(r["arm"], r["seed"]): r["file"] for r in eit.runs()
             if r["set"] == "unseeded" and r["interp"] == "lanczos" and r["arm"] in ARMS}
    missing = [str(f) for f in files.values() if not f.exists()]
    assert not missing, f"missing dumps: {missing}"
    for f in files.values():
        assert (f.parent / "keys.txt").read_text().split("\n") == keys and \
               (np.load(f.parent / "labels.npy").astype(np.int64) == L).all(), f"{f.parent}: row order/labels differ"

    fast = {k: ea.FastAP(np.load(f).astype(np.float64), L) for k, f in files.items()}
    # point estimates per run and subset (FastAP with 0/1 weights = AP on the subset)
    point = {s: {k: fa(masks[s].astype(float)) for k, fa in fast.items()} for s in SUBSETS}
    k0 = ("A", 42)
    tm = ea.tm_metrics(fast[k0].P, L)["ap"]
    assert np.allclose(point["all"][k0][0], tm, atol=1e-5), "FastAP != torchmetrics"

    val_pos = {s: L[masks[s]].sum(0) for s in SUBSETS}
    res = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "n_val": len(L),
           "n_frontal": int(masks["frontal"].sum()), "boot": args.boot, "seeds": list(eit.SEEDS),
           "groups": {g: [names[i] for i in idx] for g, idx in groups.items()}, "per_class": [], "group": {},
           "contrast": {}}

    # ---- per class (all views and frontal), sorted rarest first
    for i in np.argsort(prev):
        row = {"class": names[i], "train_pos": int(tr_pos[i]), "train_prev": float(prev[i]),
               "group": ea.class_groups.group_of(prev[i])}
        for s in SUBSETS:
            row[f"val_pos_{s}"] = int(val_pos[s][i])
            for a in ARMS:
                row[f"{s}_{a}"] = float(np.mean([point[s][(a, sd)][0][i] for sd in eit.SEEDS]))
                row[f"{s}_{a}_auc"] = float(np.mean([point[s][(a, sd)][1][i] for sd in eit.SEEDS]))
            for name, x, y in CONTRASTS:
                d = np.array([point[s][(x, sd)][0][i] - point[s][(y, sd)][0][i] for sd in eit.SEEDS])
                row[f"{s}_{name}"] = float(d.mean())
                row[f"{s}_{name}_wins"] = int((d > 0).sum())
        res["per_class"].append(row)

    # ---- group means and contrasts
    def gmean(v: np.ndarray, g: str) -> float:
        return float(np.nanmean(v[groups[g]]))

    for s in SUBSETS:
        for a in ARMS:
            for g in GROUPS:
                res["group"].setdefault(s, {}).setdefault(a, {})[g] = {
                    "ap": ea.summarize([gmean(point[s][(a, sd)][0], g) for sd in eit.SEEDS]),
                    "auc": ea.summarize([gmean(point[s][(a, sd)][1], g) for sd in eit.SEEDS])}

    rng = np.random.default_rng(0)
    boots = {s: {k: [] for k in fast} for s in SUBSETS}
    idx_all = np.arange(len(L))
    for b in range(args.boot):
        w = np.bincount(rng.choice(idx_all, len(L)), minlength=len(L)).astype(float)
        for s in SUBSETS:
            ws = w * masks[s]
            for k, fa in fast.items():
                boots[s][k].append(fa(ws)[0])
        if (b + 1) % 100 == 0:
            print(f"bootstrap {b + 1}/{args.boot}", flush=True)
    for s in SUBSETS:
        for k in fast:
            boots[s][k] = np.array(boots[s][k])  # [B, C]

    for s in SUBSETS:
        for name, x, y in CONTRASTS:
            for g in GROUPS:
                seed_d = np.array([gmean(point[s][(x, sd)][0], g) - gmean(point[s][(y, sd)][0], g) for sd in eit.SEEDS])
                bd = np.mean([np.nanmean(boots[s][(x, sd)][:, groups[g]], 1) - np.nanmean(boots[s][(y, sd)][:, groups[g]], 1)
                              for sd in eit.SEEDS], axis=0)
                res["contrast"].setdefault(s, {}).setdefault(name, {})[g] = {
                    "seeds": ea.paired(seed_d), "img_ci": [float(v) for v in np.percentile(bd, [2.5, 97.5])],
                    "img_p_le0": float((bd <= 0).mean())}

    # ---- patient history: unseen-patient subset, and tail positives split by same-class history
    hcols = groups["rare10"]                      # superset of tail; HIST_GROUPS index into it
    hsel = {g: np.isin(hcols, groups[g]) for g in HIST_GROUPS}
    keep = {"hist+": (L[:, hcols] == 0) | H[:, hcols], "hist-": (L[:, hcols] == 0) | ~H[:, hcols]}
    hist = {"n_seen": int(seen.sum()), "n_unseen": int((~seen).sum()), "classes": [], "unseen": {}, "split": {}}
    for i in hcols:
        p = L[:, i] == 1
        hist["classes"].append({"class": names[i], "val_pos": int(p.sum()), "hist_pos": int((p & H[:, i]).sum()),
                                "unseen_pos": int((p & ~seen).sum())})
    upoint = {k: fa((~seen).astype(float))[0] for k, fa in fast.items()}
    hpoint = {h: {k: ap_cols(fa, keep[h].astype(float), hcols) for k, fa in fast.items()} for h in keep}
    assert np.allclose(ap_cols(fast[k0], np.ones((len(L), len(hcols))), hcols), point["all"][k0][0][hcols])
    rng = np.random.default_rng(1)
    ub = {k: [] for k in fast}
    hb = {h: {k: [] for k in fast} for h in keep}
    for b in range(args.boot):
        w = np.bincount(rng.choice(idx_all, len(L)), minlength=len(L)).astype(float)
        for k, fa in fast.items():
            ub[k].append(fa(w * ~seen)[0])
            for h in keep:
                hb[h][k].append(ap_cols(fa, w[:, None] * keep[h], hcols))
        if (b + 1) % 100 == 0:
            print(f"history bootstrap {b + 1}/{args.boot}", flush=True)
    ub = {k: np.array(v) for k, v in ub.items()}
    hb = {h: {k: np.array(v) for k, v in d.items()} for h, d in hb.items()}

    def contrast(pt, bs, x, y, sel) -> dict:
        seed_d = np.array([np.nanmean(pt[(x, sd)][sel]) - np.nanmean(pt[(y, sd)][sel]) for sd in eit.SEEDS])
        bd = np.mean([np.nanmean(bs[(x, sd)][:, sel], 1) - np.nanmean(bs[(y, sd)][:, sel], 1) for sd in eit.SEEDS], 0)
        return {"asl": float(np.mean([np.nanmean(pt[("A", sd)][sel]) for sd in eit.SEEDS])),
                "seeds": ea.paired(seed_d), "img_ci": [float(v) for v in np.percentile(bd, [2.5, 97.5])]}

    for name, x, y in CONTRASTS:
        for g in ("head", "medium", "tail", "rare10"):
            hist["unseen"].setdefault(name, {})[g] = contrast(upoint, ub, x, y, groups[g])
        for h in keep:
            for g in HIST_GROUPS:
                hist["split"].setdefault(h, {}).setdefault(name, {})[g] = contrast(hpoint[h], hb[h], x, y, hsel[g])
    res["history"] = hist

    # ---- print
    f4 = lambda v: f"{v:.4f}"  # noqa: E731
    sg = lambda v: f"{v:+.4f}"  # noqa: E731
    for s in SUBSETS:
        print(f"\n== group mean AP, {s} views (5-seed mean; n_img={int(masks[s].sum())}) ==")
        print(f"{'group':13s} {'#cls':>4s} " + " ".join(f"{NAMES[a]:>11s}" for a in ARMS))
        for g in GROUPS:
            print(f"{g:13s} {len(groups[g]):4d} " + " ".join(f"{res['group'][s][a][g]['ap']['mean']:11.4f}" for a in ARMS))
        print(f"\n-- contrasts, {s} views: mean | seed t-CI (wins/5) | image-bootstrap CI --")
        for name, *_ in CONTRASTS:
            for g in GROUPS:
                c = res["contrast"][s][name][g]
                sd = c["seeds"]
                print(f"{name:12s} {g:13s} {sg(sd['mean'])} | [{sg(sd['ci'][0])}, {sg(sd['ci'][1])}] ({sd['wins']}/5) | "
                      f"[{sg(c['img_ci'][0])}, {sg(c['img_ci'][1])}]")
    print("\n== per class, rarest first: AP all views (5-seed mean); deltas vs ASL with wins/5; frontal Dwith-A ==")
    print(f"{'class':30s} {'tr+':>6s} {'val+':>5s} {'fr+':>5s} {'ASL':>6s} {'Ent':>6s} {'M+h':>6s} {'M-wh':>6s} "
          f"{'C-A':>9s} {'Dw-A':>9s} {'Dz-A':>9s} {'frDw-A':>8s}")
    for r in res["per_class"]:
        print(f"{r['class'][:30]:30s} {r['train_pos']:6d} {r['val_pos_all']:5d} {r['val_pos_frontal']:5d} "
              + " ".join(f4(r[f'all_{a}'])[1:].rjust(6) for a in ARMS) + " "
              + " ".join(f"{sg(r[f'all_{n}'])}({r[f'all_{n}_wins']})" for n in ("C-A", "Dwith-A", "Dzero-A"))
              + f" {sg(r['frontal_Dwith-A'])}")

    print(f"\n== patient history (seen {hist['n_seen']} / unseen {hist['n_unseen']} val images) ==")
    for c in hist["classes"]:
        print(f"{c['class'][:30]:30s} val+ {c['val_pos']:4d}  same finding in a train image of the patient "
              f"{c['hist_pos']:4d} ({c['hist_pos'] / max(1, c['val_pos']):4.0%})  unseen-patient val+ {c['unseen_pos']:3d}")
    for title, block in [("unseen patients", {"": hist["unseen"]})] + \
                        [(f"positives {h} (negatives = all val)", {h: hist["split"][h]}) for h in keep]:
        print(f"\n-- {title}: ASL AP | x - y mean | seed t-CI (wins/5) | image-bootstrap CI --")
        for d in block.values():
            for name, gs in d.items():
                for g, c in gs.items():
                    sd = c["seeds"]
                    print(f"{name:12s} {g:7s} ASL {c['asl']:.4f} | {sg(sd['mean'])} | [{sg(sd['ci'][0])}, {sg(sd['ci'][1])}] "
                          f"({sd['wins']}/5) | [{sg(c['img_ci'][0])}, {sg(c['img_ci'][1])}]")

    args.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
