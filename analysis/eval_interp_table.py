"""Eval resize filter (lanczos vs area) on every Table 1 run, with a frontal-only column (2026-10-16).

Investigation: docs/plan/2026-10-16-asl-046-ceiling-investigation.md. Training shrinks images with INTER_LINEAR and
every eval so far used INTER_LANCZOS4. The lanczos dumps are the original flip-TTA SWA evals; the area dumps come
from scripts/submit_reeval_interp.sh (evaluate_tta.py --interp area) into <dir>_area, with the same keys.txt row
order. Metrics are the Table 1 code (analysis/entropy_ablation.tm_metrics + class groups). "Frontal" = PadChest
Projection PA / AP / AP_horizontal (a metadata column, internal-val images only): the leaderboard test set is
frontal-only, the internal val is 30% lateral.

Also scores the two 2026-10-16 training arms (seeded ASL v0, seed 42) once their dumps exist, against the seeded
ASL baseline under the same eval filter (single runs; paired bootstrap over images, not over seeds):
  trainarea  --train-interp area --val-interp area    probs_v0_flip_seeded_trainarea (area eval)
  sched4     --sched-epochs 4                          probs_v0_flip_seeded_sched4 (lanczos and area evals)
trainarea is also scored against the lanczos-evaluated baseline (the whole pipeline change, train + eval filter).

Where the gain lands: area - lanczos per frequency group (head / medium / tail / extreme_tail / rare10, train
prevalence, evaluate/class_groups.py) and per class, for every run with both dumps (same weights, two eval
filters), and per group for the training arms. Group intervals are a 95% bootstrap over val images of the mean
over runs (all runs share the val images, so agreement across runs alone overstates certainty).

    python -I analysis/eval_interp_table.py           -> analysis/out/eval_interp_table.json + printed tables
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchmetrics.functional.classification import multilabel_average_precision

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entropy_ablation as ea  # noqa: E402  (tm_metrics, group_means, paired, summarize, class groups)

ROOT = ea.ROOT
O = ROOT / "analysis/out"
PADCHEST_CSV = ROOT / "data/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
FRONTAL = ("PA", "AP", "AP_horizontal")
INTERPS = ("lanczos", "area")
SEEDS = (42, 123, 456, 789, 1024)
ARMS = ("A", "C", "Dwith", "Dzero")
CONTRASTS = [("C-A", "C", "A"), ("Dwith-A", "Dwith", "A"), ("Dzero-A", "Dzero", "A"),
             ("Dwith-C", "Dwith", "C"), ("Dzero-C", "Dzero", "C")]


def dump(base: str, stem: str, interp: str, prior: str = "", split_dir: bool = True) -> Path:
    """<base>[_area]/<stem>@768_flip[_area][_prior-with|zero].npy, the evaluate_tta_v5.py naming. Re-evals
    (submit_reeval_interp.sh) write to <base>_<interp>; the training arms' own evals (submit_stage2_v0.sh
    EVAL_INTERPS) keep every filter in <base> (split_dir=False)."""
    d = O / (base if interp == "lanczos" or not split_dir else f"{base}_{interp}")
    itag = "" if interp == "lanczos" else f"_{interp}"
    return d / f"{stem}@768_flip{itag}{'_prior-' + prior if prior else ''}.npy"


def runs() -> list[dict]:
    """(set, arm, seed, interp, file) for every dump this table can use."""
    out = []
    for it in INTERPS:
        for arm, stem, prior in (("A", "v0_768_seeded__seed_42__model_swa", ""),
                                 ("C", "c__seed_42__model_swa", ""),
                                 ("Dwith", "d__seed_42__model_swa", "with"),
                                 ("Dzero", "d__seed_42__model_swa", "zero"),
                                 ("P", "p__seed_42__model_swa", "")):
            base = "probs_v0_flip_seeded" if arm == "A" else "probs_unified_flip_seeded"
            out.append(dict(set="seeded", arm=arm, seed=42, interp=it, file=dump(base, stem, it, prior)))
        # class-conditional memory loss arms (2026-10-16, --metric-loss nca): their own evals keep both filters in
        # probs_unified_flip_seeded (submit_stage2_v5_unified.sh EVAL_INTERPS), so no split directory
        for arm, stem in (("N", "n__seed_42__model_swa"), ("NA", "na__seed_42__model_swa"),
                          ("NR", "nr__seed_42__model_swa")):
            out.append(dict(set="seeded", arm=arm, seed=42, interp=it,
                            file=dump("probs_unified_flip_seeded", stem, it, split_dir=False)))
        for s in SEEDS:
            dstem = f"prior__img768_qprior_seed{s}__model_swa" if s in (42, 1024) else f"d__seed_{s}__model_swa"
            out += [dict(set="unseeded", arm="A", seed=s, interp=it, file=dump("probs_v0_flip", f"v0_768__seed_{s}__model_swa", it)),
                    dict(set="unseeded", arm="C", seed=s, interp=it, file=dump("probs_unified_flip", f"c__seed_{s}__model_swa", it)),
                    dict(set="unseeded", arm="Dwith", seed=s, interp=it, file=dump("probs_unified_flip", dstem, it, "with")),
                    dict(set="unseeded", arm="Dzero", seed=s, interp=it, file=dump("probs_unified_flip", dstem, it, "zero"))]
        out.append(dict(set="arms", arm="sched4", seed=42, interp=it,
                        file=dump("probs_v0_flip_seeded_sched4", "v0_768_seeded_sched4__seed_42__model_swa", it,
                                  split_dir=False)))
    out.append(dict(set="arms", arm="trainarea", seed=42, interp="area",
                    file=dump("probs_v0_flip_seeded_trainarea", "v0_768_seeded_trainarea__seed_42__model_swa", "area",
                              split_dir=False)))
    return out


def boot_diff(Px: np.ndarray, Py: np.ndarray, L: np.ndarray, n: int, seed: int = 0) -> list[float]:
    """95% interval of mAP(Px) - mAP(Py) over images (paired bootstrap)."""
    rng = np.random.default_rng(seed)
    px, py, y = torch.from_numpy(Px).float(), torch.from_numpy(Py).float(), torch.from_numpy(L).long()
    d = []
    for _ in range(n):
        i = torch.from_numpy(rng.integers(0, len(L), len(L)))
        d.append(float(multilabel_average_precision(px[i], y[i], num_labels=L.shape[1]))
                 - float(multilabel_average_precision(py[i], y[i], num_labels=L.shape[1])))
    return [float(v) for v in np.percentile(d, [2.5, 97.5])]


def group_deltas(pairs: list, masks: dict, groups: dict, nboot: int, seed: int = 0) -> dict:
    """Mean over `pairs` [(name, FastAP x, FastAP y)] of AP(x) - AP(y) per val subset: macro, each frequency group
    (with runs up and a 95% image bootstrap of the mean) and each class (mean and runs up)."""
    n = len(next(iter(masks.values())))
    pt = {s: np.array([x(m.astype(float))[0] - y(m.astype(float))[0] for _, x, y in pairs]) for s, m in masks.items()}
    sel = {"macro": slice(None), **groups}
    bs = {s: {g: [] for g in sel} for s in masks}
    rng = np.random.default_rng(seed)
    for _ in range(nboot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        for s, m in masks.items():
            d = np.array([x(w * m)[0] - y(w * m)[0] for _, x, y in pairs])
            for g, gi in sel.items():
                bs[s][g].append(np.nanmean(d[:, gi], 1).mean())
    out = {}
    for s, d in pt.items():
        out[s] = {"per_class_mean": np.nanmean(d, 0).tolist(), "per_class_up": (d > 0).sum(0).tolist()}
        for g, gi in sel.items():
            v = np.nanmean(d[:, gi], 1)
            out[s][g] = {"mean": float(v.mean()), "up": int((v > 0).sum()), "n": len(v),
                         "ci": [float(c) for c in np.percentile(bs[s][g], [2.5, 97.5])] if nboot else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200, help="bootstrap resamples for the single-run arm comparisons")
    ap.add_argument("--boot-groups", type=int, default=300, help="bootstrap resamples for the per-group deltas")
    ap.add_argument("--out", type=Path, default=O / "eval_interp_table.json")
    args = ap.parse_args()

    keys = (O / "probs_unified_flip/keys.txt").read_text().split("\n")
    L = np.load(O / "probs_unified_flip/labels.npy").astype(np.int64)
    meta = ea.class_groups.build()
    groups = {g: np.where(meta["groups"][g])[0] for g in ("head", "medium", "tail", "extreme_tail")}
    proj = pd.read_csv(PADCHEST_CSV, usecols=["ImageID", "Projection"]).drop_duplicates("ImageID") \
        .set_index("ImageID").reindex(keys)["Projection"]
    assert proj.notna().all(), "val image missing from the PadChest metadata"
    fr = proj.isin(FRONTAL).values

    rows, probs, missing = [], {}, []
    for r in runs():
        f = r["file"]
        if not f.exists():
            missing.append(str(f.relative_to(O)))
            continue
        d = f.parent
        assert (d / "keys.txt").read_text().split("\n") == keys, f"{d}: row order differs"
        assert (np.load(d / "labels.npy").astype(np.int64) == L).all(), f"{d}: labels differ"
        P = np.load(f).astype(np.float64)
        m, mf = ea.tm_metrics(P, L), ea.tm_metrics(P[fr], L[fr])
        rows.append({**{k: r[k] for k in ("set", "arm", "seed", "interp")}, "file": str(f.relative_to(O)),
                     **{k: m[k] for k in ("mAP", "mAUC", "mF1", "mECE")}, **ea.group_means(m["ap"], groups),
                     "frontal_mAP": mf["mAP"], "frontal_mECE": mf["mECE"]})
        probs[(r["set"], r["arm"], r["seed"], r["interp"])] = P
    get = {(r["set"], r["arm"], r["seed"], r["interp"]): r for r in rows}
    res = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "n_val": len(keys), "n_frontal": int(fr.sum()),
           "missing": missing, "rows": rows}

    print(f"val N={len(keys)} | frontal N={int(fr.sum())} | missing dumps: {len(missing)}")
    print("\n== seeded, seed 42 (Table 1 rerun): lanczos -> area ==")
    print(f"{'arm':6s} {'mAP lz':>7s} {'area':>7s} {'d':>7s} | {'frontal lz':>10s} {'area':>7s} | {'mECE lz':>7s} {'area':>7s} | "
          f"{'head':>6s} {'med':>6s} {'tail':>6s} (area)")
    for arm in ("A", "C", "Dwith", "Dzero", "P", "N", "NA", "NR"):
        a, b = get.get(("seeded", arm, 42, "lanczos")), get.get(("seeded", arm, 42, "area"))
        if a and b:
            print(f"{arm:6s} {a['mAP']:.4f} {b['mAP']:.4f} {b['mAP'] - a['mAP']:+.4f} | {a['frontal_mAP']:10.4f} {b['frontal_mAP']:.4f} | "
                  f"{a['mECE']:.4f} {b['mECE']:.4f} | {b['head']:.4f} {b['medium']:.4f} {b['tail']:.4f}")

    res["unseeded"] = {}
    print("\n== unseeded 5-seed Table 1 runs: mean (sd) over seeds ==")
    for it in INTERPS:
        for arm in ARMS:
            got = [get[k] for k in [("unseeded", arm, s, it) for s in SEEDS] if k in get]
            if not got:
                continue
            res["unseeded"].setdefault(it, {})[arm] = {met: ea.summarize([g[met] for g in got])
                                                       for met in ("mAP", "frontal_mAP", "mECE", "head", "medium", "tail")}
            s = res["unseeded"][it][arm]
            print(f"{it:8s} {arm:6s} n={len(got)} mAP {s['mAP']['mean']:.4f} ({s['mAP']['sd'] or 0:.4f}) | "
                  f"frontal {s['frontal_mAP']['mean']:.4f} ({s['frontal_mAP']['sd'] or 0:.4f}) | mECE {s['mECE']['mean']:.4f}")
    print("\n   area - lanczos per run (mAP): ", end="")
    dl = [get[("unseeded", a, s, "area")]["mAP"] - get[("unseeded", a, s, "lanczos")]["mAP"]
          for a in ARMS for s in SEEDS if ("unseeded", a, s, "area") in get and ("unseeded", a, s, "lanczos") in get]
    if dl:
        res["unseeded_area_minus_lanczos"] = ea.summarize(dl)
        print(f"mean {np.mean(dl):+.4f}, range [{min(dl):+.4f}, {max(dl):+.4f}], {sum(x > 0 for x in dl)}/{len(dl)} runs up")
    else:
        print("no pairs yet")

    res["contrasts"] = {}
    print("\n== paired contrasts over seeds (x - y), all views and frontal ==")
    for it in INTERPS:
        for name, x, y in CONTRASTS:
            common = [s for s in SEEDS if ("unseeded", x, s, it) in get and ("unseeded", y, s, it) in get]
            if len(common) < 2:
                continue
            c = {met: ea.paired(np.array([get[("unseeded", x, s, it)][met] - get[("unseeded", y, s, it)][met]
                                          for s in common])) for met in ("mAP", "frontal_mAP")}
            res["contrasts"].setdefault(it, {})[name] = c
            fm = lambda v: ea.fmt(v, sign=True)  # noqa: E731
            print(f"{it:8s} {name:8s} n={len(common)} mAP {fm(c['mAP']['mean'])} CI [{fm(c['mAP'].get('ci', [None])[0])}, "
                  f"{fm(c['mAP'].get('ci', [None, None])[1])}] p {ea.fmt(c['mAP'].get('p_signflip'), 3)} | frontal "
                  f"{fm(c['frontal_mAP']['mean'])} CI [{fm(c['frontal_mAP'].get('ci', [None])[0])}, "
                  f"{fm(c['frontal_mAP'].get('ci', [None, None])[1])}]")

    masks = {"all": np.ones(len(L), bool), "frontal": fr}
    fgroups = {**groups, "rare10": np.argsort(meta["train_prevalence"])[:10]}
    fast = {}

    def fa(k):
        if k not in fast:
            fast[k] = ea.FastAP(probs[k], L)
        return fast[k]

    ipairs = [("/".join(map(str, k[:3])), fa(k[:3] + ("area",)), fa(k))
              for k in probs if k[3] == "lanczos" and k[:3] + ("area",) in probs]
    print(f"\n== where the area gain lands: area - lanczos, same weights, {len(ipairs)} runs with both dumps "
          f"(mean | runs up | 95% image bootstrap of the mean) ==")
    if ipairs:
        ig = group_deltas(ipairs, masks, fgroups, args.boot_groups)
        res["interp_groups"] = {"runs": [p[0] for p in ipairs], **ig}
        for s in masks:
            for g in ("macro", "head", "medium", "tail", "extreme_tail", "rare10"):
                c = ig[s][g]
                print(f"{s:8s} {g:13s} {c['mean']:+.4f} ({c['up']:2d}/{c['n']} up) [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]")
        names, prev = meta["class_names"], meta["train_prevalence"]
        print(f"\n   per class, rarest first: val+ all / frontal | area - lanczos mean (runs up), all | frontal")
        for i in np.argsort(prev):
            print(f"   {names[i][:28]:28s} {int(L[:, i].sum()):5d} {int(L[fr, i].sum()):5d} | "
                  f"{ig['all']['per_class_mean'][i]:+.4f} ({ig['all']['per_class_up'][i]:2d}) | "
                  f"{ig['frontal']['per_class_mean'][i]:+.4f} ({ig['frontal']['per_class_up'][i]:2d})")

    res["arms"] = {}
    print("\n== 2026-10-16 training arms vs seeded ASL / Entropy (single runs; 95% bootstrap over images) ==")
    pairs = [(("arms", a, 42, it), ("seeded", "A", 42, bit), f"{a} ({it} eval vs ASL {bit})")
             for a, it, bit in (("trainarea", "area", "area"), ("trainarea", "area", "lanczos"),
                                ("sched4", "lanczos", "lanczos"), ("sched4", "area", "area"))]
    pairs += [(("seeded", n, 42, it), ("seeded", ref, 42, it), f"{n} vs {ref} ({it})")
              for n in ("N", "NA", "NR") for ref in ("A", "C") for it in INTERPS]
    for x, y, tag in pairs:
        if x not in get or y not in get:
            print(f"{tag}: not available yet")
            continue
        ci = boot_diff(probs[x], probs[y], L, args.boot)
        cf = boot_diff(probs[x][fr], probs[y][fr], L[fr], args.boot, seed=1)
        gd = group_deltas([(tag, fa(x), fa(y))], masks, fgroups, args.boot_groups, seed=2)
        a, b = get[x], get[y]
        res["arms"][tag] = {
            "run": a, "baseline": b, "d_mAP": a["mAP"] - b["mAP"], "ci": ci,
            "d_frontal_mAP": a["frontal_mAP"] - b["frontal_mAP"], "ci_frontal": cf, "groups": gd}
        print(f"{tag:34s} mAP {a['mAP']:.4f} vs {b['mAP']:.4f}: {a['mAP'] - b['mAP']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
              f"frontal {a['frontal_mAP']:.4f} vs {b['frontal_mAP']:.4f}: {a['frontal_mAP'] - b['frontal_mAP']:+.4f} "
              f"[{cf[0]:+.4f}, {cf[1]:+.4f}] | mECE {a['mECE']:.4f} vs {b['mECE']:.4f}")
        for s in masks:
            print(f"{'':34s} {s:8s} " + " | ".join(
                f"{g} {gd[s][g]['mean']:+.4f} [{gd[s][g]['ci'][0]:+.4f}, {gd[s][g]['ci'][1]:+.4f}]"
                for g in ("head", "medium", "tail", "rare10")))

    args.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
