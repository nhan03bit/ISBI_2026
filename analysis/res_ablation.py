"""Resolution sweep of the ASL-only vs entropy-gate contrast, 640-1024 px (2026-10-12).

Plan / pre-registration: docs/plan/2026-10-12-res-sweep-plan.md (written before any sweep run was
submitted). Question: does the entropy-gated triplet + memory-bank objective (arm C, "gate") beat the
ASL-only control (arm A, "asl") at higher resolution, where the 512 px ablation (6 seeds, MDE ~0.005 mAP)
found nothing? Same recipe at every resolution (scripts/submit_stage2_v3_res_ablation.sh: 4 epochs of an
8-epoch cosine, drop-path 0.1, effective batch 48), one flag apart; seeds 42 / 123 / 456.

Per run: mAP, mAUC, mF1@0.5, mECE, head/medium/tail AP recomputed from preds.<tag>.npz (EMA weights of the
best epoch, no TTA, evaluated at the training resolution), checked against history.json (1e-4); the SWA
validation mAP from the Slurm log is carried as a secondary number.

Contrasts, always gate - asl, paired by seed:
  * per resolution (n = 3 seeds): mean difference, 95% t-interval, exact sign-flip p (min p = 0.25), wins;
  * POOLED, the pre-registered primary: all (resolution, seed) pairs (n = 12): same statistics, sign-flip over
    2^12 patterns. Pairs of one seed share data order / initialisation across resolutions, so the interval is
    somewhat optimistic; the conservative version (per-seed mean over resolutions, n = 3) is reported next to it;
  * a descriptive trend of the paired difference against resolution (exploratory, never a headline).

    python -I analysis/res_ablation.py                 # all 24 runs must exist
    python -I analysis/res_ablation.py --allow-missing # preliminary read; writes analysis/out only
Outputs: analysis/out/res_ablation.json, res_ablation_macros.tex, res_ablation_table.tex (copies go to
docs/paper/isbi_submit/generated/ only when no run is missing).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entropy_ablation as ea  # noqa: E402  (metrics, paired(), summarize(), class groups, LaTeX helpers)

ROOT = ea.ROOT
RES = (640, 768, 896, 1024)
SEEDS = (42, 123, 456)
ARMS = ("asl", "gate")
RES_WORD = {640: "SixFourZero", 768: "SevenSixEight", 896: "EightNineSix", 1024: "OneZeroTwoFour"}
ARM_WORD = {"asl": "Asl", "gate": "Gate"}


def run_dir(root: Path, arm: str, res: int, seed: int) -> Path:
    return root / arm / f"img{res}_seed{seed}" / "Model_run"


def tag(arm: str, res: int, seed: int) -> str:
    return f"{arm}_{res}_s{seed}"


def swa_map(arm: str, res: int, seed: int) -> float | None:
    logs = sorted(ea.LOGS.glob(f"isbi2026_resabl_{arm}_{res}_s{seed}_[0-9]*.out"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    m = re.findall(r"SWA weights -> .*? mAP=([0-9.]+)", logs[-1].read_text(errors="replace"))
    return float(m[-1]) if m else None


def load_runs(root: Path, classes: list[str], groups: dict):
    runs, missing, L = {}, [], None
    for res in RES:
        for arm in ARMS:
            for seed in SEEDS:
                rd = run_dir(root, arm, res, seed)
                npz = rd / f"preds.{tag(arm, res, seed)}.npz"
                if not npz.exists():
                    missing.append(f"{arm}:{res}:{seed}")
                    continue
                d = np.load(npz, allow_pickle=True)
                assert list(d["class_names"]) == classes, f"{npz}: class order differs"
                assert int(d["img_size"]) == res, f"{npz}: evaluated at {int(d['img_size'])} px, expected {res}"
                lab = d["labels"].astype(np.int64)
                if L is None:
                    L = lab
                assert (lab == L).all(), f"{npz}: labels / row order differ"
                P = d["probs"].astype(np.float64)
                m = ea.tm_metrics(P, L)
                hist = json.loads((rd / "history.json").read_text())
                gap = abs(m["mAP"] - hist["best_metric"])
                assert gap < 1e-4, f"{arm}:{res}:{seed} preds mAP {m['mAP']:.6f} != history best {hist['best_metric']:.6f}"
                ts = hist.get("triplet_stats") or []
                runs.setdefault((arm, res), {})[seed] = {
                    **{k: m[k] for k in ("mAP", "mAUC", "mF1", "mECE")}, **ea.group_means(m["ap"], groups),
                    "per_class_ap": [float(x) for x in m["ap"]], "best_epoch": hist["best_epoch"],
                    "swa_mAP": swa_map(arm, res, seed),
                    "anchors_per_step": [t["anchors"] / t["steps"] for t in ts],
                    "triplets_per_step": [t["triplets"] / t["steps"] for t in ts],
                    "preds": str(npz)}
    return runs, missing, L


def mde(d: np.ndarray) -> float | None:
    """Smallest paired mean difference a two-sided 5% paired t-test detects with 80% power at the observed s.d."""
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return None
    return float((stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * d.std(ddof=1) / np.sqrt(n))


def write_latex(res: dict, paths: list[Path]) -> None:
    mac = ["% generated by analysis/res_ablation.py on " + res["generated"] + " -- do not edit"]
    for (arm, r), d in res["cells"].items():
        for met, mm in ea.MACRO_MET.items():
            mac.append(f"\\newcommand{{\\Rs{ARM_WORD[arm]}{RES_WORD[r]}{mm}}}{{{ea.fmt(d[met]['mean'])}}}")
            mac.append(f"\\newcommand{{\\Rs{ARM_WORD[arm]}{RES_WORD[r]}{mm}sd}}{{{ea.fmt(d[met]['sd'])}}}")
    blocks = [(f"Diff{RES_WORD[r]}", c) for r, c in res["per_res"].items()]
    blocks += [("Pool", res["pooled"]), ("SeedMean", res["seed_mean"])]
    for name, c in blocks:
        for met, mm in ea.MACRO_MET.items():
            s = c.get("paired", {}).get(met)
            if not s:
                continue
            ci = s.get("ci", [None, None])
            mac += [f"\\newcommand{{\\Rs{name}{mm}}}{{{ea.fmt(s['mean'], sign=True)}}}",
                    f"\\newcommand{{\\Rs{name}{mm}lo}}{{{ea.fmt(ci[0], sign=True)}}}",
                    f"\\newcommand{{\\Rs{name}{mm}hi}}{{{ea.fmt(ci[1], sign=True)}}}",
                    f"\\newcommand{{\\Rs{name}{mm}p}}{{{ea.fmt(s.get('p_signflip'), 3)}}}",
                    f"\\newcommand{{\\Rs{name}{mm}wins}}{{{s['wins']}/{s['n']}}}"]
        if c.get("mde_mAP") is not None:
            mac.append(f"\\newcommand{{\\Rs{name}MAPmde}}{{{c['mde_mAP']:.4f}}}")
    rows = []
    for r in RES:
        a, g, c = res["cells"].get(("asl", r)), res["cells"].get(("gate", r)), res["per_res"].get(r)
        if not (a and g and c):
            continue
        s, t = c["paired"]["mAP"], c["paired"]["tail"]
        ci = s.get("ci", [None, None])
        rows.append(f"{r} & {a['mAP']['n']} & {ea.fmt(a['mAP']['mean'], 3)}{{\\scriptsize$\\pm${ea.fmt(a['mAP']['sd'], 3)}}} & "
                    f"{ea.fmt(g['mAP']['mean'], 3)}{{\\scriptsize$\\pm${ea.fmt(g['mAP']['sd'], 3)}}} & "
                    f"${ea.fmt(s['mean'], 4, sign=True)}$ & $[{ea.fmt(ci[0], 4, sign=True)},{ea.fmt(ci[1], 4, sign=True)}]$ & "
                    f"${ea.fmt(t['mean'], 4, sign=True)}$")
    p = res["pooled"]["paired"]["mAP"]
    rows.append(f"pooled & {p['n']} & & & ${ea.fmt(p['mean'], 4, sign=True)}$ & "
                f"$[{ea.fmt(p['ci'][0], 4, sign=True)},{ea.fmt(p['ci'][1], 4, sign=True)}]$ & "
                f"${ea.fmt(res['pooled']['paired']['tail']['mean'], 4, sign=True)}$")
    # no "\\" after the last row: a "\\" at the end of an \input file breaks the tabular (see entropy_ablation.py)
    table = ("% generated by analysis/res_ablation.py on " + res["generated"] + " -- do not edit\n"
             "% columns: px & seeds & A mAP & C mAP & C-A mAP & 95% CI & C-A tail AP\n" + " \\\\\n".join(rows) + "\n")
    for pth in paths:
        pth.mkdir(parents=True, exist_ok=True)
        (pth / "res_ablation_macros.tex").write_text("\n".join(mac) + "\n")
        (pth / "res_ablation_table.tex").write_text(table)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "train/checkpoint_Triplet_3/resablation")
    ap.add_argument("--out", type=Path, default=ea.OUT)
    ap.add_argument("--paper-gen", type=Path, default=ea.PAPER_GEN)
    ap.add_argument("--allow-missing", action="store_true", help="analyse the runs that exist (preliminary)")
    args = ap.parse_args()

    meta = ea.class_groups.build()
    classes = list(meta["class_names"])
    groups = {g: np.where(meta["groups"][g])[0] for g in ("head", "medium", "tail", "extreme_tail")}
    runs, missing, L = load_runs(args.root, classes, groups)
    if missing and not args.allow_missing:
        sys.exit(f"missing runs (use --allow-missing for a preliminary read): {missing}")
    assert L is not None, "no runs found"

    res: dict = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "res": list(RES), "seeds": list(SEEDS),
                 "missing": missing, "n_val": int(L.shape[0]), "cells": {}, "per_res": {}, "pooled": {},
                 "seed_mean": {}, "trend": {}}
    for (arm, r), d in runs.items():
        cell = {met: ea.summarize([d[s][met] for s in sorted(d)]) for met in ea.METRICS}
        cell["swa_mAP"] = ea.summarize([d[s]["swa_mAP"] for s in sorted(d) if d[s]["swa_mAP"] is not None])
        cell["best_epoch"] = ea.summarize([d[s]["best_epoch"] for s in sorted(d)])
        res["cells"][(arm, r)] = cell

    # per resolution, gate - asl paired by seed
    pairs = []   # (res, seed, per-metric differences)
    for r in RES:
        a, g = runs.get(("asl", r), {}), runs.get(("gate", r), {})
        common = sorted(set(a) & set(g))
        if not common:
            continue
        c = {"seeds": common, "paired": {}}
        for met in ea.METRICS:
            c["paired"][met] = ea.paired(np.array([g[s][met] - a[s][met] for s in common]))
        c["mde_mAP"] = mde(np.array([g[s]["mAP"] - a[s]["mAP"] for s in common]))
        sw = [g[s]["swa_mAP"] - a[s]["swa_mAP"] for s in common
              if g[s]["swa_mAP"] is not None and a[s]["swa_mAP"] is not None]
        c["swa_mAP_diff"] = ea.summarize(sw)
        res["per_res"][r] = c
        for s in common:
            pairs.append((r, s, {met: g[s][met] - a[s][met] for met in ea.METRICS}))

    # pooled (primary) and per-seed mean over resolutions (conservative)
    if pairs:
        res["pooled"] = {"n": len(pairs), "paired": {met: ea.paired(np.array([p[2][met] for p in pairs]))
                                                     for met in ea.METRICS}}
        res["pooled"]["mde_mAP"] = mde(np.array([p[2]["mAP"] for p in pairs]))
        by_seed: dict = {}
        for r, s, dm in pairs:
            by_seed.setdefault(s, []).append(dm)
        full = {s: v for s, v in by_seed.items() if len(v) == len(RES)}
        if len(full) >= 2:
            sm = {met: np.array([np.mean([x[met] for x in full[s]]) for s in sorted(full)]) for met in ea.METRICS}
            res["seed_mean"] = {"seeds": sorted(full), "paired": {met: ea.paired(v) for met, v in sm.items()},
                                "mde_mAP": mde(sm["mAP"])}
        else:
            res["seed_mean"] = {"seeds": sorted(full), "paired": {}}
        if len({p[0] for p in pairs}) >= 3:      # descriptive trend of the paired mAP difference with resolution
            lr = stats.linregress([p[0] for p in pairs], [p[2]["mAP"] for p in pairs])
            res["trend"] = {"slope_per_100px": float(lr.slope * 100), "p": float(lr.pvalue), "n": len(pairs),
                            "note": "exploratory; pairs of one seed are not independent"}

    args.out.mkdir(parents=True, exist_ok=True)
    out = {k: v for k, v in res.items() if k not in ("cells", "per_res")}
    out["cells"] = {f"{a}:{r}": v for (a, r), v in res["cells"].items()}
    out["per_res"] = {str(r): v for r, v in res["per_res"].items()}
    out["runs"] = {f"{a}:{r}": {str(s): v for s, v in d.items()} for (a, r), d in runs.items()}
    (args.out / "res_ablation.json").write_text(json.dumps(out, indent=1))
    if pairs:
        write_latex(res, [args.out] + ([args.paper_gen] if args.paper_gen.parent.exists() and not missing else []))

    print(f"missing: {missing or 'none'} | val N={res['n_val']}")
    print(f"{'cell':12s} {'n':>2s} " + " ".join(f"{m:>15s}" for m in ea.METRICS) + "   SWA mAP")
    for r in RES:
        for arm in ARMS:
            d = res["cells"].get((arm, r))
            if d:
                print(f"{arm + ' ' + str(r):12s} {d['mAP']['n']:2d} " + " ".join(
                    f"{d[m]['mean']:.4f}±{(d[m]['sd'] or 0):.4f}" for m in ea.METRICS) +
                    (f"   {d['swa_mAP']['mean']:.4f}" if d["swa_mAP"]["n"] else "   -"))
    for r, c in res["per_res"].items():
        s = c["paired"]["mAP"]
        ci = s.get("ci", [None, None])
        print(f"gate-asl {r:5d}px  mAP {s['mean']:+.4f} CI [{ea.fmt(ci[0], sign=True)}, {ea.fmt(ci[1], sign=True)}] "
              f"p {ea.fmt(s.get('p_signflip'), 3)} wins {s['wins']}/{s['n']}  MDE {ea.fmt(c['mde_mAP'])}")
    for nm in ("pooled", "seed_mean"):
        c = res[nm]
        if c.get("paired"):
            s = c["paired"]["mAP"]
            print(f"{nm:9s} (n={s['n']})  mAP {s['mean']:+.4f} CI [{ea.fmt(s['ci'][0], sign=True)}, "
                  f"{ea.fmt(s['ci'][1], sign=True)}] p {ea.fmt(s['p_signflip'], 3)} wins {s['wins']}/{s['n']}  "
                  f"MDE {ea.fmt(c['mde_mAP'])}")
    if res["trend"]:
        print(f"trend (exploratory): {res['trend']['slope_per_100px']:+.4f} mAP per +100 px, p {res['trend']['p']:.3f}")
    print(f"\nwrote {args.out / 'res_ablation.json'}")


if __name__ == "__main__":
    main()
