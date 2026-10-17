"""Three-system benchmark at 768 px on the job-49375 recipe (2026-10-13).

Plan: docs/plan/2026-10-13-asl-v0-vs-entropy-v5-plan.md. Systems and the script that trains each:

  A  ASL only                       train/train_2_v0.py  (scripts/submit_stage2_v0.sh, dumps in probs_v0_flip)
  C  Triplet Memory + Entropy       train/train_2_v5.py  (scripts/submit_stage2_v5_unified.sh)
  D  C + Markov prior, scored twice: history available ("with", a hospital archive) and withheld ("zero", the
     condition of the challenge test set, whose patients have no history)
                                    train/train_2_v5.py  (seeds 42/1024 = prior/img768_qprior_*)

Seeds 42/123/456/789/1024. All rows are evaluated the same way: SWA weights, flip TTA, one row order (keys.txt /
labels.npy of analysis/out/probs_unified_flip, identical copy in probs_v0_flip). Seven metrics from those dumps:
mAP, mAUC, mF1@0.5, mECE (15 bins), head / medium / tail AP (evaluate/class_groups.py). ASL and the v5 rows are
never pooled: different training scripts, different dump directories.

Contrasts (x - y, paired by seed; mean difference, 95% t-interval, exact sign-flip p, wins, paired-t MDE at 80%
power; Holm over the mAP p-values): C-A, D(with)-A, D(zero)-A, D(with)-C, D(zero)-C. Everything is reported
whatever it shows. The same-seed reruns prior/img768_qprior_seed{42,1024}_rerun give the run-to-run noise floor.

    python -I analysis/unified_benchmark.py                  # all runs must exist
    python -I analysis/unified_benchmark.py --allow-missing  # preliminary read, writes analysis/out only
Outputs: analysis/out/unified_benchmark.json, unified_macros.tex, unified_table.tex (A, C, D rows) (copied to
docs/paper/isbi_submit/generated/ only when nothing is missing).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entropy_ablation as ea  # noqa: E402  (tm_metrics, group_means, paired, summarize, fmt, class groups)

ROOT = ea.ROOT
UNI = ROOT / "analysis/out/probs_unified_flip"   # train_2_v5.py systems (C, D)
V0 = ROOT / "analysis/out/probs_v0_flip"         # train_2_v0.py system (A)
SEEDS = (42, 123, 456, 789, 1024)
PX = 768

ROWS = {   # row key -> (label in the table, macro word)
    "A": ("ASL only (asymmetric loss)", "A"),
    "C": ("Triplet Memory + Entropy", "C"),
    "Dwith": ("Triplet Memory + Entropy + Markov prior, history available", "DWith"),
    "Dzero": ("Triplet Memory + Entropy + Markov prior, history withheld", "DZero"),
}
MAIN_ROWS = ("A", "C", "Dwith", "Dzero")
# x - y contrasts: (name used in macros, x, y)
CONTRASTS = [("CminusA", "C", "A"), ("DwithMinusA", "Dwith", "A"), ("DzeroMinusA", "Dzero", "A"),
             ("DwithMinusC", "Dwith", "C"), ("DzeroMinusC", "Dzero", "C")]


def member_file(key: str, seed: int) -> Path:
    """Dump file of one (row, seed). evaluate_tta_v5.py names a dump after the two directory levels above
    Model_run, so ASL (checkpoint_ASL/v0_768/seed_S) and the v5 systems each have their own names; the
    seed 42/1024 prior runs live under prior/img768_qprior_seed<S>."""
    suffix = f"@{PX}_flip"
    arm, mode = key[0], key[1:]
    tag = {"with": "_prior-with", "zero": "_prior-zero", "": ""}[mode]
    if arm == "A":
        return V0 / f"v0_{PX}__seed_{seed}__model_swa{suffix}{tag}.npy"
    if arm == "D" and seed in (42, 1024):
        return UNI / f"prior__img{PX}_qprior_seed{seed}__model_swa{suffix}{tag}.npy"
    return UNI / f"{arm.lower()}__seed_{seed}__model_swa{suffix}{tag}.npy"


def load(allow_missing: bool):
    L = np.load(UNI / "labels.npy").astype(np.int64)
    meta = ea.class_groups.build()
    groups = {g: np.where(meta["groups"][g])[0] for g in ("head", "medium", "tail", "extreme_tail")}
    if (V0 / "labels.npy").exists():
        assert (np.load(V0 / "labels.npy").astype(np.int64) == L).all(), "row order differs between dump dirs"
    if (V0 / "keys.txt").exists() and (UNI / "keys.txt").exists():
        assert (V0 / "keys.txt").read_text() == (UNI / "keys.txt").read_text(), "keys.txt differs"
    runs, missing = {}, []
    for key in ROWS:
        for seed in SEEDS:
            f = member_file(key, seed)
            if not f.exists():
                missing.append(f"{key}:{seed}")
                continue
            P = np.load(f).astype(np.float64)
            assert P.shape == L.shape, f"{f}: shape {P.shape} != labels {L.shape}"
            m = ea.tm_metrics(P, L)
            runs.setdefault(key, {})[seed] = {
                **{k: m[k] for k in ("mAP", "mAUC", "mF1", "mECE")}, **ea.group_means(m["ap"], groups),
                "per_class_ap": [float(x) for x in m["ap"]], "file": str(f)}
    if missing and not allow_missing:
        sys.exit(f"missing runs (use --allow-missing for a preliminary read): {missing}")
    return L, runs, missing


def mde(d: np.ndarray):
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return None
    return float((stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * d.std(ddof=1) / np.sqrt(n))


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def write_latex(res: dict, paths: list[Path]) -> None:
    mac = ["% generated by analysis/unified_benchmark.py on " + res["generated"] + " -- do not edit"]
    for key, (_, word) in ROWS.items():
        d = res["arms"].get(key)
        if not d:
            continue
        mac.append(f"\\newcommand{{\\UniN{word}}}{{{d['mAP']['n']}}}")
        for met, mm in ea.MACRO_MET.items():
            mac.append(f"\\newcommand{{\\Uni{word}{mm}}}{{{ea.fmt(d[met]['mean'])}}}")
            mac.append(f"\\newcommand{{\\Uni{word}{mm}sd}}{{{ea.fmt(d[met]['sd'])}}}")
    for name, c in res["contrasts"].items():
        for met, mm in ea.MACRO_MET.items():
            s = c["paired"].get(met)
            if not s:
                continue
            ci = s.get("ci", [None, None])
            mac += [f"\\newcommand{{\\UniDiff{name}{mm}}}{{{ea.fmt(s['mean'], sign=True)}}}",
                    f"\\newcommand{{\\UniDiff{name}{mm}lo}}{{{ea.fmt(ci[0], sign=True)}}}",
                    f"\\newcommand{{\\UniDiff{name}{mm}hi}}{{{ea.fmt(ci[1], sign=True)}}}",
                    f"\\newcommand{{\\UniDiff{name}{mm}p}}{{{ea.fmt(s.get('p_signflip'), 3)}}}",
                    f"\\newcommand{{\\UniDiff{name}{mm}wins}}{{{s['wins']}/{s['n']}}}"]
        if c.get("mde_mAP") is not None:
            mac.append(f"\\newcommand{{\\UniDiff{name}MAPmde}}{{{c['mde_mAP']:.4f}}}")
        if c.get("holm_p_mAP") is not None:
            mac.append(f"\\newcommand{{\\UniDiff{name}MAPholm}}{{{ea.fmt(c['holm_p_mAP'], 3)}}}")
    def row(key, label):
        d = res["arms"].get(key)
        if not d:
            return None
        cells = [f"{ea.fmt(d[m]['mean'], 3)}" + (f"{{\\scriptsize$\\pm${ea.fmt(d[m]['sd'], 3)}}}" if d[m]["sd"] is not None else "")
                 for m in ea.METRICS]
        return f"{label} & {d['mAP']['n']} & " + " & ".join(cells)

    def table(keys):
        rows = [row(k, ROWS[k][0]) for k in keys]
        rows = [r + " \\\\" for r in rows if r]
        # no "\\\\" after the last row (a "\\\\" at the end of an \input file breaks the tabular)
        txt = "\n".join(rows).rstrip()
        if txt.endswith("\\\\"):
            txt = txt[:-2].rstrip()
        return ("% generated by analysis/unified_benchmark.py on " + res["generated"] + " -- do not edit\n"
                "% columns: system & n & mAP & mAUC & mF1 & mECE & head AP & medium AP & tail AP\n" + txt + "\n")

    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
        (p / "unified_macros.tex").write_text("\n".join(mac) + "\n")
        (p / "unified_table.tex").write_text(table(MAIN_ROWS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--out", type=Path, default=ea.OUT)
    ap.add_argument("--paper-gen", type=Path, default=ea.PAPER_GEN)
    args = ap.parse_args()

    L, runs, missing = load(args.allow_missing)
    res: dict = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "px": PX, "seeds": list(SEEDS),
                 "missing": missing, "n_val": int(L.shape[0]), "arms": {}, "contrasts": {}}
    for key, d in runs.items():
        res["arms"][key] = {met: ea.summarize([d[s][met] for s in sorted(d)]) for met in ea.METRICS}
        res["arms"][key]["seeds"] = sorted(d)
    for name, x, y in CONTRASTS:
        if x not in runs or y not in runs:
            continue
        common = sorted(set(runs[x]) & set(runs[y]))
        if len(common) < 2:
            continue
        c = {"x": x, "y": y, "seeds": common, "paired": {}}
        for met in ea.METRICS:
            c["paired"][met] = ea.paired(np.array([runs[x][s][met] - runs[y][s][met] for s in common]))
        c["mde_mAP"] = mde(np.array([runs[x][s]["mAP"] - runs[y][s]["mAP"] for s in common]))
        res["contrasts"][name] = c
    hp = holm({n: c["paired"]["mAP"]["p_signflip"] for n, c in res["contrasts"].items()
               if "p_signflip" in c["paired"]["mAP"]})
    for n, p in hp.items():
        res["contrasts"][n]["holm_p_mAP"] = p
    res["runs"] = {k: {str(s): v for s, v in d.items()} for k, d in runs.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "unified_benchmark.json").write_text(json.dumps(res, indent=1))
    if res["arms"]:
        write_latex(res, [args.out] + ([args.paper_gen] if args.paper_gen.parent.exists() and not missing else []))

    print(f"missing: {missing or 'none'} | val N={res['n_val']}")
    print(f"{'row':8s} {'n':>2s} " + " ".join(f"{m:>15s}" for m in ea.METRICS))
    for key in ROWS:
        d = res["arms"].get(key)
        if d:
            print(f"{key:8s} {d['mAP']['n']:2d} " + " ".join(f"{d[m]['mean']:.4f}±{(d[m]['sd'] or 0):.4f}" for m in ea.METRICS))
    for name, c in res["contrasts"].items():
        s = c["paired"]["mAP"]
        ci = s.get("ci", [None, None])
        print(f"{name:16s} (n={s['n']}) mAP {s['mean']:+.4f} CI [{ea.fmt(ci[0], sign=True)}, {ea.fmt(ci[1], sign=True)}] "
              f"p {ea.fmt(s.get('p_signflip'), 3)} holm {ea.fmt(c.get('holm_p_mAP'), 3)} wins {s['wins']}/{s['n']}  "
              f"MDE {ea.fmt(c['mde_mAP'])}")
    print(f"\nwrote {args.out / 'unified_benchmark.json'}")


if __name__ == "__main__":
    main()
