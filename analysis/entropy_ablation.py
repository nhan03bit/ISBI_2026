"""Entropy ablation at 512 px: the supervisor's like-for-like test (2026-10-08).

baseline -> triplet/memory bank without entropy -> triplet/memory bank with entropy, with the
resolution (512 px), schedule (v3 seed-sweep recipe), Stage-1 checkpoint and seeds held fixed.
Plan / pre-registration: docs/plan/2026-10-08-entropy-ablation-isbi-submit-plan.md

  asl         A   ASL only (--triplet-lambda 0)          train/checkpoint_Triplet_3/ablation512/asl/
  nogate      B   triplet + bank, all anchors            .../ablation512/nogate/
  randgate    B'  triplet + bank, random anchors, same count as the gate   .../ablation512/randgate/
  gate        C   triplet + bank + entropy gate (method) train/checkpoint_Triplet_3/seedsweep/
  gate_rerun  C   reruns of seeds 86/1024 on today's code (reproducibility)   .../ablation512/gate/

Inputs per run: preds.<tag>.npz from evaluate/evaluate.py --save-preds (scripts/eval_perclass.sh,
model_best.pth = EMA weights of the best epoch, no TTA), history.json and the Slurm log.

Primary contrast (pre-registered): gate - nogate on mAP, paired by seed. Secondary: gate - randgate,
nogate - asl, gate - asl (+ randgate - asl, descriptive), on mAP, mAUC, mF1, mECE and
head/medium/tail AP (evaluate/class_groups.py: train prevalence >10% / 1-10% / <1%).
Per contrast: mean paired difference, 95% t-CI, exact two-sided sign-flip p, wins; plus an
image-level paired bootstrap of the seed-averaged (ensemble) probabilities.

    python -I analysis/entropy_ablation.py                 # all runs must exist
    python -I analysis/entropy_ablation.py --allow-missing # preliminary read with the seeds done
Outputs: analysis/out/entropy_ablation.json, entropy_ablation_macros.tex, entropy_ablation_table.tex
(and copies in docs/paper/isbi_submit/generated/, but only when no run is missing, so a
preliminary read can never leak into the paper).
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torchmetrics.classification import (BinaryCalibrationError, MultilabelAUROC,
                                         MultilabelAveragePrecision, MultilabelF1Score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluate"))
import class_groups  # noqa: E402

CK = ROOT / "train/checkpoint_Triplet_3"
LOGS = Path("/home/psytp7/logs")
OUT = ROOT / "analysis/out"
PAPER_GEN = ROOT / "docs/paper/isbi_submit/generated"
KEY_ORDER = OUT / "val_key_order.npy"
PADCHEST = Path("/data/psytp7/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv")
FRONTAL = ("PA", "AP", "AP_horizontal")

SEEDS = (42, 86, 123, 456, 789, 1024)
RERUN_SEEDS = (86, 1024)
ARMS = ("asl", "nogate", "randgate", "gate")
ARM_LABEL = {"asl": "A: ASL only", "nogate": "B: triplet+bank, no gate",
             "randgate": "B': triplet+bank, random gate", "gate": "C: triplet+bank, entropy gate",
             "gate_rerun": "C rerun"}
CONTRASTS = [("gate", "nogate", True), ("gate", "randgate", False), ("nogate", "asl", False),
             ("gate", "asl", False), ("randgate", "asl", False)]
METRICS = ("mAP", "mAUC", "mF1", "mECE", "head", "medium", "tail")


# ---- run discovery ------------------------------------------------------------------------------
def run_paths(arm: str, seed: int) -> tuple[Path, str, str]:
    """(run dir, eval tag, Slurm log glob) of one run."""
    if arm == "gate":
        return CK / f"seedsweep/seed_{seed}/Model_run", f"s{seed}", f"isbi2026_v3seed_s{seed}_*.out"
    a = "gate" if arm == "gate_rerun" else arm
    return (CK / f"ablation512/{a}/seed_{seed}/Model_run", f"{a}_s{seed}",
            f"isbi2026_abl_{a}_s{seed}_[0-9]*.out")


def parse_log(pattern: str) -> dict:
    logs = sorted(LOGS.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not logs:
        return {}
    txt = logs[-1].read_text(errors="replace")
    ep = [(int(e), float(m)) for e, m in re.findall(r"Epoch (\d+)/\d+ \| .*?mAP=([0-9.]+)", txt)]
    hit = [float(x) for x in re.findall(r"triplet hit rate = \d+/\d+ \(([0-9.]+)%\)", txt)]
    anc = [(float(a), float(t)) for a, t in
           re.findall(r"anchors/step = ([0-9.]+) triplets/step = ([0-9.]+)", txt)]
    args = re.search(r"^args: (.*)$", txt, re.M)
    return {"log": str(logs[-1]), "epoch_mAP": ep, "hit_rate_pct": hit,
            "anchors_triplets_per_step": anc, "args": args.group(1) if args else None,
            "early_stop": bool(re.search(r"Early stopping triggered", txt)),
            "finished": "Done. Best epoch" in txt}


# ---- metrics ------------------------------------------------------------------------------------
def tm_metrics(P: np.ndarray, L: np.ndarray) -> dict:
    C = P.shape[1]
    p, y = torch.from_numpy(np.ascontiguousarray(P)).float(), torch.from_numpy(L).long()
    return dict(
        mAP=float(MultilabelAveragePrecision(num_labels=C, average="macro")(p, y)),
        mAUC=float(MultilabelAUROC(num_labels=C, average="macro")(p, y)),
        mF1=float(MultilabelF1Score(num_labels=C, average="macro", threshold=0.5)(p, y)),
        mECE=float(BinaryCalibrationError(n_bins=15, norm="l1")(p.flatten(), y.flatten())),
        ap=MultilabelAveragePrecision(num_labels=C, average=None)(p, y).numpy().astype(float),
    )


def group_means(ap: np.ndarray, groups: dict) -> dict:
    return {g: float(np.mean(ap[groups[g]])) for g in ("head", "medium", "tail", "extreme_tail")}


class FastAP:
    """Weighted per-class AP / AUROC for the bootstrap (tie-grouped, as analysis/moe_compare.py)."""

    def __init__(self, P: np.ndarray, L: np.ndarray):
        self.P, self.L, self.C = P, L, P.shape[1]
        self.order = np.argsort(P, axis=0, kind="stable")
        self.grp = []
        for c in range(self.C):
            ps = P[self.order[:, c], c]
            self.grp.append(np.cumsum(np.r_[True, ps[1:] != ps[:-1]]) - 1)

    def __call__(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ap, auc = np.full(self.C, np.nan), np.full(self.C, np.nan)
        for c in range(self.C):
            o = self.order[:, c]
            y = self.L[o, c]
            ww = w[o]
            wp = ww * y
            wn = ww - wp
            Wp, Wn = wp.sum(), wn.sum()
            if Wp > 0 and Wn > 0:
                g = self.grp[c]
                ng = g[-1] + 1
                pg, ngr = np.bincount(g, wp, ng), np.bincount(g, wn, ng)
                auc[c] = (pg * (np.cumsum(ngr) - ngr + 0.5 * ngr)).sum() / (Wp * Wn)
                pr, nr = pg[::-1], ngr[::-1]
                tp, fp = np.cumsum(pr), np.cumsum(nr)
                ap[c] = (pr * tp / np.maximum(tp + fp, 1e-12)).sum() / Wp
        return ap, auc


# ---- statistics ---------------------------------------------------------------------------------
def summarize(values: list[float]) -> dict:
    v = np.asarray(values, float)
    return {"mean": float(v.mean()) if len(v) else None,
            "sd": float(v.std(ddof=1)) if len(v) > 1 else None, "n": int(len(v)),
            "values": [float(x) for x in v]}


def paired(d: np.ndarray) -> dict:
    """Mean paired difference, 95% t-CI, exact two-sided sign-flip p, wins."""
    n = len(d)
    m = float(d.mean())
    out = {"n": n, "mean": m, "diffs": [float(x) for x in d], "wins": int((d > 0).sum()),
           "losses": int((d < 0).sum())}
    if n > 1:
        sd = float(d.std(ddof=1))
        half = float(stats.t.ppf(0.975, n - 1)) * sd / np.sqrt(n)
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
        null = (signs * d).mean(1)
        out.update(sd=sd, ci=[m - half, m + half],
                   p_signflip=float((np.abs(null) >= abs(m) - 1e-12).mean()),
                   p_ttest=float(stats.ttest_1samp(d, 0.0).pvalue))
    return out


# ---- validation subsets (secondary) -------------------------------------------------------------
def subset_masks(L: np.ndarray, classes: list[str]) -> dict | None:
    """frontal / frontal & unseen-patient masks in the preds row order, or None if the row order
    cannot be confirmed. Reads only ImageID / PatientID / Projection of val/train FOLD images."""
    import pandas as pd

    if not (KEY_ORDER.exists() and PADCHEST.exists()):
        return None
    order = np.load(KEY_ORDER)
    va = pd.read_csv(ROOT / "data/val_fold_1.csv").iloc[order].reset_index(drop=True)
    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros_like(L)
    for r, labs in enumerate(va["label_list"]):
        for c in ast.literal_eval(labs):
            Y[r, idx[c]] = 1
    if not (Y == L).all():          # the guard: preds rows must be the reconstructed order
        return None
    meta = pd.read_csv(PADCHEST, usecols=["ImageID", "PatientID", "Projection"], dtype=str,
                       low_memory=False).set_index("ImageID")
    proj = meta["Projection"].reindex(va["filename"]).values
    pid = meta["PatientID"].reindex(va["filename"])
    tr = pd.read_csv(ROOT / "data/train_fold_1.csv")
    unseen = ~pid.isin(set(meta["PatientID"].reindex(tr["filename"]).dropna())).values
    front = np.isin(proj, FRONTAL)
    return {"frontal": front, "frontal_unseen": front & unseen}


def subset_map(P: np.ndarray, L: np.ndarray, mask: np.ndarray) -> float:
    keep = L[mask].sum(0) > 0                       # macro over classes with >=1 positive
    p = torch.from_numpy(np.ascontiguousarray(P[mask][:, keep])).float()
    y = torch.from_numpy(L[mask][:, keep]).long()
    return float(MultilabelAveragePrecision(num_labels=int(keep.sum()), average="macro")(p, y))


# ---- LaTeX ----------------------------------------------------------------------------------------
MACRO_ARM = {"asl": "Asl", "nogate": "Nogate", "randgate": "Randgate", "gate": "Gate",
             "gate_rerun": "Rerun"}
MACRO_MET = {"mAP": "MAP", "mAUC": "MAUC", "mF1": "MFone", "mECE": "MECE", "head": "Head",
             "medium": "Medium", "tail": "Tail"}


def fmt(x, nd=4, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def write_latex(res: dict, paths: list[Path]) -> None:
    mac = ["% generated by analysis/entropy_ablation.py on " + res["generated"] + " -- do not edit"]
    for arm, d in res["arms"].items():
        a = MACRO_ARM[arm]
        mac.append(f"\\newcommand{{\\Abl{a}N}}{{{d['mAP']['n']}}}")
        for met, mm in MACRO_MET.items():
            mac.append(f"\\newcommand{{\\Abl{a}{mm}}}{{{fmt(d[met]['mean'])}}}")
            mac.append(f"\\newcommand{{\\Abl{a}{mm}sd}}{{{fmt(d[met]['sd'])}}}")
    for key, c in res["contrasts"].items():
        x, y = key.split(" - ")
        tag = MACRO_ARM[x] + "Minus" + MACRO_ARM[y]
        for met, mm in MACRO_MET.items():
            s = c["paired"].get(met)
            if not s:
                continue
            ci = s.get("ci", [None, None])
            mac += [f"\\newcommand{{\\Abl{tag}{mm}}}{{{fmt(s['mean'], sign=True)}}}",
                    f"\\newcommand{{\\Abl{tag}{mm}lo}}{{{fmt(ci[0], sign=True)}}}",
                    f"\\newcommand{{\\Abl{tag}{mm}hi}}{{{fmt(ci[1], sign=True)}}}",
                    f"\\newcommand{{\\Abl{tag}{mm}p}}{{{fmt(s.get('p_signflip'), 3)}}}",
                    f"\\newcommand{{\\Abl{tag}{mm}wins}}{{{s['wins']}/{s['n']}}}"]
            b = c.get("bootstrap", {}).get(met)
            if b:
                mac += [f"\\newcommand{{\\Abl{tag}{mm}Ens}}{{{fmt(b['point'], sign=True)}}}",
                        f"\\newcommand{{\\Abl{tag}{mm}Enslo}}{{{fmt(b['ci'][0], sign=True)}}}",
                        f"\\newcommand{{\\Abl{tag}{mm}Enshi}}{{{fmt(b['ci'][1], sign=True)}}}"]
    rows = ["% generated by analysis/entropy_ablation.py on " + res["generated"] + " -- do not edit",
            "% columns: arm & n & mAP & mAUC & mF1 & mECE & head AP & medium AP & tail AP"]
    for arm in ARMS:
        d = res["arms"].get(arm)
        if not d:
            continue
        cells = [f"{fmt(d[m]['mean'], 3)}{{\\scriptsize$\\pm${fmt(d[m]['sd'], 3)}}}" for m in METRICS]
        rows.append(f"{ARM_LABEL[arm]} & {d['mAP']['n']} & " + " & ".join(cells) + " \\\\")
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
        (p / "entropy_ablation_macros.tex").write_text("\n".join(mac) + "\n")
        (p / "entropy_ablation_table.tex").write_text("\n".join(rows) + "\n")


# ---- main ---------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples (0 = off)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="analyse the runs that exist (preliminary); default requires all")
    args = ap.parse_args()

    meta = class_groups.build()
    classes = list(meta["class_names"])
    groups = {g: np.where(meta["groups"][g])[0] for g in ("head", "medium", "tail", "extreme_tail")}

    runs, probs, missing, L = {}, {}, [], None
    plan = [(a, s) for a in ARMS for s in SEEDS] + [("gate_rerun", s) for s in RERUN_SEEDS]
    for arm, seed in plan:
        rd, tag, logpat = run_paths(arm, seed)
        npz = rd / f"preds.{tag}.npz"
        if not npz.exists():
            missing.append(f"{arm}:{seed}")
            continue
        d = np.load(npz, allow_pickle=True)
        assert list(d["class_names"]) == classes, f"{npz}: class order differs from the shards"
        assert int(d["img_size"]) == 512, f"{npz}: evaluated at {int(d['img_size'])} px"
        lab = d["labels"].astype(np.int64)
        if L is None:
            L = lab
        assert (lab == L).all(), f"{npz}: labels / row order differ"
        P = d["probs"].astype(np.float64)
        m = tm_metrics(P, L)
        hist = json.loads((rd / "history.json").read_text())
        gap = abs(m["mAP"] - hist["best_metric"])
        assert gap < 1e-4, f"{arm}:{seed} preds mAP {m['mAP']:.6f} != history best {hist['best_metric']:.6f}"
        ev = rd / f"eval_result.{tag}.json"
        if ev.exists():
            e = json.loads(ev.read_text())
            assert abs(e["val_mAP"] - m["mAP"]) < 1e-5, f"{ev}: val_mAP differs"
            for g in ("head", "medium", "tail"):
                assert abs(e["group_mAP"][g] - group_means(m["ap"], groups)[g]) < 1e-5, f"{ev}: {g}"
        log = parse_log(logpat)
        ts = hist.get("triplet_stats") or []
        anc = log.get("anchors_triplets_per_step") or [
            (t["anchors"] / t["steps"], t["triplets"] / t["steps"]) for t in ts]
        runs.setdefault(arm, {})[seed] = {
            **{k: m[k] for k in ("mAP", "mAUC", "mF1", "mECE")}, **group_means(m["ap"], groups),
            "per_class_ap": [float(x) for x in m["ap"]], "best_epoch": hist["best_epoch"],
            "history_best": hist["best_metric"], "epoch_mAP": log.get("epoch_mAP"),
            "early_stop": log.get("early_stop"), "hit_rate_pct": log.get("hit_rate_pct"),
            "anchors_per_step": [a for a, _ in anc], "triplets_per_step": [t for _, t in anc],
            "args": log.get("args"), "log": log.get("log"), "preds": str(npz)}
        probs[(arm, seed)] = P
    if missing and not args.allow_missing:
        sys.exit(f"missing runs (use --allow-missing for a preliminary read): {missing}")
    assert L is not None, "no runs found"
    val_pos = L.sum(0)

    res = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "seeds": list(SEEDS),
           "missing": missing, "n_val": int(L.shape[0]),
           "groups": {g: [classes[i] for i in ix] for g, ix in groups.items()},
           "val_pos": {c: int(v) for c, v in zip(classes, val_pos)},
           "val_pos_group": {g: [int(val_pos[i]) for i in ix] for g, ix in groups.items()},
           "runs": {a: {str(s): r for s, r in d.items()} for a, d in runs.items()},
           "arms": {}, "contrasts": {}, "repro": {}, "subsets": {}}
    for arm, d in runs.items():
        res["arms"][arm] = {met: summarize([d[s][met] for s in sorted(d)]) for met in METRICS}
        res["arms"][arm]["seeds"] = sorted(d)
        anc = [np.mean(d[s]["anchors_per_step"]) for s in sorted(d) if d[s]["anchors_per_step"]]
        tri = [np.mean(d[s]["triplets_per_step"]) for s in sorted(d) if d[s]["triplets_per_step"]]
        res["arms"][arm]["anchors_per_step"] = summarize(anc)
        res["arms"][arm]["triplets_per_step"] = summarize(tri)
        res["arms"][arm]["best_epoch"] = summarize([d[s]["best_epoch"] for s in sorted(d)])

    # ---- paired-by-seed contrasts + bootstrap of the seed-averaged ensembles -----------------
    N = L.shape[0]
    rng = np.random.default_rng(2026)
    W = [np.bincount(rng.integers(0, N, N), minlength=N).astype(np.float64)
         for _ in range(args.boot)]
    cache = {}

    def boot_stats(arm, seeds):
        key = (arm, tuple(seeds))
        if key not in cache:
            P = np.mean([probs[(arm, s)] for s in seeds], axis=0)
            fa = FastAP(P, L)
            ap1, _ = fa(np.ones(N))
            ref = tm_metrics(P, L)
            assert np.nanmax(np.abs(ap1 - ref["ap"])) < 1e-5, f"fast AP disagrees ({arm})"
            cache[key] = (ref, [fa(w) for w in W])
        return cache[key]

    t0 = time.time()
    for x, y, primary in CONTRASTS:
        if x not in runs or y not in runs:
            continue
        common = sorted(set(runs[x]) & set(runs[y]))
        if not common:
            continue
        c = {"primary": primary, "seeds": common, "paired": {}, "bootstrap": {}}
        for met in METRICS:
            d = np.array([runs[x][s][met] - runs[y][s][met] for s in common])
            c["paired"][met] = paired(d)
        if args.boot:
            (rx, bx), (ry, by) = boot_stats(x, common), boot_stats(y, common)
            gx, gy = group_means(rx["ap"], groups), group_means(ry["ap"], groups)
            point = {"mAP": rx["mAP"] - ry["mAP"], "mAUC": rx["mAUC"] - ry["mAUC"],
                     **{g: gx[g] - gy[g] for g in ("head", "medium", "tail")}}
            for met in point:
                if met == "mAP":
                    dd = [np.nanmean(a1) - np.nanmean(a2) for (a1, _), (a2, _) in zip(bx, by)]
                elif met == "mAUC":
                    dd = [np.nanmean(u1) - np.nanmean(u2) for (_, u1), (_, u2) in zip(bx, by)]
                else:
                    ix = groups[met]
                    dd = [np.nanmean(a1[ix]) - np.nanmean(a2[ix]) for (a1, _), (a2, _) in zip(bx, by)]
                dd = np.asarray(dd)
                c["bootstrap"][met] = {"point": float(point[met]),
                                       "ci": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
                                       "frac_le0": float((dd <= 0).mean()), "B": args.boot,
                                       "ensemble": f"{len(common)}-seed probability average"}
            c["bootstrap"]["ensemble_mAP"] = {x: rx["mAP"], y: ry["mAP"]}
        c["per_class_delta_mean"] = {cls: float(np.mean([runs[x][s]["per_class_ap"][i] -
                                                         runs[y][s]["per_class_ap"][i] for s in common]))
                                     for i, cls in enumerate(classes)}
        res["contrasts"][f"{x} - {y}"] = c
        print(f"contrast {x} - {y}: {len(common)} seeds ({time.time() - t0:.0f}s)")

    # ---- reproducibility of the entropy arm (same seed, today's code) ------------------------
    if "gate_rerun" in runs and "gate" in runs:
        for s in sorted(set(runs["gate_rerun"]) & set(runs["gate"])):
            res["repro"][str(s)] = {met: runs["gate_rerun"][s][met] - runs["gate"][s][met]
                                    for met in METRICS}
        res["arms"]["gate_rerun"] = {met: summarize([runs["gate_rerun"][s][met]
                                                     for s in sorted(runs["gate_rerun"])])
                                     for met in METRICS}

    # ---- secondary: frontal and frontal & unseen-patient subsets -----------------------------
    masks = subset_masks(L, classes)
    if masks is None:
        res["subsets"] = {"note": "row order not confirmed; subsets skipped"}
    else:
        res["subsets"]["n"] = {k: int(v.sum()) for k, v in masks.items()}
        for (arm, seed), P in probs.items():
            res["runs"][arm][str(seed)]["subset_mAP"] = {k: subset_map(P, L, v) for k, v in masks.items()}
        for x, y, _ in CONTRASTS:
            if f"{x} - {y}" not in res["contrasts"]:
                continue
            common = res["contrasts"][f"{x} - {y}"]["seeds"]
            res["contrasts"][f"{x} - {y}"]["subsets"] = {
                k: paired(np.array([res["runs"][x][str(s)]["subset_mAP"][k] -
                                    res["runs"][y][str(s)]["subset_mAP"][k] for s in common]))
                for k in masks}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "entropy_ablation.json").write_text(json.dumps(res, indent=1))
    # the paper only ever receives numbers from the complete set of runs
    write_latex(res, [OUT] + ([PAPER_GEN] if PAPER_GEN.parent.exists() and not missing else []))

    # ---- console summary --------------------------------------------------------------------------
    print(f"\nmissing: {missing or 'none'} | val N={N} | positives per tail class: "
          f"{res['val_pos_group']['tail']}")
    print(f"{'arm':30s} {'n':>2s} " + " ".join(f"{m:>15s}" for m in METRICS))
    for arm in list(ARMS) + ["gate_rerun"]:
        if arm in res["arms"]:
            d = res["arms"][arm]
            print(f"{ARM_LABEL[arm]:30s} {d['mAP']['n']:2d} " + " ".join(
                f"{d[m]['mean']:.4f}±{(d[m]['sd'] or 0):.4f}" for m in METRICS))
    for key, c in res["contrasts"].items():
        s = c["paired"]["mAP"]
        b = c["bootstrap"].get("mAP")
        print(f"{key:20s}{' (PRIMARY)' if c['primary'] else '':10s} mAP {s['mean']:+.4f} "
              f"CI [{fmt(s.get('ci', [None])[0], sign=True)}, {fmt(s.get('ci', [None, None])[1], sign=True)}] "
              f"p_signflip {fmt(s.get('p_signflip'), 3)} wins {s['wins']}/{s['n']}"
              + (f" | ens {b['point']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]" if b else ""))
    print(f"\nwrote {OUT / 'entropy_ablation.json'}")


if __name__ == "__main__":
    main()
