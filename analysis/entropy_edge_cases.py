"""Edge cases where Entropy + Triplet Memory (C) might beat ASL (A), 2026-10-17.

Pre-registration (subset list, metric, tests, decision rule): docs/plan/2026-10-17-entropy-edge-cases.md, sections 1.x,
written before this script was first run. Offline: cached flip-TTA probability dumps only.

Primary    768 px, flip-TTA SWA, INTER_AREA: C - A over seeds 42 / 123 / 456 / 789 / 1024 (nominal pairing by seed).
Secondary  512 px ablation (no TTA, model_best, lanczos), seeds 42 / 86 / 123 / 456 / 789 / 1024: C - A and C - B.
Per subset (S1-S10, family F1) and per tail class (family F2): seed-level paired t interval / p / sign-flip p / wins,
95% paired image bootstrap of the seed mean (B resamples shared by every run and subset), decision p = max(p_t,
p_img), Holm within each family. Sensitivity: patient-cluster bootstrap, Welch t, lanczos 768 point estimates.

    python -I analysis/entropy_edge_cases.py [--boot 2000] [--workers 8]  -> analysis/out/entropy_edge_cases.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edge_common as ec  # noqa: E402

ea, O = ec.ea, ec.O
SEEDS768 = (42, 123, 456, 789, 1024)
SEEDS512 = (42, 86, 123, 456, 789, 1024)
SUBS = [("S1", "frontal"), ("S2", "frontal & unseen patient"), ("S3", "lateral"), ("S4", "AP_horizontal"),
        ("S5", "abnormal only (>=1 non-Normal label)"), ("S6", ">=3 findings"), ("S7", "tail group, all views"),
        ("S8", "tail group, frontal"), ("S9", "rare10, all views"), ("S10", "extreme tail, all views")]
F1 = [s for s, _ in SUBS]


def p768(arm: str, seed: int, interp: str) -> Path:
    sfx = "_area" if interp == "area" else ""
    if arm == "A":
        return O / f"probs_v0_flip{sfx}" / f"v0_768__seed_{seed}__model_swa@768_flip{sfx}.npy"
    return O / f"probs_unified_flip{sfx}" / f"c__seed_{seed}__model_swa@768_flip{sfx}.npy"


def p512(arm: str, seed: int) -> Path:
    if arm == "C":
        return ea.CK / f"seedsweep/seed_{seed}/Model_run/preds.s{seed}.npz"
    a = {"A": "asl", "B": "nogate"}[arm]
    return ea.CK / f"ablation512/{a}/seed_{seed}/Model_run/preds.{a}_s{seed}.npz"


def load768(arm, seed, interp, L):
    f = p768(arm, seed, interp)
    d = f.parent
    assert (np.load(d / "labels.npy").astype(np.int64) == L).all(), f"{d}: labels differ"
    return np.load(f).astype(np.float64)


def load512(arm, seed, L):
    d = np.load(p512(arm, seed), allow_pickle=True)
    assert int(d["img_size"]) == 512
    assert (d["labels"].astype(np.int64) == L).all(), "512 px preds: label / row order differs"
    return d["probs"].astype(np.float64)


def seed_stats(x: np.ndarray, y: np.ndarray, welch: bool) -> dict:
    """Seed-level paired statistics of x - y (x, y per-seed values)."""
    d = x - y
    out = ea.paired(d)
    out["mde80"] = ec.mde80(out["sd"], len(d))
    out["x_mean"], out["y_mean"] = float(np.mean(x)), float(np.mean(y))
    if welch:
        out["p_welch"] = float(stats.ttest_ind(x, y, equal_var=False).pvalue)
    return out


def analyse(vals: dict, boot: dict, keys: list, x: str, y: str, seeds, welch: bool, cluster: dict | None = None) -> dict:
    """Per-key statistics of the contrast x - y. vals[(arm, seed)][key] point values; boot[(arm, seed)] -> list over
    replicates of the same dicts (image bootstrap); cluster likewise for the patient-cluster bootstrap."""
    res = {}
    for k in keys:
        xs = np.array([vals[(x, s)][k] for s in seeds])
        ys = np.array([vals[(y, s)][k] for s in seeds])
        st = seed_stats(xs, ys, welch)
        for tag, bsrc in (("img", boot), ("cluster", cluster)):
            if bsrc is None:
                continue
            dstar = np.mean([bsrc[(x, s)][k] - bsrc[(y, s)][k] for s in seeds], axis=0)
            bs = ec.boot_summary(dstar)
            st[f"{tag}_ci"], st[f"{tag}_p"], st[f"{tag}_B"] = bs["ci"], bs["p"], bs["B"]
            st[f"{tag}_halfwidth"] = (bs["ci"][1] - bs["ci"][0]) / 2
        if boot is not None:
            st["p_dec"] = max(st["p_ttest"], st["img_p"])
        res[k] = st
    return res


def add_holm(res: dict, fam: list) -> None:
    adj = ec.holm({k: res[k]["p_dec"] for k in fam})
    for k in fam:
        res[k]["holm_p"] = adj[k]
        res[k]["survives"] = bool(adj[k] < 0.05)


def fmt(v, nd=4, sign=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def table(res: dict, keys: list, labels: dict, title: str, holm: bool) -> None:
    print(f"\n### {title}")
    hdr = "| id | subset | M(A) | M(C or x) | diff | seed 95% CI (wins) | p_t | p_signflip | image 95% CI | p_img | p_dec | " \
          + ("Holm | " if holm else "") + "MDE80 |"
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for k in keys:
        r = res[k]
        print(f"| {k} | {labels.get(k, k)} | {r['y_mean']:.4f} | {r['x_mean']:.4f} | {fmt(r['mean'])} | "
              f"[{fmt(r['ci'][0])}, {fmt(r['ci'][1])}] ({r['wins']}/{r['n']}) | {r['p_ttest']:.3f} | "
              f"{r['p_signflip']:.4f} | [{fmt(r['img_ci'][0])}, {fmt(r['img_ci'][1])}] | {r['img_p']:.3f} | "
              f"{r['p_dec']:.3f} | "
              + ((f"{r['holm_p']:.3f}{'*' if r['survives'] else ''} | " if "holm_p" in r else "ref | ") if holm else "")
              + f"{r['mde80']:.4f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--out", type=Path, default=O / "entropy_edge_cases.json")
    args = ap.parse_args()
    t0 = time.time()
    print(f"start {datetime.now():%Y-%m-%d %H:%M:%S} | boot {args.boot} workers {args.workers} seed {args.seed}", flush=True)

    ctx = ec.load_context()
    L, names = ctx["L"], ctx["names"]
    tail_keys = [f"C:{names[i]}" for i in ctx["tail_sorted"]]
    allkeys = ["ALL"] + F1 + tail_keys
    labels = dict(SUBS)
    labels["ALL"] = "all views, 30 classes (reference)"
    labels.update({f"C:{names[i]}": f"{names[i]} ({int(L[:, i].sum())} val+)" for i in ctx["tail_sorted"]})

    # ---- load runs and point estimates ----------------------------------------------------------------
    runs = {}      # (set, arm, seed) -> FastAPM
    for s in SEEDS768:
        for arm in ("A", "C"):
            runs[("768a", arm, s)] = ec.FastAPM(load768(arm, s, "area", L), L)
            runs[("768l", arm, s)] = ec.FastAPM(load768(arm, s, "lanczos", L), L)
    for s in SEEDS512:
        for arm in ("A", "B", "C"):
            runs[("512", arm, s)] = ec.FastAPM(load512(arm, s, L), L)
    print(f"loaded {len(runs)} runs ({time.time() - t0:.0f}s)", flush=True)

    pap = {k: fa(ctx["mmat"]) for k, fa in runs.items()}            # (masks, classes) AP of every run
    point = {k: ec.subset_values(a, ctx) for k, a in pap.items()}
    pv = {st: {(arm, s): v for (set_, arm, s), v in point.items() if set_ == st} for st in ("768a", "768l", "512")}

    # ---- validation of the pipeline against existing numbers -----------------------------------------
    chk = {}
    for s in SEEDS768:
        for arm, stem in (("A", O / f"probs_v0_flip_area/eval_v0_s{s}.json"), ("C", O / f"probs_unified_flip_area/eval_c_s{s}.json")):
            ev = json.loads(stem.read_text())["val_mAP"]
            chk[f"768a {arm} s{s} ALL vs eval json"] = abs(pv["768a"][(arm, s)]["ALL"] - ev)
    abl = json.loads((O / "entropy_ablation.json").read_text())
    arm_name = {"A": "asl", "B": "nogate", "C": "gate"}
    for s in SEEDS512:
        for arm in ("A", "B", "C"):
            sm = abl["runs"][arm_name[arm]][str(s)]["subset_mAP"]
            chk[f"512 {arm} s{s} S1 vs entropy_ablation.json"] = abs(pv["512"][(arm, s)]["S1"] - sm["frontal"])
            chk[f"512 {arm} s{s} S2 vs entropy_ablation.json"] = abs(pv["512"][(arm, s)]["S2"] - sm["frontal_unseen"])
            chk[f"512 {arm} s{s} ALL vs entropy_ablation.json"] = abs(
                pv["512"][(arm, s)]["ALL"] - abl["runs"][arm_name[arm]][str(s)]["mAP"])
    worst = max(chk.items(), key=lambda kv: kv[1])
    print(f"pipeline checks: {len(chk)} comparisons, worst |diff| = {worst[1]:.2e} ({worst[0]})", flush=True)
    assert worst[1] < 2e-4, worst

    # ---- bootstraps ---------------------------------------------------------------------------------------
    def per_replicate(AP, run_keys):
        """AP (B, n_runs, M, C) -> {run key: {metric: array over replicates}}."""
        out = {rk: {} for rk in run_keys}
        for r, rk in enumerate(run_keys):
            vs = [ec.subset_values(AP[b, r], ctx) for b in range(AP.shape[0])]
            out[rk] = {k: np.array([v[k] for v in vs]) for k in vs[0]}
        return out

    k768 = [("768a", a, s) for s in SEEDS768 for a in ("A", "C")]
    k512 = [("512", a, s) for s in SEEDS512 for a in ("A", "B", "C")]
    t1 = time.time()
    AP768 = ec.boot_replicates([runs[k] for k in k768], ctx["mmat"], args.boot, args.seed, args.workers)
    print(f"image bootstrap 768 done ({time.time() - t1:.0f}s)", flush=True)
    t1 = time.time()
    APcl = ec.boot_replicates([runs[k] for k in k768], ctx["mmat"], args.boot, args.seed + 1, args.workers,
                              cluster=(ctx["pid_codes"], ctx["n_patients"]))
    print(f"patient-cluster bootstrap 768 done ({time.time() - t1:.0f}s)", flush=True)
    t1 = time.time()
    AP512 = ec.boot_replicates([runs[k] for k in k512], ctx["mmat"], args.boot, args.seed, args.workers)
    print(f"image bootstrap 512 done ({time.time() - t1:.0f}s)", flush=True)
    b768 = per_replicate(AP768, k768)
    bcl = per_replicate(APcl, k768)
    b512 = per_replicate(AP512, k512)
    to_arm = lambda d, st: {(k[1], k[2]): v for k, v in d.items() if k[0] == st}  # noqa: E731
    b768, bcl, b512 = to_arm(b768, "768a"), to_arm(bcl, "768a"), to_arm(b512, "512")

    # ---- statistics ---------------------------------------------------------------------------------------
    R = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "boot": args.boot, "seed": args.seed,
         "checks_worst_abs_diff": worst[1], "n_val": len(L),
         "n_images": {m: int(v.sum()) for m, v in ctx["masks"].items()}, "labels": labels,
         "tail_classes": [names[i] for i in ctx["tail_sorted"]],
         "val_pos_tail": {names[i]: int(L[:, i].sum()) for i in ctx["tail_sorted"]},
         "val_pos_tail_frontal": {names[i]: int(L[ctx["masks"]["frontal"], i].sum()) for i in ctx["tail_sorted"]}}
    prim = analyse(pv["768a"], b768, allkeys, "C", "A", SEEDS768, True, cluster=bcl)
    add_holm(prim, F1)
    add_holm(prim, tail_keys)
    R["primary_768_area_C_minus_A"] = prim
    lz = {}
    for k in allkeys:
        lz[k] = seed_stats(np.array([pv["768l"][("C", s)][k] for s in SEEDS768]),
                           np.array([pv["768l"][("A", s)][k] for s in SEEDS768]), True)
    R["sens_768_lanczos_C_minus_A"] = lz
    sec = {}
    for nm, (x, y) in {"C-A": ("C", "A"), "C-B": ("C", "B")}.items():
        r = analyse(pv["512"], b512, allkeys, x, y, SEEDS512, False)
        add_holm(r, F1)
        add_holm(r, tail_keys)
        sec[nm] = r
    R["secondary_512"] = sec

    # frontal per-class tail values at 768 (context only, no tests)
    fr = {}
    for i in ctx["tail_sorted"]:
        a = np.array([pap[("768a", "A", s)][ec.MIDX["frontal"], i] for s in SEEDS768])
        c = np.array([pap[("768a", "C", s)][ec.MIDX["frontal"], i] for s in SEEDS768])
        fr[names[i]] = {"A": float(a.mean()), "C": float(c.mean()), "diff": float((c - a).mean()),
                        "wins": int(((c - a) > 0).sum())}
    R["context_frontal_per_class_tail_768"] = fr

    # consistency table (point estimates)
    cons = {}
    for k in F1 + tail_keys:
        s768, s512a, s512b = prim[k]["mean"], sec["C-A"][k]["mean"], sec["C-B"][k]["mean"]
        cons[k] = {"768_C-A": s768, "512_C-A": s512a, "512_C-B": s512b,
                   "all_positive": bool(s768 > 0 and s512a > 0 and s512b > 0),
                   "all_negative": bool(s768 < 0 and s512a < 0 and s512b < 0),
                   "lanczos768_C-A": lz[k]["mean"]}
    R["consistency"] = cons
    args.out.write_text(json.dumps(R, indent=1, default=float))

    # ---- print --------------------------------------------------------------------------------------------
    print("\nn images:", R["n_images"])
    table(prim, ["ALL"] + F1, labels, "PRIMARY 768 px, area, C - A, 5 seeds (family F1; * = survives Holm)", True)
    table(prim, tail_keys, labels, "PRIMARY 768 px, area, C - A, per tail class (family F2)", True)
    for nm in ("C-A", "C-B"):
        table(sec[nm], ["ALL"] + F1, labels, f"SECONDARY 512 px, lanczos, no TTA, {nm}, 6 seeds (family F1)", True)
        table(sec[nm], tail_keys, labels, f"SECONDARY 512 px, {nm}, per tail class (family F2)", True)
    print("\n### sensitivity at 768 px C - A")
    print("| id | lanczos diff | lanczos seed CI (wins) | area diff | area cluster-bootstrap CI | area image CI | Welch p (area) | paired t p (area) |")
    print("|---|---|---|---|---|---|---|---|")
    for k in ["ALL"] + F1 + tail_keys:
        p, l = prim[k], lz[k]
        print(f"| {k} | {fmt(l['mean'])} | [{fmt(l['ci'][0])}, {fmt(l['ci'][1])}] ({l['wins']}/5) | {fmt(p['mean'])} | "
              f"[{fmt(p['cluster_ci'][0])}, {fmt(p['cluster_ci'][1])}] | [{fmt(p['img_ci'][0])}, {fmt(p['img_ci'][1])}] | "
              f"{p['p_welch']:.3f} | {p['p_ttest']:.3f} |")
    print("\n### consistency of sign (768 area C-A | 512 C-A | 512 C-B)")
    for k in F1 + tail_keys:
        c = cons[k]
        print(f"{k:28s} {fmt(c['768_C-A'])} {fmt(c['512_C-A'])} {fmt(c['512_C-B'])}  "
              f"{'ALL POSITIVE' if c['all_positive'] else ('all negative' if c['all_negative'] else '')}")
    print("\nfrontal per-class tail context (768 area, 5-seed mean):")
    for n, v in fr.items():
        print(f"  {n:26s} A {v['A']:.4f} C {v['C']:.4f} diff {v['diff']:+.4f} wins {v['wins']}/5")
    print(f"\nwrote {args.out} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
