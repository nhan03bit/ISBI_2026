"""Best configuration of Entropy + Triplet Memory Bank (2026-10-17): ranking of every available checkpoint and the five
fixed ensembles, the fair 5-seed ASL ensemble control, ranking stability and selection optimism.

Pre-registration (candidates, ensembles, control, submission-candidate rule): docs/result/2026-10-17-entropy-best-config.md
section 1, written before this script was first run. Offline: cached flip-TTA probability dumps (lanczos unless the
area control is named), all in the row order of analysis/out/probs_unified_flip/keys.txt.

    python -I analysis/entropy_best_config.py [--boot 2000] [--workers 8]  -> analysis/out/entropy_best_config.json
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
W01 = "probs_w01_flip"
SEEDS = (42, 123, 456, 789, 1024)
# id -> (dump relative to analysis/out, training run dir relative to train/, weights, scale, code, training job)
SINGLES = {}
for s, job in zip(SEEDS, (51832, 51843, 51857, 51871, 51883)):
    SINGLES[f"U{s}"] = (f"probs_unified_flip/c__seed_{s}__model_swa@768_flip.npy",
                        f"checkpoint_Triplet_3/unified768/c/seed_{s}/Model_run", "model_swa.pth", 768, "v5", job)
SINGLES["S42"] = ("probs_unified_flip_seeded/c__seed_42__model_swa@768_flip.npy",
                  "checkpoint_Triplet_3/unified768_seeded/c/seed_42/Model_run", "model_swa.pth", 768, "v5", 52650)
for tag, d, s, w, sc, job in (("P768-42", "img768_dp01_seed42", 42, "swa", 768, 49000),
                              ("P768-1024", "img768_dp01_seed1024", 1024, "swa", 768, 48999),
                              ("P896-86", "img896_dp01_seed86", 86, "swa", 896, 48998),
                              ("P1024-86", "img1024_dp01_seed86", 86, "swa", 1024, 48997)):
    SINGLES[tag] = (f"{W01}/push__{d}__model_swa@{sc}_flip.npy", f"checkpoint_Triplet_3/push/{d}/Model_run",
                    "model_swa.pth", sc, "v3", job)
for tag, d, sc, job in (("R640-86", "img640_seed86", 640, 47094), ("R640-1024", "img640_seed1024", 640, 47096),
                        ("R768-86", "img768_seed86", 768, 47095)):
    SINGLES[tag] = (f"{W01}/res_sweep__{d}__model_best@{sc}_flip.npy", f"checkpoint_Triplet_3/res_sweep/{d}/Model_run",
                    "model_best.pth", sc, "v3", job)
for tag, s, job in (("W512-42", 42, 47097), ("W512-1024", 1024, 46476)):
    SINGLES[tag] = (f"{W01}/seedsweep__seed_{s}__model_best@512_flip.npy",
                    f"checkpoint_Triplet_3/seedsweep/seed_{s}/Model_run", "model_best.pth", 512, "v3", job)
SINGLES["D512-86"] = (f"{W01}/dp01__seed_86__model_best@512_flip.npy",
                      "checkpoint_Triplet_3/sweep2/dp01/seed_86/Model_run", "model_best.pth", 512, "v3", 46469)
ENS = {"E1": ["U42", "U123", "U456", "U789", "U1024"],
       "E2": ["U42", "U123", "U456", "U789", "U1024", "S42"],
       "E3": ["P768-42", "P768-1024", "P896-86", "P1024-86"],
       "E4": ["U42", "U123", "U456", "U789", "U1024", "P768-42", "P768-1024", "P896-86", "P1024-86"],
       "E5": ["P768-42", "P1024-86", "R640-86"]}
ASL = {f"A{s}": f"probs_v0_flip/v0_768__seed_{s}__model_swa@768_flip.npy" for s in SEEDS}
ASL["A42-seeded"] = "probs_v0_flip_seeded/v0_768_seeded__seed_42__model_swa@768_flip.npy"
TOL = 0.002


def apply_rule(score: dict, size: dict, tol: float = TOL):
    top = max(score, key=score.get)
    within = [k for k in score if score[k] >= score[top] - tol]
    return top, within, sorted(within, key=lambda k: (size[k], -score[k]))[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--boot-ece", type=int, default=500)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--out", type=Path, default=O / "entropy_best_config.json")
    args = ap.parse_args()
    t0 = time.time()
    print(f"start {datetime.now():%Y-%m-%d %H:%M:%S} | boot {args.boot} workers {args.workers}", flush=True)

    ctx = ec.load_context()
    L, names, g = ctx["L"], ctx["names"], ctx["groups"]
    tail = g["tail"]
    N = len(L)

    def load(rel: str) -> np.ndarray:
        f = O / rel
        assert (np.load(f.parent / "labels.npy").astype(np.int64) == L).all(), f"{f.parent}: labels differ"
        P = np.load(f).astype(np.float64)
        assert P.shape == L.shape
        return P

    P = {k: load(v[0]) for k, v in SINGLES.items()}
    for k, members in ENS.items():
        P[k] = np.mean([P[m] for m in members], 0)
    pa = {k: load(v) for k, v in ASL.items()}
    P["A5"] = np.mean([pa[f"A{s}"] for s in SEEDS], 0)
    area = {"E1": np.mean([load(f"probs_unified_flip_area/c__seed_{s}__model_swa@768_flip_area.npy") for s in SEEDS], 0),
            "A5": np.mean([load(f"probs_v0_flip_area/v0_768__seed_{s}__model_swa@768_flip_area.npy") for s in SEEDS], 0)}
    size = {**{k: 1 for k in SINGLES}, **{k: len(v) for k, v in ENS.items()}}
    cand = list(SINGLES) + list(ENS)                     # the 21 eligible rows
    print(f"loaded {len(P) + len(pa) + 2} matrices ({time.time() - t0:.0f}s)", flush=True)

    # point estimates: weight columns all / frontal / frontal & unseen
    cols = ["all", "frontal", "frontal_unseen"]
    m3 = np.stack([ctx["masks"][c] for c in cols], 1).astype(np.float64)
    fapm = {k: ec.FastAPM(v, L) for k, v in P.items()}
    for k, v in pa.items():
        fapm[k] = ec.FastAPM(v, L)
    fapm["E1_area"], fapm["A5_area"] = ec.FastAPM(area["E1"], L), ec.FastAPM(area["A5"], L)
    probs = {**P, **pa, "E1_area": area["E1"], "A5_area": area["A5"]}

    def metrics(k: str) -> dict:
        a = fapm[k](m3)
        return {"mAP": float(np.nanmean(a[0])), "frontal": float(np.nanmean(a[1])), "frontal_unseen": float(np.nanmean(a[2])),
                "tail": float(np.nanmean(a[0][tail])), "tail_frontal": float(np.nanmean(a[1][tail])),
                "mECE": ec.ece_l1(probs[k], L), "n_classes_fu": int((~np.isnan(a[2])).sum())}

    M = {k: metrics(k) for k in fapm}

    # ---- checks against existing numbers ------------------------------------------------------------------
    chk = {}
    old = json.loads((O / "ens_select_w01_flip.json").read_text())
    for k, v in SINGLES.items():
        name = Path(v[0]).name
        if name in old["single"]:
            chk[f"{k} vs ens_select_w01_flip.json"] = abs(M[k]["mAP"] - old["single"][name])
    chk["E5 vs greedy-3 0.472238"] = abs(M["E5"]["mAP"] - old["ensemble_mAP"])
    for s in SEEDS:
        ev = json.loads((O / f"probs_unified_flip/eval_c_s{s}.json").read_text())["val_mAP"]
        chk[f"U{s} vs eval_c_s{s}.json"] = abs(M[f"U{s}"]["mAP"] - ev)
        ev = json.loads((O / f"probs_v0_flip/eval_v0_s{s}.json").read_text())["val_mAP"]
        chk[f"A{s} vs eval_v0_s{s}.json"] = abs(M[f"A{s}"]["mAP"] - ev)
    tmE1 = ea.tm_metrics(P["E1"], L)
    chk["E1 mAP vs torchmetrics"] = abs(M["E1"]["mAP"] - tmE1["mAP"])
    chk["E1 mECE vs torchmetrics"] = abs(M["E1"]["mECE"] - tmE1["mECE"])
    chk["E1 frontal vs ea.subset_map"] = abs(M["E1"]["frontal"] - ea.subset_map(P["E1"], L, ctx["masks"]["frontal"]))
    chk["E1 frontal_unseen vs ea.subset_map"] = abs(
        M["E1"]["frontal_unseen"] - ea.subset_map(P["E1"], L, ctx["masks"]["frontal_unseen"]))
    worst = max(chk.items(), key=lambda kv: kv[1])
    print(f"checks: {len(chk)} comparisons, worst |diff| = {worst[1]:.2e} ({worst[0]})", flush=True)
    assert worst[1] < 2e-5, worst

    # ---- ranking and the pre-registered rule -------------------------------------------------------------
    fr = {k: M[k]["frontal"] for k in cand}
    top, within, win = apply_rule(fr, size)
    rank = {}
    for met in ("mAP", "frontal", "frontal_unseen"):
        order = sorted(cand, key=lambda k: -M[k][met])
        for i, k in enumerate(order):
            rank.setdefault(k, {})[met] = i + 1
    R = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "boot": args.boot, "seed": args.seed,
         "checks_worst_abs_diff": worst[1], "metrics": M, "rank": rank, "size": size,
         "rule": {"top": top, "within_0.002": within, "winner": win, "tol": TOL},
         "singles": {k: {"dump": v[0], "run_dir": v[1], "weights": v[2], "scale": v[3], "code": v[4], "job": v[5]}
                     for k, v in SINGLES.items()},
         "ensembles": ENS}

    # ---- paired image bootstrap ---------------------------------------------------------------------------
    brun = cand + ["A5", "E1_area", "A5_area"]
    t1 = time.time()
    AP = ec.boot_replicates([fapm[k] for k in brun], m3, args.boot, args.seed, args.workers)
    print(f"bootstrap done ({time.time() - t1:.0f}s)", flush=True)
    bm = {"mAP": np.nanmean(AP[:, :, 0, :], -1), "frontal": np.nanmean(AP[:, :, 1, :], -1),
          "frontal_unseen": np.nanmean(AP[:, :, 2, :], -1), "tail": np.nanmean(AP[:, :, 0, :][..., tail], -1)}
    ix = {k: i for i, k in enumerate(brun)}

    def diff(x: str, y: str) -> dict:
        out = {}
        for met in ("mAP", "frontal", "frontal_unseen", "tail"):
            d = bm[met][:, ix[x]] - bm[met][:, ix[y]]
            bs = ec.boot_summary(d)
            out[met] = {"point": M[x][met] - M[y][met], "ci": bs["ci"], "p": bs["p"], "B": bs["B"]}
        return out

    R["E1_vs_A5_lanczos"] = diff("E1", "A5")
    R["E1_vs_A5_area"] = diff("E1_area", "A5_area")
    # mECE difference with a smaller bootstrap (a weighted 15-bin ECE per resample)
    for tag, (x, y) in {"lanczos": ("E1", "A5"), "area": ("E1_area", "A5_area")}.items():
        dd = []
        for b in range(args.boot_ece):
            rng = np.random.default_rng([args.seed, b])
            w = np.bincount(rng.integers(0, N, N), minlength=N).astype(np.float64)
            dd.append(ec.ece_l1(probs[x], L, w) - ec.ece_l1(probs[y], L, w))
        bs = ec.boot_summary(np.array(dd))
        R[f"E1_vs_A5_{tag}"]["mECE"] = {"point": M[x]["mECE"] - M[y]["mECE"], "ci": bs["ci"], "p": bs["p"], "B": bs["B"]}

    # stability of the ranking by frontal mAP
    F = bm["frontal"][:, [ix[k] for k in cand]]              # (B, 21)
    first = np.bincount(np.nanargmax(F, 1), minlength=len(cand)) / len(F)
    picks = []
    for b in range(F.shape[0]):
        _, _, w_ = apply_rule(dict(zip(cand, F[b])), size)
        picks.append(w_)
    pick_share = {k: float(np.mean([p == k for p in picks])) for k in cand}
    R["stability"] = {
        "p_first_by_frontal": dict(zip(cand, map(float, first))),
        "p_rule_pick": pick_share,
        "vs_E1": {k: {"frontal_diff": M[k]["frontal"] - M["E1"]["frontal"],
                      "ci": ec.boot_summary(bm["frontal"][:, ix[k]] - bm["frontal"][:, ix["E1"]])["ci"],
                      "mAP_diff": M[k]["mAP"] - M["E1"]["mAP"],
                      "mAP_ci": ec.boot_summary(bm["mAP"][:, ix[k]] - bm["mAP"][:, ix["E1"]])["ci"]}
                  for k in cand if k != "E1"},
    }

    # ---- selection optimism: patient-level halves ---------------------------------------------------------
    t1 = time.time()
    codes, P_pat = ctx["pid_codes"], ctx["n_patients"]
    frm = ctx["masks"]["frontal"]
    rows = []
    for r in range(args.splits):
        rng = np.random.default_rng([args.seed + 7, r])
        h1 = (rng.random(P_pat) < 0.5)[codes]
        W2 = np.stack([frm & h1, frm & ~h1], 1).astype(np.float64)
        sc = {k: np.nanmean(fapm[k](W2), 1) for k in cand}   # k -> (score half 1, score half 2)
        s1 = {k: float(v[0]) for k, v in sc.items()}
        s2 = {k: float(v[1]) for k, v in sc.items()}
        _, _, pick = apply_rule(s1, size)
        m1, m2 = np.mean(list(s1.values())), np.mean(list(s2.values()))
        order2 = sorted(cand, key=lambda k: -s2[k])
        rows.append({"pick": pick, "adv_sel": s1[pick] - m1, "adv_held": s2[pick] - m2,
                     "rank_held": order2.index(pick) + 1, "best_held_adv": max(s2.values()) - m2,
                     "spearman": float(stats.spearmanr([s1[k] for k in cand], [s2[k] for k in cand])[0]),
                     "same_as_full": pick == win, "n1": int((frm & h1).sum()), "n2": int((frm & ~h1).sum())})
    cv = {k: [r[k] for r in rows] for k in ("adv_sel", "adv_held", "rank_held", "best_held_adv", "spearman", "same_as_full")}
    R["selection_cv"] = {
        "splits": args.splits,
        "adv_selection_half_mean": float(np.mean(cv["adv_sel"])),
        "adv_heldout_half_mean": float(np.mean(cv["adv_held"])),
        "adv_heldout_half_sd": float(np.std(cv["adv_held"], ddof=1)),
        "optimism": float(np.mean(cv["adv_sel"]) - np.mean(cv["adv_held"])),
        "rank_heldout_mean": float(np.mean(cv["rank_held"])), "rank_heldout_median": float(np.median(cv["rank_held"])),
        "best_possible_heldout_adv_mean": float(np.mean(cv["best_held_adv"])),
        "spearman_between_halves_mean": float(np.mean(cv["spearman"])),
        "spearman_between_halves_sd": float(np.std(cv["spearman"], ddof=1)),
        "pick_same_as_full_data_share": float(np.mean(cv["same_as_full"])),
        "pick_counts": {k: int(sum(r["pick"] == k for r in rows)) for k in sorted({r["pick"] for r in rows})},
    }
    print(f"selection CV done ({time.time() - t1:.0f}s)", flush=True)
    args.out.write_text(json.dumps(R, indent=1, default=float))

    # ---- print --------------------------------------------------------------------------------------------
    f4 = lambda v: f"{v:.4f}"  # noqa: E731
    print("\n### ranking by frontal mAP (lanczos, flip TTA); rank in brackets: all / frontal / frontal&unseen")
    print("| rank (fr) | id | members | scales | all mAP [rank] | frontal mAP | fr&unseen mAP [rank] | tail AP | mECE |")
    print("|---|---|---|---|---|---|---|---|---|")
    scales = {k: [SINGLES[k][3]] for k in SINGLES}
    scales.update({k: [SINGLES[m][3] for m in v] for k, v in ENS.items()})
    for k in sorted(cand, key=lambda k: -fr[k]):
        m = M[k]
        sc_txt = "/".join(str(x) for x in scales[k])
        print(f"| {rank[k]['frontal']} | {k} | {size[k]} | {sc_txt} | {f4(m['mAP'])} [{rank[k]['mAP']}] | {f4(m['frontal'])} | "
              f"{f4(m['frontal_unseen'])} [{rank[k]['frontal_unseen']}] | {f4(m['tail'])} | {f4(m['mECE'])} |")
    print("\ncontrols (not eligible):")
    for k in list(ASL) + ["A5", "E1_area", "A5_area"]:
        m = M[k]
        print(f"  {k:12s} all {f4(m['mAP'])} frontal {f4(m['frontal'])} fr&unseen {f4(m['frontal_unseen'])} tail {f4(m['tail'])} mECE {f4(m['mECE'])}")
    print(f"\nrule: top = {top} ({fr[top]:.4f}); within 0.002 = {within}; winner = {win}")
    for tag in ("E1_vs_A5_lanczos", "E1_vs_A5_area"):
        print(f"\n{tag}:")
        for met, v in R[tag].items():
            print(f"  {met:15s} {v['point']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] p_boot {v['p']:.3f}")
    print("\nstability (share of resamples first by frontal mAP | share where the rule picks it | frontal diff vs E1 [CI]):")
    for k in sorted(cand, key=lambda k: -fr[k]):
        v = R["stability"]["vs_E1"].get(k)
        print(f"  {k:10s} first {R['stability']['p_first_by_frontal'][k]:.3f} rule {R['stability']['p_rule_pick'][k]:.3f}"
              + (f" | vs E1 {v['frontal_diff']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]" if v else " | (reference)"))
    print("\nselection CV:", json.dumps(R["selection_cv"], indent=1))
    print(f"\nwrote {args.out} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
