"""A vs B comparison on the internal val fold (docs/plan/2026-09-30-markov-ab-training-plan.md).

Arms (all 768 px, same Stage-1 checkpoint and push-wave recipe; SWA weights, flip TTA):
  v3        baseline, jobs 49000 (seed 42) / 48999 (seed 1024)
  A         v4 label-graph Markov layer: mk1, mk2, mk1fix (seed 42) + A-strong mk1 lr x100
  B-late    v3 probabilities + patient-chain HMM fusion (w = 1, and CV-chosen w)
  B-early   v5 patient-prior label-query conditioning (seeds 42, 1024), fed each image's prior
  B-early0  the same v5 models with every prior zeroed (= no history, the leaderboard condition)
B-late and B-early use the identical prior vector (analysis/patient_prior.py -> priors.pt).

Decision rule (fixed in the plan before any result):
  primary metric   macro mAP on has-prior & frontal val images, paired patient-clustered
                   bootstrap delta vs the matched-seed v3 baseline (300 resamples, 95% CI)
  internal winner  largest primary delta whose CI excludes 0 and whose sign replicates at seed
                   1024 (A arms have seed 42 only -> flagged)
  guard 1          B-early must not be worse than v3 on no-prior images (CI not entirely < 0)
  guard 2          B-early must not be worse than v3 on new-finding stratum AP (copy check)
  leaderboard      only arms usable without patient history (A, A-strong, B-early0) vs v3 on
                   all val and on frontal val

    srun -c 4 --mem=32G -p general --time=1:00:00 python analysis/markov_ab_compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchmetrics.functional.classification import binary_average_precision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import markov_check as mc  # noqa: E402
import patient_markov_check as pmc  # noqa: E402
import patient_prior as pp  # noqa: E402
from class_groups import frequency_groups, load_train_prevalence  # noqa: E402

DUMP = ROOT / "analysis/out/probs_ab_flip"
OUT = ROOT / "analysis/out/markov_ab"
W_GRID = pmc.W_GRID
MIN_STRATUM = 5  # >= 5 positives and >= 5 negatives for a class to count in a stratum

ARMS = {  # arm -> (dump file, seed)
    "v3": {42: "push__img768_dp01_seed42__model_swa@768_flip.npy",
           1024: "push__img768_dp01_seed1024__model_swa@768_flip.npy"},
    "A mk1": {42: "markov__img768_mk1_seed42__model_swa@768_flip.npy"},
    "A mk2": {42: "markov__img768_mk2_seed42__model_swa@768_flip.npy"},
    "A mk1fix": {42: "markov__img768_mk1fix_seed42__model_swa@768_flip.npy"},
    "A-strong mk1 lr100": {42: "markov__img768_mk1_lr100_seed42__model_swa@768_flip.npy"},
    "B-early": {42: "prior__img768_qprior_seed42__model_swa@768_flip_prior-with.npy",
                1024: "prior__img768_qprior_seed1024__model_swa@768_flip_prior-with.npy"},
    "B-early0": {42: "prior__img768_qprior_seed42__model_swa@768_flip_prior-zero.npy",
                 1024: "prior__img768_qprior_seed1024__model_swa@768_flip_prior-zero.npy"},
}


def stratum_map(P: np.ndarray, L: np.ndarray, prior: np.ndarray, rows: np.ndarray,
                positive: bool, classes: np.ndarray) -> float:
    """Macro AP, per class, within the rows whose prior state for that class is `positive`
    (prior-positive: persistent vs resolved; prior-negative: new vs absent)."""
    aps = []
    for c in classes:
        rr = rows[(prior[rows, c] == 1) == positive]
        y = L[rr, c]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        aps.append(float(binary_average_precision(torch.from_numpy(np.ascontiguousarray(P[rr, c])),
                                                  torch.from_numpy(y.astype(np.int64)))))
    return float(np.mean(aps)) if aps else float("nan")


def eligible(L: np.ndarray, prior: np.ndarray, rows: np.ndarray, positive: bool) -> np.ndarray:
    out = []
    for c in range(L.shape[1]):
        y = L[rows[(prior[rows, c] == 1) == positive], c]
        if y.sum() >= MIN_STRATUM and (len(y) - y.sum()) >= MIN_STRATUM:
            out.append(c)
    return np.array(out, int)


def cv_fuse(P: np.ndarray, L: np.ndarray, log_ratio: np.ndarray, rows: np.ndarray,
            cv_fold: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """B-late with a global w chosen by 2-fold patient-grouped CV over `rows` (as in pmc)."""
    Q = P.copy()
    chosen = []
    for f in (True, False):
        fit, ev = rows[cv_fold[rows] == f], rows[cv_fold[rows] != f]
        w_best = max(W_GRID, key=lambda w: pmc.subset_map(pmc.fuse(P, log_ratio, w), L, fit)[0])
        chosen.append(float(w_best))
        Q[ev] = pmc.fuse(P[ev], log_ratio[ev], w_best)
    return Q, chosen


def main() -> None:
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"dump_dir": str(DUMP), "n_boot": pmc.N_BOOT, "ci": "95% percentile"}

    classes = list(torch.load(pp.LABEL_INFO, weights_only=False)["class_names"])
    _, prev, _ = load_train_prevalence(classes)
    grp = frequency_groups(prev)
    group_name = np.where(grp["head"], "head", np.where(grp["tail"], "tail", "medium"))
    tr, va_full = pp.load_folds(classes)
    sg, _ = pp.study_table(tr, va_full)

    # ---- row alignment: keys.txt (written by evaluate_tta_v5) is the ground truth
    keys = (DUMP / "keys.txt").read_text().split("\n")
    L = np.load(DUMP / "labels.npy").astype(np.uint8)
    pos = {f: i for i, f in enumerate(va_full["filename"])}
    va = va_full.iloc[[pos[k] for k in keys]].reset_index(drop=True)
    assert (pp.decode(va["code"].to_numpy()) == L).all(), "labels.npy != fold labels for keys.txt"
    order = np.load(mc.KEY_ORDER)
    same_order = va_full["filename"].iloc[order].tolist() == keys
    print(f"[0] rows {len(keys)} | keys.txt == reconstructed val_key_order.npy: {same_order}")
    res["keys_match_val_key_order"] = same_order
    old = np.load(mc.PROBS / ARMS["v3"][42])
    new = np.load(DUMP / ARMS["v3"][42])
    d_old = float(np.abs(old - new).max()) if same_order and old.shape == new.shape else None
    print(f"    v3 s42 SWA re-dump vs analysis/out/probs_w01_flip: max |d| {d_old}")
    res["v3_redump_max_abs_diff"] = d_old

    pri = torch.load(pp.PRIORS, weights_only=False)
    ridx = {n: i for i, n in enumerate(pri["val"]["fnames"])}
    r = pri["val"]["r"][[ridx[k] for k in keys]]
    log_ratio, has = r[:, :30].astype(np.float64), r[:, 30] == 1
    q = pp.lookup_prior(va["pid"].to_numpy(), va["date"].to_numpy(), pp.date_states(sg))
    prior, has_q = pp.prior_bits(q)
    assert (has_q == has).all()

    tr_sids, tr_pids = set(tr["sid"]), set(tr["pid"])
    frontal = va["proj"].isin(mc.FRONTAL).to_numpy()
    s_in = va["sid"].isin(tr_sids).to_numpy()
    p_unseen = ~va["pid"].isin(tr_pids).to_numpy()
    pid = va["pid"].to_numpy()
    subsets = {
        "all": np.ones(len(va), bool), "frontal": frontal,
        "has_prior": has, "has_prior & frontal (PRIMARY)": has & frontal,
        "has_prior & study_not_in_train": has & ~s_in, "no_prior": ~has,
        "has_prior & patient_unseen (n small)": has & p_unseen,
    }
    PRIMARY = "has_prior & frontal (PRIMARY)"
    print("    subsets: " + ", ".join(f"{k} {int(m.sum())}" for k, m in subsets.items()))

    # ---- load arms; B-late from the v3 baseline of the same seed
    rng = np.random.default_rng(0)
    upid = np.unique(pid[has])
    half = set(rng.permutation(upid)[: len(upid) // 2])
    cv_fold = np.array([p_ in half for p_ in pid])
    probs: dict[tuple[str, int], np.ndarray] = {}
    for arm, per_seed in ARMS.items():
        for seed, f in per_seed.items():
            if (DUMP / f).exists():
                probs[(arm, seed)] = np.load(DUMP / f)
            else:
                print(f"    MISSING dump for {arm} seed {seed}: {f}")
    for seed in (42, 1024):
        if ("v3", seed) in probs:
            P = probs[("v3", seed)]
            probs[("B-late w1", seed)] = pmc.fuse(P, log_ratio, 1.0)
            probs[("B-late CV-w", seed)], chosen = cv_fuse(P, L, log_ratio, np.where(has)[0], cv_fold)
            res[f"B-late CV-w chosen s{seed}"] = chosen
            print(f"    B-late CV-w seed {seed}: w chosen {chosen}")

    # ---- [1] mAP of every arm on every subset, and paired delta vs the matched-seed baseline
    print(f"\n[1] macro mAP by subset; d = arm - v3 (same seed), paired patient bootstrap {pmc.N_BOOT}x")
    res["subsets"] = {}
    for (arm, seed), P in sorted(probs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        base = probs.get(("v3", seed))
        line = [f"    {arm:20s} s{seed:<5d}"]
        for name, msk in subsets.items():
            rows = np.where(msk)[0]
            m, nc = pmc.subset_map(P, L, rows)
            entry = {"n": len(rows), "classes": nc, "mAP": m}
            if base is not None and arm != "v3":
                d = pmc.bootstrap(pmc.by_patient(pid, rows),
                                  lambda rr, P=P, base=base: pmc.subset_map(P, L, rr)[0] - pmc.subset_map(base, L, rr)[0],
                                  seed=3)
                entry["d_vs_v3"] = d
                entry["d_point"] = m - pmc.subset_map(base, L, rows)[0]
            res["subsets"].setdefault(f"{arm} s{seed}", {})[name] = entry
        print(line[0])
        for name, e in res["subsets"][f"{arm} s{seed}"].items():
            dd = f" | d {e['d_point']:+.4f} (boot {pmc.fmt_ci(e['d_vs_v3'])})" if "d_vs_v3" in e else ""
            print(f"        {name:38s} n {e['n']:5d} cls {e['classes']:2d} mAP {e['mAP']:.4f}{dd}")

    # ---- [2] B-early vs B-late, paired, on the primary subset (and has_prior)
    print("\n[2] B-early vs B-late (paired patient bootstrap)")
    res["b_early_vs_late"] = {}
    for seed in (42, 1024):
        for late in ("B-late w1", "B-late CV-w"):
            if ("B-early", seed) not in probs or (late, seed) not in probs:
                continue
            for name in ("has_prior", PRIMARY):
                rows = np.where(subsets[name])[0]
                Pe, Pl = probs[("B-early", seed)], probs[(late, seed)]
                d = pmc.bootstrap(pmc.by_patient(pid, rows),
                                  lambda rr: pmc.subset_map(Pe, L, rr)[0] - pmc.subset_map(Pl, L, rr)[0], seed=4)
                pt = pmc.subset_map(Pe, L, rows)[0] - pmc.subset_map(Pl, L, rows)[0]
                print(f"    s{seed} B-early - {late:12s} on {name:32s}: {pt:+.4f} (boot {pmc.fmt_ci(d)})")
                res["b_early_vs_late"][f"s{seed} {late} {name}"] = {"d_point": pt, "d": d}

    # ---- [3] changed-finding strata (copy-risk test), has-prior rows
    print(f"\n[3] stratified AP on has-prior rows (classes with >= {MIN_STRATUM} pos and neg per stratum)")
    res["strata"] = {}
    h_rows = np.where(has)[0]
    for positive, label in ((False, "prior-negative: new vs absent"), (True, "prior-positive: persistent vs resolved")):
        cls = eligible(L, prior, h_rows, positive)
        print(f"    {label} | classes {len(cls)}")
        for (arm, seed), P in sorted(probs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            m = stratum_map(P, L, prior, h_rows, positive, cls)
            entry = {"mAP": m, "classes": len(cls)}
            base = probs.get(("v3", seed))
            if base is not None and arm != "v3":
                entry["d_point"] = m - stratum_map(base, L, prior, h_rows, positive, cls)
                entry["d_vs_v3"] = pmc.bootstrap(
                    pmc.by_patient(pid, h_rows),
                    lambda rr, P=P, base=base: (stratum_map(P, L, prior, rr, positive, cls)
                                                - stratum_map(base, L, prior, rr, positive, cls)), seed=5)
            res["strata"].setdefault(label, {})[f"{arm} s{seed}"] = entry
            dd = f" | d {entry['d_point']:+.4f} (boot {pmc.fmt_ci(entry['d_vs_v3'])})" if "d_vs_v3" in entry else ""
            print(f"        {arm:20s} s{seed:<5d} {m:.4f}{dd}")

    # ---- [4] head / medium / tail per-class dAP on the primary subset
    print(f"\n[4] per-class dAP vs v3 on {PRIMARY}")
    rows = np.where(subsets[PRIMARY])[0]
    res["groups"] = {}
    for (arm, seed), P in sorted(probs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        base = probs.get(("v3", seed))
        if base is None or arm == "v3":
            continue
        dap = pmc.per_class_ap(P, L, rows) - pmc.per_class_ap(base, L, rows)
        parts = []
        for g in ("head", "medium", "tail"):
            gm = (group_name == g) & ~np.isnan(dap)
            parts.append(f"{g} {np.nanmean(dap[gm]):+.4f} ({int((dap[gm] > 0).sum())}/{int(gm.sum())} up)")
            res["groups"].setdefault(f"{arm} s{seed}", {})[g] = {
                "mean": float(np.nanmean(dap[gm])), "improved": int((dap[gm] > 0).sum()), "classes": int(gm.sum())}
        print(f"    {arm:20s} s{seed:<5d} " + " | ".join(parts))

    # ---- [5] decision rule
    print("\n[5] decision (rule fixed in docs/plan/2026-09-30-markov-ab-training-plan.md)")
    verdict = {}

    def prim(arm, seed, name=PRIMARY):
        return res["subsets"].get(f"{arm} s{seed}", {}).get(name)

    cands = []
    for arm in ("A mk1", "A mk2", "A mk1fix", "A-strong mk1 lr100", "B-late w1", "B-late CV-w", "B-early"):
        e42, e1024 = prim(arm, 42), prim(arm, 1024)
        if e42 is None:
            continue
        ci42 = e42["d_vs_v3"]["ci95"]
        excl = ci42[0] > 0 or ci42[1] < 0
        rep = None if e1024 is None else (np.sign(e1024["d_point"]) == np.sign(e42["d_point"]))
        ok = excl and e42["d_point"] > 0 and (rep is not False)
        cands.append((arm, e42["d_point"], ci42, rep, ok))
        s1024 = "n/a (one seed)" if e1024 is None else f"{e1024['d_point']:+.4f}"
        flag = " (single seed - flagged)" if ok and e1024 is None else ""
        print(f"    {arm:20s} primary d s42 {e42['d_point']:+.4f} CI [{ci42[0]:+.4f}, {ci42[1]:+.4f}]"
              f" | s1024 {s1024} | qualifies: {ok}{flag}")
    winners = [c for c in cands if c[4]]
    win = max(winners, key=lambda c: c[1])[0] if winners else None
    verdict["internal_winner"] = win
    for seed in (42, 1024):
        g1 = prim("B-early", seed, "no_prior")
        g2 = res["strata"].get("prior-negative: new vs absent", {}).get(f"B-early s{seed}")
        if g1 is None or g2 is None:
            continue
        guard1 = not (g1["d_vs_v3"]["ci95"][1] < 0)
        guard2 = not (g2["d_vs_v3"]["ci95"][1] < 0)
        verdict[f"B-early s{seed} guard1_no_prior_ok"] = guard1
        verdict[f"B-early s{seed} guard2_new_finding_ok"] = guard2
        print(f"    B-early s{seed}: guard 1 (no-prior not worse) {guard1} | guard 2 (new findings not worse) {guard2}")
    print(f"    INTERNAL WINNER (longitudinal simulation): {win}")
    print("    leaderboard-usable arms (no patient history needed), d vs v3 on all / frontal val:")
    for arm in ("A mk1", "A mk2", "A mk1fix", "A-strong mk1 lr100", "B-early0"):
        for seed in (42, 1024):
            ea, ef = prim(arm, seed, "all"), prim(arm, seed, "frontal")
            if ea is None:
                continue
            print(f"        {arm:20s} s{seed:<5d} all {ea['d_point']:+.4f} {pmc.fmt_ci(ea['d_vs_v3'])} | "
                  f"frontal {ef['d_point']:+.4f} {pmc.fmt_ci(ef['d_vs_v3'])}")
    res["verdict"] = verdict

    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
