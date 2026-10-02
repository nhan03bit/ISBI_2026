"""Patient-history Markov chain on the CXR-LT 2026 Task 1 internal folds (2026-09-30).

Clinical-deployment SIMULATION on internal train/val data only. The leaderboard test set has no
usable patient history (patients de-duplicated against training, IDs withheld), and the PadChest
metadata CSV also carries the report-derived labels of the evaluation images. So:
  * never join dev/test/leaderboard images to the PadChest CSV;
  * read only ImageID / StudyID / PatientID / StudyDate_DICOM / Projection from it
    (never Labels, Report, PatientBirth or sex);
  * take labels only from the 30-class label_list of data/{train,val}_fold_1.csv.
The chain and prior lookup live in analysis/patient_prior.py (shared with train_2_v5.py).

1. Study-level leak: val images whose StudyID (and therefore report labels) is also in train.
2. Per-class 2-state patient Markov chain, fitted on consecutive train-fold studies:
   a_c = P(y_t=1 | y_{t-1}=1), b_c = P(y_t=1 | y_{t-1}=0), overall and by time gap.
3. History-only predictor on val images that have a strictly earlier study.
4. HMM late fusion with the cached greedy-3 ensemble (ConvNeXt = emission model):
   logit q = logit p + w * (logit pi_c(prior) - logit pi_c).

Report: docs/plan/2026-09-30-patient-markov-research.md (plan: docs/plan/2026-09-30-patient-markov-research-plan.md)

Usage
-----
    srun -c 4 --mem=16G -p general --time=0:45:00 python analysis/patient_markov_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluate"))
import markov_check as mc  # noqa: E402
import patient_prior as pp  # noqa: E402
from class_groups import frequency_groups, load_train_prevalence  # noqa: E402
from ens_select import macro_map  # noqa: E402
from torchmetrics.functional.classification import multilabel_average_precision  # noqa: E402

ROOT = mc.ROOT
OUT = ROOT / "analysis/out/patient_markov"
GAP_BINS = (("<=30d", 0, 30), ("31-365d", 31, 365), (">365d", 366, 10**6))
MIN_PAIRS = 50
N_BOOT = 300
W_GRID = np.round(np.arange(0.0, 1.51, 0.25), 2)
decode, logit = pp.decode, pp.logit


def subset_map(P: np.ndarray, L: np.ndarray, rows: np.ndarray) -> tuple[float, int]:
    """Macro mAP over the classes with >= 1 positive among `rows` (as in markov_check)."""
    y = L[rows]
    keep = y.sum(0) > 0
    return macro_map(np.ascontiguousarray(P[rows][:, keep]), torch.from_numpy(y[:, keep]).long()), int(keep.sum())


def per_class_ap(P: np.ndarray, L: np.ndarray, rows: np.ndarray) -> np.ndarray:
    y = torch.from_numpy(L[rows]).long()
    ap = multilabel_average_precision(torch.from_numpy(np.ascontiguousarray(P[rows])), y,
                                      num_labels=L.shape[1], average="none").numpy()
    ap[(L[rows].sum(0) == 0)] = np.nan
    return ap


def by_patient(pid: np.ndarray, rows: np.ndarray) -> list[np.ndarray]:
    s = pd.Series(rows, index=pid[rows])
    return [g.to_numpy() for _, g in s.groupby(level=0)]


def bootstrap(groups: list[np.ndarray], fn, seed: int) -> dict:
    """Patient-clustered bootstrap: resample patients with replacement, keep all their rows."""
    rng = np.random.default_rng(seed)
    k = len(groups)
    vals = np.array([fn(np.concatenate([groups[j] for j in rng.integers(0, k, k)]))
                     for _ in range(N_BOOT)])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"mean": float(vals.mean()), "ci95": [float(lo), float(hi)]}


def fuse(p: np.ndarray, log_ratio: np.ndarray, w: float) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-(logit(p) + w * log_ratio)))).astype(np.float32)


def fmt_ci(d: dict) -> str:
    return f"{d['mean']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"


def main() -> None:
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"padchest_csv": str(pp.PADCHEST), "n_boot": N_BOOT, "ci": "95% percentile"}

    classes = list(torch.load(mc.LABEL_INFO, weights_only=False)["class_names"])  # shard order
    _, prev, n_train_csv = load_train_prevalence(classes)
    groups = frequency_groups(prev)
    group_name = np.where(groups["head"], "head", np.where(groups["tail"], "tail", "medium"))

    tr, va_full = pp.load_folds(classes)

    # ------------------------------------------------------------------ 1. study-level leak
    tr_sids, tr_pids = set(tr["sid"]), set(tr["pid"])
    leak_full = va_full["sid"].isin(tr_sids).to_numpy()
    tr_code = tr.groupby("sid")["code"].agg(["first", "nunique"])
    same = (va_full.loc[leak_full, "code"].to_numpy() == tr_code["first"].reindex(va_full.loc[leak_full, "sid"]).to_numpy())
    print(f"[1] study-level leak | val images (CSV rows) {len(va_full)} | StudyID also in train "
          f"{int(leak_full.sum())} ({leak_full.mean():.3f}) | identical label_list {same.mean():.4f} | "
          f"train studies with >1 distinct label_list {int((tr_code['nunique'] > 1).sum())}")
    res["leak"] = {"n_val_csv": len(va_full), "n_study_in_train": int(leak_full.sum()),
                   "frac_study_in_train": float(leak_full.mean()), "frac_identical_labels": float(same.mean())}

    order = np.load(mc.KEY_ORDER)
    va = va_full.iloc[order].reset_index(drop=True)
    L = np.load(mc.PROBS / "labels.npy").astype(np.uint8)
    assert (decode(va["code"].to_numpy()) == L).all(), "rebuilt val row order does not reproduce labels.npy"
    P = np.mean([np.load(mc.PROBS / n) for n in mc.ENSEMBLE], 0).astype(np.float32)
    pid = va["pid"].to_numpy()
    allrows = np.arange(len(va))
    base_all, _ = subset_map(P, L, allrows)
    assert round(base_all, 4) == 0.4722, f"ensemble baseline {base_all:.4f} != 0.4722"

    frontal = va["proj"].isin(mc.FRONTAL).to_numpy()
    s_in = va["sid"].isin(tr_sids).to_numpy()
    p_unseen = ~va["pid"].isin(tr_pids).to_numpy()
    masks = {"all": np.ones(len(va), bool), "study_in_train": s_in, "study_not_in_train": ~s_in,
             "frontal & study_in_train": frontal & s_in, "frontal & study_not_in_train": frontal & ~s_in,
             "patient_seen & study_not_in_train": ~p_unseen & ~s_in, "patient_unseen": p_unseen}
    print(f"    greedy-3 ensemble (flip) macro mAP by subset:")
    res["leak"]["subsets"] = {}
    for name, msk in masks.items():
        m_, nc = subset_map(P, L, np.where(msk)[0])
        print(f"      {name:36s} n {int(msk.sum()):6d} | classes {nc:2d} | mAP {m_:.4f}")
        res["leak"]["subsets"][name] = {"n": int(msk.sum()), "classes": nc, "mAP": m_}

    def diff_fn(a_mask, b_mask):
        def fn(rows):
            a, b = rows[a_mask[rows]], rows[b_mask[rows]]
            return subset_map(P, L, a)[0] - subset_map(P, L, b)[0]
        return fn

    for a, b in (("study_in_train", "study_not_in_train"),
                 ("frontal & study_in_train", "frontal & study_not_in_train")):
        d = bootstrap(by_patient(pid, np.where(masks[a] | masks[b])[0]), diff_fn(masks[a], masks[b]), seed=1)
        print(f"    bootstrap mAP({a}) - mAP({b}): {fmt_ci(d)}")
        res["leak"][f"diff {a} - {b}"] = d

    # ------------------------------------------------------------------ 2. patient Markov chain
    sg, stats = pp.study_table(tr, va_full)
    print(f"\n[2] patient Markov chain | studies {len(sg)} (inconsistent labels within a study: "
          f"{stats['n_inconsistent']}; excluded, StudyID spans >1 patient or date: {stats['n_ambiguous']}) | "
          f"images without a date dropped {stats['n_undated']}")
    res["chain_n_ambiguous_studies"] = stats["n_ambiguous"]

    ds_tr = pp.date_states(sg[sg["in_tr"]])
    ch = pp.fit_chain(ds_tr)
    prev_c, next_c, gap, n1, a, b = ch["prev_c"], ch["next_c"], ch["gap"], ch["n1"], ch["a"], ch["b"]
    print(f"    consecutive train-fold (patient, date) pairs: {len(prev_c)} | patients contributing "
          f"{ch['n_patients']} | same-day multi-study dates "
          f"{int((ds_tr['n_studies'] > 1).sum())} | median gap {np.median(gap):.0f} d")
    print(f"    {'class':32s} {'group':6s} {'prev':>7s} {'n_prev+':>7s} {'a=P(1|1)':>9s} {'b=P(1|0)':>9s} {'a/prev':>7s}")
    chain = []
    for c in np.argsort(-prev):
        print(f"    {classes[c]:32s} {group_name[c]:6s} {prev[c]:7.4f} {int(n1[c]):7d} {a[c]:9.4f} {b[c]:9.4f} {a[c] / prev[c]:7.1f}")
        chain.append({"class": classes[c], "group": str(group_name[c]), "prevalence": float(prev[c]),
                      "n_prev_pos": int(n1[c]), "a": float(a[c]), "b": float(b[c])})
    for g in ("head", "medium", "tail"):
        gm = group_name == g
        print(f"    {g:6s} mean a {a[gm].mean():.3f} | mean b {b[gm].mean():.4f} | mean prevalence {prev[gm].mean():.4f}")
    res["chain"] = {"n_pairs": int(len(prev_c)), "median_gap_days": float(np.median(gap)), "per_class": chain}

    a_bin, b_bin = {}, {}
    print("    by gap (a shown where n_prev+ >= 50, else pooled):")
    for name, lo, hi in GAP_BINS:
        gm = (gap >= lo) & (gap <= hi)
        pc, nc_ = prev_c[gm], next_c[gm]
        k1 = pc.sum(0)
        k0 = len(pc) - k1
        ab = np.where(k1 >= MIN_PAIRS, ((pc & nc_).sum(0) + 1) / (k1 + 2), a)
        bb = np.where(k0 >= MIN_PAIRS, (((1 - pc) & nc_).sum(0) + 1) / (k0 + 2), b)
        a_bin[name], b_bin[name] = ab, bb
        ok = k1 >= MIN_PAIRS
        print(f"      {name:8s} pairs {int(gm.sum()):6d} | classes with own a {int(ok.sum()):2d} | "
              f"mean a over those {ab[ok].mean() if ok.any() else float('nan'):.3f} (pooled {a[ok].mean() if ok.any() else float('nan'):.3f})")
        res["chain"][f"gap {name}"] = {"n_pairs": int(gm.sum()), "classes_with_own_a": int(ok.sum()),
                                       "mean_a_own": float(ab[ok].mean()) if ok.any() else None}

    # ------------------------------------------------------------------ 3. history-only predictor
    q = pp.lookup_prior(pid, va["date"].to_numpy(), pp.date_states(sg))
    has = q["prior_date"].notna().to_numpy()
    if q.attrs["n_undated"]:
        print(f"    val images without a study date (no prior looked up): {q.attrs['n_undated']}")
    own_dates = sg["date"].reindex(va["sid"]).to_numpy()
    chk = has & ~pd.isna(own_dates)
    assert (own_dates[chk] == q.loc[chk, "date"].to_numpy()).all(), "image date differs from its study date"
    prior, _ = pp.prior_bits(q)
    prior_in_tr = has & q["any_tr"].fillna(False).to_numpy(bool)
    gap_v = np.full(len(va), np.nan)
    gap_v[has] = (q.loc[has, "date"] - q.loc[has, "prior_date"]).dt.days.to_numpy()
    print(f"\n[3] val images (aligned) with a strictly earlier study: {int(has.sum())} ({has.mean():.3f}) | "
          f"prior in train {int(prior_in_tr.sum())} | frontal {int((has & frontal).sum())} | "
          f"patient unseen in train {int((has & p_unseen).sum())} | median gap {np.nanmedian(gap_v):.0f} d")

    pi_hist = np.where(prior == 1, a, b)  # P(y_t = 1 | prior state), per image and class
    h_rows = np.where(has)[0]
    m_hist, nc = subset_map(pi_hist.astype(np.float32), L, h_rows)
    m_base, _ = subset_map(P, L, h_rows)
    print(f"    has-prior subset: history-only predictor mAP {m_hist:.4f} (tie-heavy: 2 distinct scores "
          f"per class) vs greedy-3 ensemble {m_base:.4f} | classes {nc}")
    res["history_only"] = {"n": int(has.sum()), "mAP_history_only": m_hist, "mAP_ensemble": m_base, "classes": nc}

    # ------------------------------------------------------------------ 4. HMM late fusion
    log_ratio = pp.prior_log_ratio(prior, has, a, b, prev)
    gb = np.select([gap_v <= 30, gap_v <= 365], [0, 1], 2)
    names = [n for n, _, _ in GAP_BINS]
    pi_gap = np.stack([np.where(prior == 1, a_bin[n], b_bin[n]) for n in names])[gb, allrows]
    log_ratio_gap = np.zeros(L.shape)
    log_ratio_gap[has] = logit(pi_gap[has]) - logit(np.broadcast_to(prev, (int(has.sum()), 30)))

    assert abs(subset_map(fuse(P, log_ratio, 0.0), L, allrows)[0] - base_all) < 1e-6, "w=0 fusion != baseline"
    fused1 = fuse(P, log_ratio, 1.0)
    fused1_gap = fuse(P, log_ratio_gap, 1.0)

    rng = np.random.default_rng(0)
    upid = np.unique(pid[has])
    half = set(rng.permutation(upid)[: len(upid) // 2])
    cv_fold = np.array([p_ in half for p_ in pid])

    subsets = {
        "has_prior": has, "has_prior & frontal": has & frontal,
        "has_prior & prior_in_train": prior_in_tr, "has_prior & patient_unseen": has & p_unseen,
        "has_prior & study_not_in_train": has & ~s_in,
        "has_prior & frontal & study_not_in_train": has & frontal & ~s_in,
    }
    print("\n[4] HMM late fusion, logit q = logit p + w (logit pi(prior) - logit pi) | "
          f"bootstrap {N_BOOT} patient-clustered resamples, 95% percentile CI")
    res["fusion"] = {}
    for name, msk in subsets.items():
        rows = np.where(msk)[0]
        mb, nc = subset_map(P, L, rows)
        m1, _ = subset_map(fused1, L, rows)
        mg, _ = subset_map(fused1_gap, L, rows)
        # 2-fold patient-grouped CV for a global w: pick w on one half, score the other
        cv_scores = np.empty_like(P)
        chosen = []
        for f in (True, False):
            fit, ev = rows[cv_fold[rows] == f], rows[cv_fold[rows] != f]
            w_best = max(W_GRID, key=lambda w: subset_map(fuse(P, log_ratio, w), L, fit)[0])
            chosen.append(float(w_best))
            cv_scores[ev] = fuse(P[ev], log_ratio[ev], w_best)
        mcv, _ = subset_map(cv_scores, L, rows)
        d = bootstrap(by_patient(pid, rows),
                      lambda r: subset_map(fused1, L, r)[0] - subset_map(P, L, r)[0], seed=2)
        print(f"    {name:42s} n {len(rows):5d} | cls {nc:2d} | base {mb:.4f} | w=1 {m1:.4f} "
              f"(d {fmt_ci(d)}) | w=1 gap-binned {mg:.4f} | CV-w {mcv:.4f} (w chosen {chosen})")
        res["fusion"][name] = {"n": len(rows), "classes": nc, "base": mb, "w1": m1, "w1_gap": mg,
                               "w1_minus_base": d, "cv": mcv, "cv_w": chosen}

    rows = np.where(has)[0]
    dap = per_class_ap(fused1, L, rows) - per_class_ap(P, L, rows)
    for g in ("head", "medium", "tail"):
        gm = (group_name == g) & ~np.isnan(dap)
        print(f"    has_prior per-class dAP (w=1), {g:6s}: mean {np.nanmean(dap[gm]):+.4f} over {int(gm.sum())} "
              f"classes | improved {int((dap[gm] > 0).sum())}")
        res["fusion"][f"dAP {g}"] = {"mean": float(np.nanmean(dap[gm])), "classes": int(gm.sum()),
                                     "improved": int((dap[gm] > 0).sum())}
    m_full, _ = subset_map(fused1, L, allrows)
    print(f"    full val (no-prior rows unchanged): base {base_all:.4f} -> w=1 {m_full:.4f} ({m_full - base_all:+.4f})")
    print("    leaderboard: 0 by construction (no patient history exists or may be used at test time)")
    res["fusion"]["full_val"] = {"base": base_all, "w1": m_full}

    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
