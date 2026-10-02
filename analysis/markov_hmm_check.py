"""Post-prediction patient HMM: fit, cross-fit and evaluate on the internal folds (2026-10-01).

Steps 4-5 of docs/plan/2026-10-01-markov-hmm-redesign-plan.md; model in analysis/markov_hmm.py,
metadata in analysis/patient_meta.py. CLINICAL SIMULATION ONLY: the leaderboard has no patient
history or metadata (and dev/test must never be joined to the PadChest CSV), so its effect there is 0.

Arms (emission = frozen classifier probabilities; H4 is the pre-specified primary arm):
  E      emission only
  H0     pooled 2-state chain (patient_prior.fit_chain), w_h = 1, no calibration  == B-late (asserted)
  H1     H0 + calibration + cross-fitted w_h
  H2     logistic transition with gap terms
  H3     + sex / age / acuity in transition and initial prior (w_0 also moves rows without history)
  H4     + cross-class temporal edges B                                          (primary)
  H5     H4 + within-state MRF edges J (ablation; w_J searched after H4's w_h, w_0)
  H4-acq H4 + per-state DICOM acquisition features in phi (ablation; care-setting / scanner proxies)
  H4-F   H4, filtered history (forward pass over earlier images; needs the train-fold probs dump)
  B-early / B-early0  v5 checkpoints (comparators, from analysis/out/probs_ab_flip when present)

Prior models are fitted on train-fold (patient, date) states only. Calibration and fusion weights are
cross-fitted on val: 2 patient-grouped halves (fit on one, score the other) for split seeds 0, 1, 2.
Weights maximise macro mAP over all fit-half rows on the grid {0, 0.25, ..., 1.5}. Ranking metrics are
computed within each half and averaged (pooling halves fitted with different transforms would mix scales).

    srun -c 4 --mem=48G -p general --time=3:00:00 .venv/bin/python analysis/markov_hmm_check.py
    ... --limit 3000          # dry run: 3000 val patients, 1 split seed, 20 bootstrap resamples
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluate"))
import markov_ab_compare as mab  # noqa: E402
import markov_check as mc  # noqa: E402
import markov_hmm as mh  # noqa: E402
import patient_markov_check as pmc  # noqa: E402
import patient_meta as pm  # noqa: E402
import patient_prior as pp  # noqa: E402
from class_groups import frequency_groups, load_train_prevalence  # noqa: E402

ROOT = mc.ROOT
OUT = ROOT / "analysis/out/markov_hmm"
TRAIN_DUMP = ROOT / "analysis/out/probs_train_flip"
AB_DUMP = mab.DUMP
W_GRID = pmc.W_GRID                       # 0, 0.25, ..., 1.5
WJ_GRID = np.array([0.0, 0.25, 0.5, 1.0])
SEEDS = (0, 1, 2)
L2_GRID = (1e-4, 1e-3, 1e-2)
L2_CROSS_GRID = (1e-3, 1e-2, 1e-1)
EMISSIONS = {
    "s42": ["push__img768_dp01_seed42__model_swa@768_flip.npy"],     # job 49000
    "s1024": ["push__img768_dp01_seed1024__model_swa@768_flip.npy"],  # job 48999
    "ens": mc.ENSEMBLE,                                               # greedy-3
}
FILTER_EMISSION = "s42"  # the train-fold dump exists for 49000 only
BLATE = ROOT / "analysis/out/patient_markov/results.json"


@dataclass
class Arm:
    name: str
    spec: mh.PriorSpec | None   # None = emission only
    cal: bool = True
    cv: bool = True             # cross-fit w_h (and w_0 if init)
    init: bool = False          # search w_0
    mrf: bool = False
    filtered: bool = False


def ece(P: np.ndarray, L: np.ndarray, bins: int = 15) -> float:
    p, y = P.reshape(-1).astype(np.float64), L.reshape(-1).astype(np.float64)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = np.bincount(idx, minlength=bins)
    conf = np.bincount(idx, p, bins)
    acc = np.bincount(idx, y, bins)
    ok = tot > 0
    return float(np.abs(conf[ok] - acc[ok]).sum() / len(p))


def patient_halves(pid: np.ndarray, seed: int) -> np.ndarray:
    codes, uniq = pd.factorize(pid)
    return np.random.default_rng(seed).integers(0, 2, len(uniq))[codes]


# ---------------------------------------------------------------------- train-CV prior diagnostics
def transition_cv(prev, nxt, g, phi, pid, specs: dict, table: dict | None) -> dict:
    """Held-out (patient-grouped 5-fold) log-loss / AUROC of nested transition models on train pairs."""
    from torchmetrics.functional.classification import multilabel_auroc
    codes, uniq = pd.factorize(pid)
    fold = np.random.default_rng(0).integers(0, 5, len(uniq))[codes]
    out = {}
    for name, spec in specs.items():
        Z = np.zeros(nxt.shape)
        for f in range(5):
            tr_, te_ = fold != f, fold == f
            if spec.kind == "table":
                n1 = prev[tr_].sum(0)
                n0 = tr_.sum() - n1
                a = ((prev[tr_] & nxt[tr_]).sum(0) + 1) / (n1 + 2)
                b = (((1 - prev[tr_]) & nxt[tr_]).sum(0) + 1) / (n0 + 2)
                Z[te_] = mh.logit(np.where(prev[te_] == 1, a, b))
            else:
                m = mh.PriorModel(spec).fit(prev[tr_], nxt[tr_], g[tr_], phi[tr_], nxt[tr_][:1], phi[tr_][:1])
                Z[te_] = m.predict_logit_observed(prev[te_], g[te_], phi[te_])
        y = nxt.astype(float)
        ll = -(y * np.log(mh.sigmoid(Z) + 1e-12) + (1 - y) * np.log(1 - mh.sigmoid(Z) + 1e-12)).mean(0)
        auc = multilabel_auroc(torch.from_numpy(Z).float(), torch.from_numpy(nxt.astype(np.int64)),
                               num_labels=nxt.shape[1], average="none").numpy()
        out[name] = {"logloss_macro": float(ll.mean()), "logloss_sum": float(ll.sum()),
                     "auroc_macro": float(np.nanmean(auc)), "logloss": ll.tolist(), "auroc": auc.tolist()}
    return out


def initial_cv(first, phi_first, pid) -> dict:
    codes, uniq = pd.factorize(pid)
    fold = np.random.default_rng(0).integers(0, 5, len(uniq))[codes]
    out = {}
    for name, use_meta in (("intercept", False), ("+meta", True)):
        Z = np.zeros(first.shape)
        for f in range(5):
            tr_, te_ = fold != f, fold == f
            F = np.concatenate([np.ones((len(first), 1))] + ([phi_first] if use_meta else []), 1)
            fit = mh.fit_logistic(F[tr_], first[tr_].astype(float), l2=np.r_[0.0, [1e-3] * (F.shape[1] - 1)])
            Z[te_] = mh.logistic_logits(F[te_], fit)
        y = first.astype(float)
        ll = -(y * np.log(mh.sigmoid(Z) + 1e-12) + (1 - y) * np.log(1 - mh.sigmoid(Z) + 1e-12)).mean(0)
        out[name] = {"logloss_macro": float(ll.mean())}
    return out


# ---------------------------------------------------------------------- main
def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--limit", type=int, default=0, help="dry run: keep N val patients, 1 seed, 20 resamples")
    ap_.add_argument("--no-filtered", action="store_true")
    ap_.add_argument("--out", default=str(OUT), help="output directory (results.json, model_H4_*.pt)")
    args = ap_.parse_args()
    out_dir = Path(args.out)
    torch.set_num_threads(4)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_boot = 20 if args.limit else pmc.N_BOOT
    pmc.N_BOOT = n_boot
    seeds = SEEDS[:1] if args.limit else SEEDS
    res: dict = {"n_boot": n_boot, "seeds": list(seeds), "w_grid": W_GRID.tolist(), "limit": args.limit}

    classes = list(torch.load(pp.LABEL_INFO, weights_only=False)["class_names"])
    _, prev, _ = load_train_prevalence(classes)
    lpi = mh.logit(prev)
    grp = frequency_groups(prev)
    group_name = np.where(grp["head"], "head", np.where(grp["tail"], "tail", "medium"))

    # ------------------------------------------------------------------ states and train-fold prior fit
    tr, va_full = pm.load_folds_meta(classes, acq=True)
    sg, stats = pp.study_table(tr, va_full)
    images = pd.concat([tr, va_full], ignore_index=True)
    st_all = pm.date_states_meta(sg, images)                 # history for val targets: train + val
    st_tr = pm.date_states_meta(sg[sg["in_tr"]], tr)          # fitting: train-fold states only
    feat = pm.Featurizer().fit(st_tr["sex"], st_tr["age"], st_tr["acuity"])
    phi_tr = feat(st_tr["sex"], st_tr["age"], st_tr["acuity"])
    Ytr = pp.decode(st_tr["code"].to_numpy())
    nxt_i = np.where(st_tr["prev"].to_numpy() >= 0)[0]
    prv_i = st_tr["prev"].to_numpy()[nxt_i]
    P_prev, P_next = Ytr[prv_i], Ytr[nxt_i]
    P_g = pm.gap_feature(st_tr["gap"].to_numpy()[nxt_i])
    P_phi, P_pid = phi_tr[nxt_i], st_tr["pid"].to_numpy()[nxt_i]
    first = st_tr["rank"].to_numpy() == 0
    chain = pp.fit_chain(pp.date_states(sg[sg["in_tr"]]))
    assert len(chain["prev_c"]) == len(P_prev) and (chain["prev_c"] == P_prev).all(), "pair table != fit_chain"
    # acquisition features per state (H4-acq only; care-setting / scanner proxies): mean over the day's images
    A_img, acq_names = pm.acquisition_features(images)
    img_si = pm.state_index(st_all, images["pid"].to_numpy(), images["date"].to_numpy())
    A_st, n_st = np.zeros((len(st_all), A_img.shape[1])), np.zeros(len(st_all))
    np.add.at(A_st, img_si[img_si >= 0], A_img[img_si >= 0])
    np.add.at(n_st, img_si[img_si >= 0], 1)
    A_st[n_st > 0] /= n_st[n_st > 0, None]
    A_tr = A_st[pm.state_index(st_all, st_tr["pid"].to_numpy(), st_tr["date"].to_numpy())]
    a_mu, a_sd = A_tr.mean(0), A_tr.std(0) + 1e-8
    phi_acq_tr = np.concatenate([phi_tr, (A_tr - a_mu) / a_sd], 1)
    print(f"[0] states: all {len(st_all)} | train {len(st_tr)} | train pairs {len(P_prev)} | train first states "
          f"{int(first.sum())} | excluded ambiguous studies {stats['n_ambiguous']} | phi = {pm.Featurizer.names}")

    # unpenalised intercept + beta logistic == unsmoothed chain counts
    own = mh.PriorModel(mh.PriorSpec(l2=0.0)).fit(P_prev, P_next, P_g, P_phi, Ytr[first], phi_tr[first])
    k11, n1 = (P_prev & P_next).sum(0), P_prev.sum(0)
    k01, n0 = ((1 - P_prev) & P_next).sum(0), len(P_prev) - P_prev.sum(0)
    ok = (k11 > 0) & (k11 < n1) & (k01 > 0) & (k01 < n0)
    one, zero = np.ones((1, 30)), np.zeros((1, 30))
    a_fit = mh.sigmoid(own.predict_logit_observed(one, np.zeros(1), P_phi[:1]))[0]
    b_fit = mh.sigmoid(own.predict_logit_observed(zero, np.zeros(1), P_phi[:1]))[0]
    err = max(np.abs(a_fit - k11 / n1)[ok].max(), np.abs(b_fit - k01 / n0)[ok].max())
    print(f"    unpenalised logistic vs unsmoothed counts: max |diff| {err:.2e} over {int(ok.sum())} classes")
    assert err < 1e-3

    # ---- L2 selection (H4 spec) and nested held-out diagnostics on train pairs
    print("\n[1] transition prior, patient-grouped 5-fold CV on train pairs (held-out)")
    sel = {}
    for l2 in L2_GRID:
        for l2c in L2_CROSS_GRID:
            sp = mh.PriorSpec(gap=True, meta=True, cross=True, l2=l2, l2_cross=l2c)
            r = transition_cv(P_prev, P_next, P_g, P_phi, P_pid, {"x": sp}, None)["x"]
            sel[(l2, l2c)] = r["logloss_sum"]
            print(f"    l2 {l2:g} l2_cross {l2c:g}: summed per-class log-loss {r['logloss_sum']:.5f}")
    l2, l2c = min(sel, key=sel.get)
    print(f"    chosen l2 {l2:g}, l2_cross {l2c:g}")
    res["l2"] = {"l2": l2, "l2_cross": l2c, "grid": {f"{k[0]:g},{k[1]:g}": v for k, v in sel.items()}}
    specs = {
        "table (pooled a, b)": mh.PriorSpec(kind="table"),
        "own (alpha, beta)": mh.PriorSpec(l2=l2),
        "+ gap": mh.PriorSpec(gap=True, l2=l2),
        "+ meta": mh.PriorSpec(gap=True, meta=True, l2=l2),
        "+ cross (H4 prior)": mh.PriorSpec(gap=True, meta=True, cross=True, l2=l2, l2_cross=l2c),
    }
    diag = transition_cv(P_prev, P_next, P_g, P_phi, P_pid, specs, chain)
    for k, v in diag.items():
        print(f"    {k:22s} macro log-loss {v['logloss_macro']:.5f} | macro AUROC {v['auroc_macro']:.4f}")
    res["transition_cv"] = {k: {kk: v[kk] for kk in ("logloss_macro", "logloss_sum", "auroc_macro")} for k, v in diag.items()}
    res["transition_cv_per_class"] = {k: {"logloss": v["logloss"], "auroc": v["auroc"]} for k, v in diag.items()}
    icv = initial_cv(Ytr[first], phi_tr[first], st_tr["pid"].to_numpy()[first])
    for k, v in icv.items():
        print(f"    initial prior {k:10s} macro log-loss {v['logloss_macro']:.5f}")
    res["initial_cv"] = icv
    dacq = transition_cv(P_prev, P_next, P_g, phi_acq_tr[nxt_i], P_pid, {"+ cross + acq (H4-acq prior)": specs["+ cross (H4 prior)"]}, None)
    for k, v in dacq.items():
        print(f"    {k:22s} macro log-loss {v['logloss_macro']:.5f} | macro AUROC {v['auroc_macro']:.4f}")
        res["transition_cv"][k] = {kk: v[kk] for kk in ("logloss_macro", "logloss_sum", "auroc_macro")}

    # fit the arms' priors on all train pairs
    priors = {}
    for key, sp in {"table": mh.PriorSpec(kind="table"), "gap": specs["+ gap"], "meta": specs["+ meta"],
                    "full": specs["+ cross (H4 prior)"]}.items():
        priors[key] = mh.PriorModel(sp).fit(P_prev, P_next, P_g, P_phi, Ytr[first], phi_tr[first], table=chain)
    priors["acq"] = mh.PriorModel(specs["+ cross (H4 prior)"]).fit(P_prev, P_next, P_g, phi_acq_tr[nxt_i],
                                                                   Ytr[first], phi_acq_tr[first])
    pw = mh.fit_pairwise(Ytr.astype(float), l2=1e-2)
    J = pw["J"]
    top = np.dstack(np.unravel_index(np.argsort(-np.abs(np.triu(J, 1)), axis=None)[:6], J.shape))[0]
    print("    within-state MRF J (pseudo-likelihood, train states), strongest couplings: "
          + ", ".join(f"{classes[i]}~{classes[j]} {J[i, j]:+.2f}" for i, j in top))
    tf = priors["full"].trans
    d0 = 1 + 1 + phi_tr.shape[1]
    B = tf["W"][d0:]
    tb = np.dstack(np.unravel_index(np.argsort(-np.abs(B), axis=None)[:8], B.shape))[0]
    print("    H4 cross-class temporal edges B[k -> c] (largest |B|): "
          + ", ".join(f"{classes[k]}->{classes[c]} {B[k, c]:+.2f}" for k, c in tb))
    res["top_B"] = [{"from": classes[k], "to": classes[c], "B": float(B[k, c])} for k, c in tb]
    res["top_J"] = [{"a": classes[i], "b": classes[j], "J": float(J[i, j])} for i, j in top]

    # ------------------------------------------------------------------ val targets
    order = np.load(mc.KEY_ORDER)
    va = va_full.iloc[order].reset_index(drop=True)
    L = np.load(mc.PROBS / "labels.npy").astype(np.uint8)
    assert (pp.decode(va["code"].to_numpy()) == L).all(), "rebuilt val row order does not reproduce labels.npy"
    pid, date = va["pid"].to_numpy(), va["date"].to_numpy()
    pidx = pm.prev_state_index(st_all, pid, date)
    has = pidx >= 0
    q = pp.lookup_prior(pid, date, pp.date_states(sg))
    prior_bits, has_pp = pp.prior_bits(q)
    assert (has == has_pp).all() and (pp.decode(st_all["code"].to_numpy()[pidx[has]]) == prior_bits[has]).all(), \
        "state-index prior != patient_prior.lookup_prior"
    sidx = pm.state_index(st_all, pid, date)
    t_sex = np.where(sidx >= 0, st_all["sex"].to_numpy()[np.maximum(sidx, 0)], va["sex"].to_numpy())
    t_age = np.where(sidx >= 0, st_all["age"].to_numpy()[np.maximum(sidx, 0)], va["age"].to_numpy())
    t_acu = np.where(sidx >= 0, st_all["acuity"].to_numpy()[np.maximum(sidx, 0)], va["acuity"].to_numpy())
    phi_t = feat(t_sex, t_age, t_acu)
    A_t = np.where((sidx >= 0)[:, None], A_st[np.maximum(sidx, 0)],
                   pm.acquisition_features(va)[0])
    phi_t_acq = np.concatenate([phi_t, (A_t - a_mu) / a_sd], 1)
    gap_t = np.full(len(va), np.nan)
    gap_t[has] = (pd.to_datetime(date[has]) - st_all["date"].to_numpy()[pidx[has]]) / np.timedelta64(1, "D")
    g_t = pm.gap_feature(gap_t)

    frontal = va["proj"].isin(mc.FRONTAL).to_numpy()
    s_in = va["sid"].isin(set(tr["sid"])).to_numpy()
    p_unseen = ~va["pid"].isin(set(tr["pid"])).to_numpy()
    # filtered-clean: no train-fold image anywhere in the target's history
    hist_tr = st_all["any_tr"].to_numpy().astype(bool).copy()
    prv_all, rk_all = st_all["prev"].to_numpy(), st_all["rank"].to_numpy()
    for r in range(1, int(rk_all.max()) + 1):
        ix = np.where(rk_all == r)[0]
        hist_tr[ix] |= hist_tr[prv_all[ix]]
    clean = has & ~hist_tr[np.maximum(pidx, 0)]
    subsets = {"full": np.ones(len(va), bool), "has_prior": has, "has_prior & frontal (PRIMARY)": has & frontal,
               "has_prior & study_not_in_train": has & ~s_in, "no_prior": ~has, "patient_unseen": p_unseen,
               "has_prior & history val-only": clean}
    PRIMARY = "has_prior & frontal (PRIMARY)"
    keep = np.ones(len(va), bool)
    if args.limit:
        upid = pd.unique(pid)
        keep = np.isin(pid, np.random.default_rng(0).permutation(upid)[: args.limit])
    print(f"\n[2] val targets {len(va)} | with an earlier state {int(has.sum())} ({has.mean():.3f}) | primary "
          f"{int((has & frontal).sum())} | history val-only {int(clean.sum())} | evaluated rows {int(keep.sum())}")
    res["subset_n"] = {k: int((m & keep).sum()) for k, m in subsets.items()}

    # observed-history prior terms
    def hist_term_observed(pr: mh.PriorModel, ph: np.ndarray) -> np.ndarray:
        H = np.zeros(L.shape)
        H[has] = pr.predict_logit_observed(prior_bits[has], g_t[has], ph[has]) - lpi
        return H

    phis = {k: (phi_t_acq if k == "acq" else phi_t) for k in priors}
    H_obs = {k: hist_term_observed(p, phis[k]) for k, p in priors.items()}
    I_meta = {k: p.initial_logit(phis[k]) - lpi for k, p in priors.items()}

    # ------------------------------------------------------------------ filtered-history ingredients
    filt = None
    if not args.no_filtered and (TRAIN_DUMP / "keys.txt").exists():
        tkeys = (TRAIN_DUMP / "keys.txt").read_text().split("\n")
        tfile = next(TRAIN_DUMP.glob("push__img768_dp01_seed42__model_swa@768_flip.npy"))
        tprobs = np.load(tfile)
        assert len(tkeys) == len(tprobs)
        img_si = pm.state_index(st_all, images["pid"].to_numpy(), images["date"].to_numpy())
        pos_t = {k: i for i, k in enumerate(tkeys)}
        pos_v = {f: i for i, f in enumerate(va["filename"])}
        src, row = [], []
        for f, in_tr in zip(images["filename"], np.r_[np.ones(len(tr), bool), np.zeros(len(va_full), bool)]):
            if in_tr and f in pos_t:
                src.append(0); row.append(pos_t[f])
            elif (not in_tr) and f in pos_v:
                src.append(1); row.append(pos_v[f])
            else:
                src.append(-1); row.append(-1)
        src, row = np.array(src), np.array(row)
        okimg = (src >= 0) & (img_si >= 0)
        print(f"    filtered mode: train-dump rows {len(tkeys)} | fold images with an emission and a state "
              f"{int(okimg.sum())}/{len(images)} (train {int(((src == 0) & okimg).sum())}, val {int(((src == 1) & okimg).sum())})")
        filt = {"src": src, "row": row, "ok": okimg, "si": img_si, "tprobs": tprobs}
        res["filtered_coverage"] = {"images_with_emission": int(okimg.sum()), "images": len(images)}
    elif not args.no_filtered:
        print(f"    filtered mode skipped: {TRAIN_DUMP / 'keys.txt'} not found (GPU dump pending)")

    st_g = pm.gap_feature(st_all["gap"].to_numpy())
    st_phi = feat(st_all["sex"], st_all["age"], st_all["acuity"])

    def hist_term_filtered(pr: mh.PriorModel, P_val: np.ndarray, cal: dict | None, w_e: float,
                           observed: bool = False) -> np.ndarray:
        """Forward pass over all states; ebar = mean calibrated LLR of each state's images."""
        S = len(st_all)
        lp_img = np.zeros((len(images), 30))
        o = filt["ok"]
        tsel, vsel = o & (filt["src"] == 0), o & (filt["src"] == 1)
        lp_img[tsel] = mh.logit(filt["tprobs"][filt["row"][tsel]])
        lp_img[vsel] = mh.logit(P_val[filt["row"][vsel]])
        llr = mh.calibrate(lp_img, cal) - lpi
        ebar, cnt = np.zeros((S, 30)), np.zeros(S)
        np.add.at(ebar, filt["si"][o], llr[o])
        np.add.at(cnt, filt["si"][o], 1)
        ebar[cnt > 0] /= cnt[cnt > 0, None]
        obs = pp.decode(st_all["code"].to_numpy()) if observed else None
        belief = mh.forward_filter(pr, prv_all, rk_all, st_g, st_phi, ebar, w_e, observed=obs)
        H = np.zeros(L.shape)
        H[has] = mh.logit(pr.predict(belief[pidx[has]], g_t[has], phi_t[has])) - lpi
        return H

    if filt is not None:  # one-hot beliefs reproduce observed mode
        Pv = np.mean([np.load(mc.PROBS / f) for f in EMISSIONS[FILTER_EMISSION]], 0)
        Hx = hist_term_filtered(priors["full"], Pv, None, 1.0, observed=True)
        assert np.allclose(Hx[has], H_obs["full"][has], atol=1e-9), "filtered(one-hot) != observed"
        print("    check: filtered mode with one-hot (observed) beliefs == observed mode")

    # ------------------------------------------------------------------ arms
    arms = [Arm("E", None, cal=False, cv=False),
            Arm("H0", mh.PriorSpec(kind="table"), cal=False, cv=False),
            Arm("H1", mh.PriorSpec(kind="table")),
            Arm("H2", specs["+ gap"]),
            Arm("H3", specs["+ meta"], init=True),
            Arm("H4", specs["+ cross (H4 prior)"], init=True),
            Arm("H5", specs["+ cross (H4 prior)"], init=True, mrf=True),
            Arm("H4-acq", specs["+ cross (H4 prior)"], init=True),
            Arm("H4-F", specs["+ cross (H4 prior)"], init=True, filtered=True)]
    prior_key = {"H0": "table", "H1": "table", "H2": "gap", "H3": "meta", "H4": "full", "H5": "full", "H4-F": "full",
                 "H4-acq": "acq"}

    def grid_search(lpc, Hh, I0, fit_rows, arm: Arm, w_fixed=None):
        best, best_w = -1.0, None
        hgrid = W_GRID if arm.cv else np.array([1.0])
        ogrid = W_GRID if arm.init else np.array([0.0])
        Lf = L[fit_rows]
        for w_h in hgrid:
            for w_0 in ogrid:
                z = lpc[fit_rows] + np.where(has[fit_rows, None], w_h * Hh[fit_rows], w_0 * I0[fit_rows])
                m_, _ = pmc.subset_map(mh.sigmoid(z).astype(np.float32), Lf, np.arange(len(Lf)))
                if m_ > best:
                    best, best_w = m_, (float(w_h), float(w_0))
        return best_w, best

    def run_arm(arm: Arm, P: np.ndarray, ename: str) -> dict:
        lp = mh.logit(P)
        if arm.spec is None:
            return {"Q": [P.astype(np.float32)] * len(seeds), "w": []}
        pr = priors[prior_key[arm.name]]
        if not arm.cv:  # H0: fixed Bayes w = 1, raw emission
            z = mh.fuse(lp, H_obs[prior_key[arm.name]], I_meta[prior_key[arm.name]], has, mh.FusionWeights(1.0, 0.0))
            return {"Q": [mh.sigmoid(z).astype(np.float32)] * len(seeds), "w": [[1.0, 0.0]]}
        Qs, ws = [], []
        for half in halves:
            Q = np.zeros(L.shape, np.float32)
            for h in (0, 1):
                fit_rows = np.where((half != h) & keep)[0]
                ev = half == h
                cal = mh.fit_calibration(lp[fit_rows], L[fit_rows]) if arm.cal else None
                lpc = mh.calibrate(lp, cal)
                I0 = I_meta[prior_key[arm.name]]
                if arm.filtered:
                    best = (-1.0, None, None)
                    for w_e in W_GRID[1:]:
                        Hh = hist_term_filtered(pr, P, cal, float(w_e))
                        w, m_ = grid_search(lpc, Hh, I0, fit_rows, arm)
                        if m_ > best[0]:
                            best = (m_, (*w, float(w_e)), Hh)
                    _, (w_h, w_0, w_e), Hh = best
                    wJ = 0.0
                else:
                    Hh = H_obs[prior_key[arm.name]]
                    (w_h, w_0), _ = grid_search(lpc, Hh, I0, fit_rows, arm)
                    w_e, wJ = 1.0, 0.0
                    if arm.mrf:
                        def score_j(wj):
                            z = mh.fuse(lpc[fit_rows], Hh[fit_rows], I0[fit_rows], has[fit_rows],
                                        mh.FusionWeights(w_h, w_0, 1.0, float(wj)), J)
                            return pmc.subset_map(mh.sigmoid(z).astype(np.float32), L[fit_rows], np.arange(len(fit_rows)))[0]
                        wJ = float(max(WJ_GRID, key=score_j))
                wts = mh.FusionWeights(w_h, w_0, w_e, wJ)
                z = mh.fuse(lpc[ev], Hh[ev], I0[ev], has[ev], wts, J if arm.mrf else None)
                Q[ev] = mh.sigmoid(z)
                ws.append([w_h, w_0, w_e, wJ] + ([cal["a"]] if cal else []))
                if arm.mrf:  # w_J = 0 reproduces H4 exactly
                    z0 = mh.fuse(lpc[ev], Hh[ev], I0[ev], has[ev], replace(wts, w_J=0.0), J)
                    z1 = mh.fuse(lpc[ev], Hh[ev], I0[ev], has[ev], replace(wts, w_J=0.0), None)
                    assert np.array_equal(z0, z1)
            Qs.append(Q)
        return {"Q": Qs, "w": ws}

    # ------------------------------------------------------------------ evaluation
    allrows = np.where(keep)[0]
    halves = [patient_halves(pid, seed) for seed in seeds]
    prior_rows = np.where(has & keep)[0]
    elig = {pos: mab.eligible(L, prior_bits, prior_rows, pos) for pos in (True, False)}

    # Cross-fitted predictions of the two halves come from differently fitted calibrations / weights, so
    # pooling them would mix scales inside each class ranking. Every ranking metric is therefore computed
    # within each half and averaged over the halves (the cross-validated estimator), for every arm alike.
    def cf_map(Q: np.ndarray, rows: np.ndarray, half: np.ndarray) -> float:
        return float(np.mean([pmc.subset_map(Q, L, rows[half[rows] == h])[0] for h in (0, 1)]))

    def cf_per_class(Q: np.ndarray, rows: np.ndarray, half: np.ndarray) -> np.ndarray:
        return np.nanmean([pmc.per_class_ap(Q, L, rows[half[rows] == h]) for h in (0, 1)], 0)

    def cf_stratum(Q: np.ndarray, rows: np.ndarray, half: np.ndarray, pos: bool) -> float:
        return float(np.nanmean([mab.stratum_map(Q, L, prior_bits, rows[half[rows] == h], pos, elig[pos]) for h in (0, 1)]))

    def evaluate(Qs: list[np.ndarray], base: np.ndarray, boot: bool, boot_subsets) -> dict:
        r = {}
        for sname, msk in subsets.items():
            rows = np.where(msk & keep)[0]
            if len(rows) == 0:
                continue
            vals = [cf_map(Q, rows, halves[i]) for i, Q in enumerate(Qs)]
            bvals = [cf_map(base, rows, halves[i]) for i in range(len(Qs))]
            e = {"n": len(rows), "mAP": float(np.mean(vals)), "range": [float(min(vals)), float(max(vals))],
                 "base": float(np.mean(bvals)), "mAP_pooled_seed0": pmc.subset_map(Qs[0], L, rows)[0],
                 "base_pooled": pmc.subset_map(base, L, rows)[0]}
            if boot and sname in boot_subsets:
                Q0, h0 = Qs[0], halves[0]
                e["delta_ci"] = pmc.bootstrap(pmc.by_patient(pid, rows),
                                              lambda rr: cf_map(Q0, rr, h0) - cf_map(base, rr, h0), seed=7)
            r[sname] = e
        rows, h0 = prior_rows, halves[0]
        dap = cf_per_class(Qs[0], rows, h0) - cf_per_class(base, rows, h0)
        r["dAP_groups_has_prior"] = {g: float(np.nanmean(dap[group_name == g])) for g in ("head", "medium", "tail")}
        r["stratum_AP"] = {("persistent vs resolved" if pos else "new vs absent"):
                           {"arm": cf_stratum(Qs[0], rows, h0, pos), "base": cf_stratum(base, rows, h0, pos),
                            "classes": int(len(elig[pos]))} for pos in (True, False)}
        r["mECE"] = ece(Qs[0][allrows], L[allrows])
        return r

    main_boot = {"full", PRIMARY, "no_prior", "has_prior"}
    res["arms"], models, q_store = {}, {}, {}
    for ename, files in EMISSIONS.items():
        P = np.mean([np.load(mc.PROBS / f) for f in files], 0).astype(np.float32)
        print(f"\n[3] emission {ename} ({', '.join(files)})")
        res["arms"][ename] = {}
        for arm in arms:
            if arm.filtered and (filt is None or ename != FILTER_EMISSION):
                continue
            tt = time.time()
            out = run_arm(arm, P, ename)
            if arm.name == "H0" and ename == "ens" and not args.limit:  # B-late reproduction
                bl = json.loads(BLATE.read_text())["fusion"]
                m_hp = pmc.subset_map(out["Q"][0], L, np.where(has)[0])[0]
                m_full = pmc.subset_map(out["Q"][0], L, allrows)[0]
                assert abs(m_hp - bl["has_prior"]["w1"]) < 1e-5 and abs(m_full - bl["full_val"]["w1"]) < 1e-5, \
                    f"H0 {m_hp:.6f}/{m_full:.6f} != B-late {bl['has_prior']['w1']:.6f}/{bl['full_val']['w1']:.6f}"
                print(f"    check: H0 reproduces B-late (has_prior {m_hp:.4f}, full {m_full:.4f})")
            if arm.name == "E":
                assert np.array_equal(out["Q"][0], P)
                z0 = mh.fuse(mh.logit(P), H_obs["full"], I_meta["full"], has, mh.FusionWeights(0.0, 0.0))
                assert np.allclose(mh.sigmoid(z0), P, atol=1e-6), "zero weights != emission"
            boot = arm.name in ("H0", "H1", "H3", "H4", "H4-F", "H4-acq")  # H2 / H5: point estimates only
            ev = evaluate(out["Q"], P, boot=boot and arm.name != "E",
                          boot_subsets=set(subsets) if arm.name in ("H4", "H4-F") else main_boot)
            ev["weights"] = out["w"]
            res["arms"][ename][arm.name] = ev
            q_store[(ename, arm.name)] = out["Q"][0]
            line = " | ".join(f"{k.split(' (')[0]} {v['mAP']:.4f}" + (f" (d {pmc.fmt_ci(v['delta_ci'])})" if "delta_ci" in v else "")
                              for k, v in ev.items() if isinstance(v, dict) and "mAP" in v and k in main_boot)
            wtxt = "" if not out["w"] else " | w(h,0,e,J[,a]) " + "; ".join(",".join(f"{x:.2f}" for x in w) for w in out["w"][:2])
            print(f"    {arm.name:5s} {line} | mECE {ev['mECE']:.4f}{wtxt} | {time.time() - tt:.0f}s")
            if arm.name in ("H4", "H4-F"):
                for k, v in ev.items():
                    if isinstance(v, dict) and "mAP" in v and k not in main_boot:
                        print(f"          {k:34s} n {v['n']:5d} | base {v['base']:.4f} -> {v['mAP']:.4f} "
                              f"[{v['range'][0]:.4f}, {v['range'][1]:.4f}]" + (f" | d {pmc.fmt_ci(v['delta_ci'])}" if "delta_ci" in v else ""))
                print(f"          dAP has_prior head/medium/tail: " + ", ".join(f"{g} {x:+.4f}" for g, x in ev["dAP_groups_has_prior"].items()))
                print(f"          stratum AP: " + "; ".join(f"{k} {v['base']:.4f} -> {v['arm']:.4f} ({v['classes']} cls)" for k, v in ev["stratum_AP"].items()))
            if arm.name == "H4" and not args.limit:  # deployable model: calibration + weights on all val rows
                lp = mh.logit(P)
                cal = mh.fit_calibration(lp, L)
                (w_h, w_0), _ = grid_search(mh.calibrate(lp, cal), H_obs["full"], I_meta["full"], allrows, arm)
                models[ename] = mh.MarkovHMM(priors["full"], feat.state_dict(), prev, cal, mh.FusionWeights(w_h, w_0), None, "observed")
                models[ename].save(out_dir / f"model_H4_{ename}.pt")

        # ---- comparators: v5 B-early (prior fed into the ML-Decoder queries), from the A/B dump
        if ename in ("s42", "s1024") and (AB_DUMP / "keys.txt").exists():
            keys = (AB_DUMP / "keys.txt").read_text().split("\n")
            pos = {k: i for i, k in enumerate(keys)}
            ridx = np.array([pos[f] for f in va["filename"]])
            assert (np.load(AB_DUMP / "labels.npy")[ridx] == L).all()
            seed = 42 if ename == "s42" else 1024
            for comp in ("B-early", "B-early0"):
                f = AB_DUMP / mab.ARMS[comp][seed]
                if not f.exists():
                    print(f"    {comp}: {f.name} missing")
                    continue
                Pb = np.load(f)[ridx].astype(np.float32)
                ev = evaluate([Pb] * len(seeds), P, boot=True, boot_subsets=main_boot)
                res["arms"][ename][comp] = ev
                print(f"    {comp:8s} " + " | ".join(f"{k.split(' (')[0]} {v['mAP']:.4f} (d vs E {pmc.fmt_ci(v['delta_ci'])})"
                                                      for k, v in ev.items() if isinstance(v, dict) and "delta_ci" in v))
                if comp == "B-early":  # decision rule: paired H4 - B-early at the matched seed (H1, H4-F exploratory)
                    for an in ("H4", "H1", "H4-F"):
                        if (ename, an) not in q_store:
                            continue
                        QH = q_store[(ename, an)]
                        tag = f"{an} - B-early" + ("" if an == "H4" else " (exploratory)")
                        res["arms"][ename][tag] = {}
                        for sname in ("full", PRIMARY, "has_prior", "no_prior"):
                            rows = np.where(subsets[sname] & keep)[0]
                            d = pmc.bootstrap(pmc.by_patient(pid, rows),
                                              lambda rr: cf_map(QH, rr, halves[0]) - cf_map(Pb, rr, halves[0]), seed=11)
                            res["arms"][ename][tag][sname] = d
                            print(f"    {tag} (seed {seed}) {sname:32s} {pmc.fmt_ci(d)}")
                q_store[(ename, comp)] = Pb
            res.setdefault("_comparator_files", {})[ename] = str(AB_DUMP)

    (out_dir / "results.json").write_text(json.dumps(res, indent=2, default=float))
    # seed-0 out-of-fold predictions of every arm (val row order = analysis/out/val_key_order.npy), for later analysis
    np.savez_compressed(out_dir / "oof_seed0.npz", filenames=va["filename"].to_numpy().astype(str), labels=L,
                        half_seed0=halves[0], has_prior=has, **{f"{e}__{a}": q.astype(np.float16) for (e, a), q in q_store.items()})
    print(f"\nwrote {out_dir / 'results.json'} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
