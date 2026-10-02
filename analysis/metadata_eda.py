"""EDA: which PadChest patient / study / acquisition metadata are associated with the 30 labels? (2026-10-01)

Step 2 of docs/plan/2026-10-01-markov-hmm-redesign-plan.md. Train fold only for the screening; val is
used only for [5] (does metadata add anything on top of the image model?).
Data rules: analysis/patient_meta.py (column whitelist, fold images only, labels from the fold CSVs).

[1] univariate screen, every field x class: Cramer's V, mutual information, chi-square p (BH-FDR),
    the most extreme level's log-OR (Haldane) with a patient-clustered Poisson-bootstrap 95% CI,
    and single-feature AUROC for numeric fields. Units: study for patient/study fields, image for
    view/acquisition fields.
[2] multivariate: per-class metadata-only logistic, patient-grouped 5-fold CV on train images;
    held-out macro AUROC / AP, each feature group alone, all, and all minus each group.
[3] transition modulation: persistence a = P(1|1) and onset b = P(1|0) by sex / age band / acuity /
    gap on consecutive train (patient, date) states.
[4] figures (plain SVG, analysis/out/metadata_eda/figures/).
[5] incremental value over the image (val, cross-fitted on patient halves, mAP per half then averaged):
    stacking logit p + metadata vs logit p alone and vs recalibration only; paired patient-clustered bootstrap.

    srun -c 4 --mem=32G -p general --time=1:00:00 .venv/bin/python analysis/metadata_eda.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluate"))
import markov_check as mc  # noqa: E402
import markov_hmm as mh  # noqa: E402
import patient_markov_check as pmc  # noqa: E402
import patient_meta as pm  # noqa: E402
import patient_prior as pp  # noqa: E402
from class_groups import frequency_groups, load_train_prevalence  # noqa: E402
from torchmetrics.functional.classification import multilabel_auroc  # noqa: E402

ROOT = mc.ROOT
OUT = ROOT / "analysis/out/metadata_eda"
N_BOOT = 200
MIN_LEVEL = 50          # rarer levels are pooled into "other"
MIN_TOP_LEVEL = 200     # the reported extreme level must have at least this many rows
AGE_BANDS = (0, 50, 70, 80, 200)
EMISSIONS = {"s42": ["push__img768_dp01_seed42__model_swa@768_flip.npy"], "ens": mc.ENSEMBLE}


# ---------------------------------------------------------------------- tables
def study_table(df: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """One row per study (labels are per study): patient fields + the study's history features."""
    st = df.groupby("sid").agg(pid=("pid", "first"), date=("date", "first"), code=("code", "first"),
                               sex=("sex", "first"), age=("age", "first"), acuity=("acuity", "max"),
                               n_img=("filename", "size"), pediatric=("Pediatric", "first"))
    si = pm.state_index(states, st["pid"].to_numpy(), st["date"].to_numpy())
    ok = si >= 0
    st["n_prior_states"] = np.where(ok, states["rank"].to_numpy()[np.maximum(si, 0)], np.nan)
    st["gap_days"] = np.where(ok, states["gap"].to_numpy()[np.maximum(si, 0)], np.nan)
    st["year"] = st["date"].dt.year
    return st


def binned(x: pd.Series, q: int = 5) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce")
    try:
        b = pd.qcut(v, q, duplicates="drop")
    except ValueError:
        b = pd.cut(v, q)
    return b.astype(str).where(v.notna(), "NA")


def categorical(x: pd.Series) -> pd.Series:
    s = x.astype(str).where(x.notna(), "NA")
    vc = s.value_counts()
    return s.where(s.map(vc) >= MIN_LEVEL, "other")


# ---------------------------------------------------------------------- [1] univariate screen
def screen(fields: dict[str, pd.Series], numeric: dict[str, pd.Series], Y: np.ndarray, pid: np.ndarray,
           unit: str, classes: list[str], rng: np.random.Generator) -> list[dict]:
    n, C = Y.shape
    codes, uniq = pd.factorize(pid)
    Wb = rng.poisson(1.0, (N_BOOT, len(uniq))).astype(np.float32)[:, codes]  # patient-clustered weights
    rows = []
    for name, s in fields.items():
        lv, levels = pd.factorize(s)
        X = np.eye(len(levels), dtype=np.float32)[lv]                 # (n, K)
        nk = X.sum(0)
        pos = X.T @ Y                                                  # (K, C)
        p1 = Y.mean(0)
        exp1 = nk[:, None] * p1[None]
        exp0 = nk[:, None] * (1 - p1)[None]
        chi2 = (((pos - exp1) ** 2) / np.maximum(exp1, 1e-12) + (((nk[:, None] - pos) - exp0) ** 2) / np.maximum(exp0, 1e-12)).sum(0)
        dof = max(len(levels) - 1, 1)
        pval = stats.chi2.sf(chi2, dof)
        V = np.sqrt(chi2 / n)
        pj = np.stack([pos, nk[:, None] - pos], -1) / n                # (K, C, 2)
        pk = (nk / n)[:, None, None]
        py = np.stack([p1, 1 - p1], -1)[None]
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = np.nansum(np.where(pj > 0, pj * np.log2(np.maximum(pj, 1e-300) / (pk * py)), 0), axis=(0, 2))
        # level-vs-rest log OR (Haldane), point and bootstrap
        tot1 = Y.sum(0)

        def log_or(pos_, nk_, tot1_, n_):
            a = pos_ + 0.5
            b = nk_[..., None] - pos_ + 0.5
            c = tot1_[..., None, :] - pos_ + 0.5
            d = (n_[..., None, None] - nk_[..., None]) - c + 1.0
            return np.log(a * d / (b * c))

        lor = log_or(pos, nk, tot1, np.float64(n))
        bw_nk = Wb @ X                                                 # (B, K)
        bw_pos = np.stack([Wb @ (X[:, k:k + 1] * Y) for k in range(len(levels))], 1)  # (B, K, C)
        bw_tot1 = Wb @ Y                                               # (B, C)
        bw_n = Wb.sum(1)
        lor_b = log_or(bw_pos, bw_nk, bw_tot1, bw_n)                   # (B, K, C)
        eligible = (nk >= MIN_TOP_LEVEL) & (np.array(levels) != "NA") & (len(levels) > 1)
        num = numeric.get(name)
        if num is not None:
            v = pd.to_numeric(num, errors="coerce").to_numpy(float)
            okv = ~np.isnan(v)
            rk = stats.rankdata(v[okv])
        for c in range(C):
            if eligible.any():
                k = int(np.argmax(np.where(eligible, np.abs(lor[:, c]), -np.inf)))
                lo, hi = np.percentile(lor_b[:, k, c], [2.5, 97.5])
                top = {"level": str(levels[k]), "n_level": int(nk[k]), "prev_level": float(pos[k, c] / nk[k]),
                       "log_or": float(lor[k, c]), "log_or_ci": [float(lo), float(hi)]}
            else:
                top = {"level": None, "n_level": 0, "prev_level": np.nan, "log_or": np.nan, "log_or_ci": [np.nan, np.nan]}
            auroc = np.nan
            if num is not None:
                y = Y[okv, c]
                n1 = y.sum()
                if 0 < n1 < len(y):
                    auroc = float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))
            rows.append({"field": name, "unit": unit, "class": classes[c], "n_levels": len(levels),
                         "cramers_v": float(V[c]), "mi_bits": float(mi[c]), "chi2_p": float(pval[c]),
                         "auroc": auroc, **top})
    return rows


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.clip(q, 0, 1)
    return out


# ---------------------------------------------------------------------- [2] multivariate
def image_design(df: pd.DataFrame, states: pd.DataFrame, feat: pm.Featurizer) -> dict[str, np.ndarray]:
    si = pm.state_index(states, df["pid"].to_numpy(), df["date"].to_numpy())
    ok = si >= 0
    rank = np.where(ok, states["rank"].to_numpy()[np.maximum(si, 0)], 0)
    gap = np.where(ok, states["gap"].to_numpy()[np.maximum(si, 0)], np.nan)
    phi = feat(df["sex"].to_numpy(), df["age"].to_numpy(), np.zeros(len(df), int))[:, :5]
    acq, _ = pm.acquisition_features(df)
    groups = {
        "patient": phi,
        "temporal": np.column_stack([rank > 0, pm.gap_feature(gap), np.log1p(rank)]).astype(float),
        "view": np.column_stack([df["acuity"].eq(1), df["acuity"].eq(2), df["proj"].eq("L")]).astype(float),
        "acquisition": acq,
    }
    for k, v in groups.items():
        mu, sd = v.mean(0), v.std(0) + 1e-8
        groups[k] = (v - mu) / sd
    return groups


def patient_folds(pid: np.ndarray, k: int, seed: int) -> np.ndarray:
    codes, uniq = pd.factorize(pid)
    f = np.random.default_rng(seed).integers(0, k, len(uniq))
    return f[codes]


def cv_metadata_only(groups: dict, Y: np.ndarray, pid: np.ndarray, use: list[str], l2: float = 1e-4) -> dict:
    X = np.column_stack([np.ones(len(Y))] + [groups[g] for g in use])
    fold = patient_folds(pid, 5, seed=0)
    Z = np.zeros(Y.shape)
    for f in range(5):
        tr_, te_ = fold != f, fold == f
        fit = mh.fit_logistic(X[tr_], Y[tr_], l2=np.r_[0.0, [l2] * (X.shape[1] - 1)])
        Z[te_] = mh.logistic_logits(X[te_], fit)
    Yt = torch.from_numpy(Y.astype(np.int64))
    auc = multilabel_auroc(torch.from_numpy(Z).float(), Yt, num_labels=Y.shape[1], average="none").numpy()
    ap = pmc.per_class_ap(mh.sigmoid(Z).astype(np.float32), Y, np.arange(len(Y)))
    return {"groups": use, "macro_auroc": float(np.nanmean(auc)), "macro_ap": float(np.nanmean(ap)),
            "auroc": auc.tolist(), "ap": ap.tolist()}


# ---------------------------------------------------------------------- [3] transition modulation
def chain_by(prev_c, next_c, mask) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pc, nc = prev_c[mask], next_c[mask]
    n1 = pc.sum(0)
    n0 = len(pc) - n1
    a = ((pc & nc).sum(0) + 1) / (n1 + 2)
    b = (((1 - pc) & nc).sum(0) + 1) / (n0 + 2)
    return a, b, n1


# ---------------------------------------------------------------------- [4] figures (plain SVG, no plotting deps)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference palette slots 1-3 (all-pairs CVD-safe)
FONT = "font-family='Helvetica, Arial, sans-serif'"
_STYLE = {"fs": 1.0, "title": True}  # paper mode: fs 1.3, no inline titles (the LaTeX caption carries them)
PAPER_FIELDS = ["study_acuity", "projection", "gap_days", "age", "n_prior_states", "sex", "study_year",
                "SpatialResolution_DICOM", "RelativeXRayExposure_DICOM", "ExposureTime", "XRayTubeCurrent_DICOM",
                "Rows_DICOM", "Manufacturer_DICOM", "Modality_DICOM"]


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _text(x, y, s, size=11, fill=INK2, anchor="start", rot=None, weight="normal") -> str:
    tr = f" transform='rotate({rot} {x:.1f} {y:.1f})'" if rot is not None else ""
    return (f"<text x='{x:.1f}' y='{y:.1f}' {FONT} font-size='{size * _STYLE['fs']:.1f}' fill='{fill}' "
            f"text-anchor='{anchor}' font-weight='{weight}'{tr}>{_esc(s)}</text>")


def _title(s: str) -> list[str]:
    return [_text(16, 26, s, 14, INK, weight="bold")] if _STYLE["title"] else []


def _svg(w, h, body: list[str], path: Path) -> None:
    y0 = 0 if _STYLE["title"] else 34  # crop the empty title band in paper mode
    path.write_text(f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h - y0}' viewBox='0 {y0} {w} {h - y0}'>"
                    f"<rect y='{y0}' width='{w}' height='{h - y0}' fill='{SURFACE}'/>" + "".join(body) + "</svg>")


def _nice_ticks(hi: float, n: int = 3) -> np.ndarray:
    raw = hi / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    return np.arange(0, hi + 1e-12, step)


def _seq(v: float, vmax: float) -> str:
    t = min(max(v / vmax, 0.0), 1.0) * (len(SEQ) - 1)
    i = min(int(t), len(SEQ) - 2)
    f = t - i
    a, b = (np.array([int(c[k:k + 2], 16) for k in (1, 3, 5)]) for c in (SEQ[i], SEQ[i + 1]))
    r, g, bb = (a + (b - a) * f).round().astype(int)
    return f"#{r:02x}{g:02x}{bb:02x}"


def _axis_y(x0, y0, h, lo, hi, ticks, fmt, w) -> list[str]:
    out = []
    for t in ticks:
        y = y0 + h - (t - lo) / (hi - lo) * h
        out.append(f"<line x1='{x0}' x2='{x0 + w}' y1='{y:.1f}' y2='{y:.1f}' stroke='{GRID}' stroke-width='1'/>")
        out.append(_text(x0 - 6, y + 4, fmt(t), 10, anchor="end"))
    return out


def figures(assoc: pd.DataFrame, classes, prev, age_curves, persist, multiv, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # (a) heatmap: Cramer's V, class (by prevalence) x field
    order = [classes[i] for i in np.argsort(-prev)]
    fields = assoc.groupby("field")["cramers_v"].max().sort_values(ascending=False).index.tolist()
    if not _STYLE["title"]:
        fields = [f for f in PAPER_FIELDS if f in fields]
    M = assoc.pivot(index="class", columns="field", values="cramers_v").reindex(index=order, columns=fields).to_numpy()
    vmax = max(0.3, float(np.nanmax(M)))
    cw, ch, left, top = (30, 18, 210, 60) if _STYLE["title"] else (36, 21, 265, 50)
    w, h = left + cw * len(fields) + 120, top + ch * len(order) + 190
    b = _title("Metadata × label association, Cramér's V (train fold; classes ordered by prevalence)")
    for i, cname in enumerate(order):
        y = top + i * ch
        b.append(_text(left - 8, y + ch * 0.7, cname, 11, anchor="end"))
        for j in range(len(fields)):
            v = M[i, j]
            b.append(f"<rect x='{left + j * cw + 1}' y='{y + 1}' width='{cw - 2}' height='{ch - 2}' rx='2' "
                     f"fill='{_seq(0 if np.isnan(v) else v, vmax)}'><title>{_esc(cname)} × {_esc(fields[j])}: V = {v:.3f}</title></rect>")
            if v >= 0.15:
                b.append(_text(left + j * cw + cw / 2, y + ch * 0.68, f"{v:.2f}"[1:], 9,
                               "#ffffff" if v > vmax * 0.55 else INK, anchor="middle"))
    yb = top + ch * len(order) + 8
    for j, f in enumerate(fields):
        b.append(_text(left + j * cw + cw / 2 + 4, yb, f.replace("_DICOM", ""), 10, anchor="end", rot=-60))
    lx = left + cw * len(fields) + 24
    for k in range(101):
        b.append(f"<rect x='{lx}' y='{top + (100 - k) * 2}' width='14' height='2.2' fill='{_seq(k / 100 * vmax, vmax)}'/>")
    b += [_text(lx + 20, top + 8, f"{vmax:.2f}", 10), _text(lx + 20, top + 202, "0", 10),
          _text(lx, top + 224, "Cramér's V", 11, INK)]
    _svg(w, h, b, out / "assoc_heatmap.svg")

    # (b) prevalence by age band, small multiples for the most age-associated classes
    mids, top_c = np.array(age_curves["mid"]), age_curves["top"]
    pw, ph, gx, gy = 250, 150, 70, 60
    W, H = 3 * (pw + gx) + 20, 2 * (ph + gy) + 70
    b = _title("Prevalence by age band: the six classes most associated with age (train studies)")
    for k, cname in enumerate(top_c):
        r, c = divmod(k, 3)
        x0, y0 = 60 + c * (pw + gx), 60 + r * (ph + gy)
        y = np.array(age_curves["prev"][cname], float)
        ticks = _nice_ticks(float(np.nanmax(y)))
        hi = max(float(np.nanmax(y)) * 1.12, float(ticks[-1]) * 1.02) or 1.0
        b += _axis_y(x0, y0, ph, 0, hi, ticks, lambda t: f"{100 * t:g}%", pw)
        b.append(_text(x0, y0 - 8, cname, 12, INK, weight="bold"))
        xs = x0 + (mids - 20) / 80 * pw
        ys = y0 + ph - np.nan_to_num(y) / hi * ph
        pts = " ".join(f"{a:.1f},{bb:.1f}" for a, bb in zip(xs, ys))
        b.append(f"<polyline points='{pts}' fill='none' stroke='{CAT[0]}' stroke-width='2' stroke-linejoin='round'/>")
        for a, bb, m, v in zip(xs, ys, mids, y):
            b.append(f"<circle cx='{a:.1f}' cy='{bb:.1f}' r='4' fill='{CAT[0]}' stroke='{SURFACE}' stroke-width='2'>"
                     f"<title>{_esc(cname)}, age {int(m) - 5}-{int(m) + 4}: {v:.3f}</title></circle>")
        for m in (30, 50, 70, 90):
            b.append(_text(x0 + (m - 20) / 80 * pw, y0 + ph + 16, str(m), 10, anchor="middle"))
        b.append(f"<line x1='{x0}' x2='{x0 + pw}' y1='{y0 + ph}' y2='{y0 + ph}' stroke='{INK2}' stroke-width='1'/>")
    b.append(_text(W / 2, H - 12, "age at study (years); y = prevalence among studies in the band", 11, anchor="middle"))
    _svg(W, H, b, out / "age_prevalence.svg")

    # (c) persistence a = P(1|1) by acuity of the later state
    names = persist["classes"]
    sp, x0, y0, ph = 46, 70, 70, 260
    W, H = max(x0 + sp * len(names) + 40, 780), y0 + ph + 190
    b = _title("Persistence P(finding at t | finding at t−1), by acuity of the later study (train pairs)")
    b += _axis_y(x0, y0, ph, 0, 1, [0, 0.25, 0.5, 0.75, 1.0], lambda t: f"{t:.2f}", sp * len(names))
    for j, (lab, col) in enumerate(zip(pm.ACUITY_NAMES, CAT)):
        lx = x0 + j * int(150 * _STYLE["fs"])
        b.append(f"<circle cx='{lx + 6}' cy='{48}' r='5' fill='{col}'/>")
        b.append(_text(lx + 16, 52, f"acuity at t: {lab}", 11))
        for i, a in enumerate(persist["a"][lab]):
            cx = x0 + i * sp + sp / 2 + (j - 1) * 10
            cy = y0 + ph - a * ph
            b.append(f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='5' fill='{col}' stroke='{SURFACE}' stroke-width='2'>"
                     f"<title>{_esc(names[i])}, {lab}: {a:.3f}</title></circle>")
    for i, cname in enumerate(names):
        b.append(_text(x0 + i * sp + sp / 2 + 4, y0 + ph + 14, cname, 10, anchor="end", rot=-55))
    _svg(W, H, b, out / "persistence_by_acuity.svg")

    # (d) held-out metadata-only AUROC per class (all groups); bars diverge from chance (0.5)
    auc = np.array(multiv["all"]["auroc"], float)
    o = [i for i in np.argsort(-np.nan_to_num(auc, nan=-1)) if not np.isnan(auc[i])]
    lo = min(0.4, float(np.floor(np.nanmin(auc) * 10) / 10))
    left, top, bw, bh = (210, 50, 380, 16) if _STYLE["title"] else (265, 50, 330, 19)
    W, H = left + bw + 60, top + bh * len(o) + 60
    b = _title("Held-out AUROC of metadata alone (patient-grouped 5-fold CV, train fold)")
    sx = lambda v: left + (v - lo) / (1.0 - lo) * bw  # noqa: E731
    for t in np.arange(lo, 1.0001, 0.1):
        b.append(f"<line x1='{sx(t):.1f}' x2='{sx(t):.1f}' y1='{top}' y2='{top + bh * len(o)}' stroke='{GRID}'/>")
        b.append(_text(sx(t), top + bh * len(o) + 16, f"{t:.1f}", 10, anchor="middle"))
    b.append(f"<line x1='{sx(0.5):.1f}' x2='{sx(0.5):.1f}' y1='{top}' y2='{top + bh * len(o)}' stroke='{INK2}'/>")
    for r, i in enumerate(o):
        y = top + r * bh
        x_a, x_b = sorted((sx(0.5), sx(auc[i])))
        b.append(_text(left - 8, y + bh * 0.72, classes[i], 11, anchor="end"))
        b.append(f"<rect x='{x_a:.1f}' y='{y + 2}' width='{max(x_b - x_a, 1):.1f}' height='{bh - 4}' rx='2' "
                 f"fill='{CAT[0]}'><title>{_esc(classes[i])}: {auc[i]:.3f}</title></rect>")
        b.append(_text(x_b + 4, y + bh * 0.72, f"{auc[i]:.2f}", 10))
    b.append(_text(left + bw / 2, H - 10, "AUROC (line = chance, 0.5)", 11, anchor="middle"))
    _svg(W, H, b, out / "metadata_only_auroc.svg")


# ---------------------------------------------------------------------- [5] incremental value over the image
def incremental(tr: pd.DataFrame, va: pd.DataFrame, feat: pm.Featurizer, res: dict) -> None:
    """Per-class stacking a_c logit p + b_c + gamma_c . phi, fitted on one patient half of val and scored on
    the other. The halves are fitted separately, so mAP is computed within each half and averaged (pooling
    them would mix two scales inside each class ranking). 'recalibration only' (gamma = 0) is the reference
    that isolates what the metadata adds; it is monotone per class, so it should equal the emission."""
    order = np.load(mc.KEY_ORDER)
    vv = va.iloc[order].reset_index(drop=True)
    L = np.load(mc.PROBS / "labels.npy").astype(np.uint8)
    assert (pp.decode(vv["code"].to_numpy()) == L).all(), "rebuilt val row order does not reproduce labels.npy"
    sg_all, _ = pp.study_table(tr, va)
    states_all = pm.date_states_meta(sg_all, pd.concat([tr, va]))
    g_va = image_design(vv, states_all, feat)
    pid = vv["pid"].to_numpy()
    half = patient_folds(pid, 2, seed=0)
    rows_all = np.arange(len(vv))
    frontal = np.where(vv["proj"].isin(mc.FRONTAL).to_numpy())[0]

    def cf(Q, rows):
        return float(np.mean([pmc.subset_map(Q, L, rows[half[rows] == h])[0] for h in (0, 1)]))

    print("\n[5] incremental value over the image model on val: per-class stacking a_c logit p + b_c + gamma_c phi,")
    print("    cross-fitted on patient halves, mAP within each half then averaged; paired patient-clustered bootstrap")
    res["incremental"] = {}
    variants = {"recalibration only": [], "patient": ["patient"], "view": ["view"], "acquisition": ["acquisition"],
                "patient+view+acquisition": ["patient", "view", "acquisition"]}
    for ename, files in EMISSIONS.items():
        P = np.mean([np.load(mc.PROBS / f) for f in files], 0).astype(np.float32)
        lp = mh.logit(P)
        Qs = {}
        for vname, use in variants.items():
            X = np.column_stack([np.ones(len(vv))] + [g_va[g] for g in use])
            Z = np.zeros(lp.shape)
            for h in (0, 1):
                f_, e_ = half != h, half == h
                fit = mh.fit_logistic(X[f_], L[f_].astype(float), own=lp[f_][..., None],
                                      l2=np.r_[0.0, [1e-3] * (X.shape[1] - 1)], l2_own=0.0)
                Z[e_] = mh.logistic_logits(X[e_], fit, lp[e_][..., None])
            Qs[vname] = mh.sigmoid(Z).astype(np.float32)
        for subset, rows in (("all", rows_all), ("frontal", frontal)):
            base = cf(P, rows)
            for vname, Q in Qs.items():
                QR = Qs["recalibration only"]
                m_ = cf(Q, rows)
                d = pmc.bootstrap(pmc.by_patient(pid, rows), lambda r: cf(Q, r) - cf(P, r), seed=3)
                dr = None if vname == "recalibration only" else \
                    pmc.bootstrap(pmc.by_patient(pid, rows), lambda r: cf(Q, r) - cf(QR, r), seed=4)
                key = f"{ename} | {subset} | {vname}"
                print(f"    {key:52s} base {base:.4f} -> {m_:.4f} | d vs emission {pmc.fmt_ci(d)}"
                      + (f" | d vs recal-only {pmc.fmt_ci(dr)}" if dr else ""))
                res["incremental"][key] = {"base": base, "stacked": m_, "delta_vs_emission": d, "delta_vs_recal": dr}


def redraw_figures() -> None:
    """Figures from the saved outputs (no data re-run): web SVGs with titles, plus paper versions (no inline
    titles, larger type, curated heatmap fields) converted to vector PDF with rsvg-convert."""
    import subprocess
    classes = list(torch.load(pp.LABEL_INFO, weights_only=False)["class_names"])
    _, prev, _ = load_train_prevalence(classes)
    assoc = pd.read_csv(OUT / "assoc.csv")
    r = json.loads((OUT / "eda.json").read_text())
    multiv = {"all": {"auroc": r["multivariate_per_class"]["auroc_all"]}}
    figures(assoc, classes, prev, r["age_curves"], r["persistence_by_acuity"], multiv, OUT / "figures")
    paper = ROOT / "docs/paper/markov/figures"
    _STYLE.update(fs=1.3, title=False)
    try:
        figures(assoc, classes, prev, r["age_curves"], r["persistence_by_acuity"], multiv, paper)
    finally:
        _STYLE.update(fs=1.0, title=True)
    for svg in sorted(paper.glob("*.svg")):
        subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(svg.with_suffix(".pdf")), str(svg)], check=True)
        print(f"wrote {svg.with_suffix('.pdf')}")


# ---------------------------------------------------------------------- main
def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--limit", type=int, default=0, help="dry run on the first N train patients")
    ap_.add_argument("--figures-only", action="store_true",
                     help="redraw the figures from eda.json / assoc.csv: web SVGs here, paper PDFs in docs/paper/markov/figures")
    ap_.add_argument("--incremental-only", action="store_true",
                     help="rerun only [5] and merge it into the existing eda.json")
    args = ap_.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    res: dict = {"n_boot": N_BOOT}
    if args.figures_only:
        redraw_figures()
        return

    classes = list(torch.load(pp.LABEL_INFO, weights_only=False)["class_names"])
    _, prev, _ = load_train_prevalence(classes)
    grp = frequency_groups(prev)
    group_name = np.where(grp["head"], "head", np.where(grp["tail"], "tail", "medium"))
    tr, va = pm.load_folds_meta(classes, acq=True)
    if args.limit:
        keep = set(pd.unique(tr["pid"])[: args.limit])
        tr = tr[tr["pid"].isin(keep)].reset_index(drop=True)
    print(f"train images {len(tr)} | patients {tr['pid'].nunique()} | val images {len(va)}")

    sg_tr, stats_ = pp.study_table(tr, va.iloc[:0])
    states_tr = pm.date_states_meta(sg_tr, tr)
    st = study_table(tr[tr["sid"].isin(sg_tr.index)], states_tr)
    Ys = pp.decode(st["code"].to_numpy()).astype(np.float32)
    Yi = pp.decode(tr["code"].to_numpy()).astype(np.float32)
    print(f"studies {len(st)} | (patient, date) states {len(states_tr)} | excluded ambiguous studies {stats_['n_ambiguous']}")
    if args.incremental_only:
        feat = pm.Featurizer().fit(states_tr["sex"], states_tr["age"], states_tr["acuity"])
        res = json.loads((OUT / "eda.json").read_text())
        incremental(tr, va, feat, res)
        (OUT / "eda.json").write_text(json.dumps(res, indent=2, default=float))
        print(f"\nupdated [5] in {OUT / 'eda.json'}")
        return

    # ---- [1] univariate screen
    study_fields = {"sex": categorical(st["sex"]), "age": binned(st["age"]), "pediatric": categorical(st["pediatric"]),
                    "n_prior_states": binned(st["n_prior_states"].clip(upper=10), 4),
                    "gap_days": binned(st["gap_days"]), "study_year": categorical(st["year"]),
                    "study_acuity": categorical(st["acuity"].map(dict(enumerate(pm.ACUITY_NAMES))))}
    study_num = {"age": st["age"], "n_prior_states": st["n_prior_states"], "gap_days": st["gap_days"], "study_year": st["year"]}
    img_fields = {"projection": categorical(tr["proj"]), "view_position": categorical(tr["view_pos"])}
    for c in pm.ACQ_CATEGORICAL:
        if c != "Pediatric":
            img_fields[c] = categorical(tr[c])
    img_num = {}
    for c in pm.ACQ_NUMERIC:
        img_fields[c] = binned(tr[c])
        img_num[c] = tr[c]
    rows = screen(study_fields, study_num, Ys, st["pid"].to_numpy(), "study", classes, rng)
    rows += screen(img_fields, img_num, Yi, tr["pid"].to_numpy(), "image", classes, rng)
    assoc = pd.DataFrame(rows)
    assoc["group"] = assoc["class"].map(dict(zip(classes, group_name)))
    assoc["q_bh"] = bh(assoc["chi2_p"].to_numpy())
    assoc.to_csv(OUT / "assoc.csv", index=False)
    print("\n[1] univariate screen (train fold): strongest field per class by Cramer's V")
    for c in np.argsort(-prev):
        a = assoc[assoc["class"] == classes[c]].sort_values("cramers_v", ascending=False).head(3)
        desc = " | ".join(f"{r.field} V={r.cramers_v:.3f}"
                          + (f" (top '{r.level}' logOR {r.log_or:+.2f} [{r.log_or_ci[0]:+.2f},{r.log_or_ci[1]:+.2f}])" if r.level else "")
                          for r in a.itertuples())
        print(f"    {classes[c]:30s} {group_name[c]:6s} {desc}")
    by_field = assoc.groupby(["field", "unit"]).agg(max_V=("cramers_v", "max"), mean_V=("cramers_v", "mean"),
                                                    max_MI=("mi_bits", "max"),
                                                    n_sig=("q_bh", lambda q: int((q < 0.05).sum()))).sort_values("max_V", ascending=False)
    print("\n    by field (max / mean Cramer's V over classes, classes with q_BH < 0.05):")
    for (f, u), r in by_field.iterrows():
        print(f"      {f:32s} {u:5s} max V {r.max_V:.3f} | mean V {r.mean_V:.3f} | max MI {r.max_MI:.4f} bits | sig classes {int(r.n_sig):2d}/30")
    res["by_field"] = by_field.reset_index().to_dict(orient="records")
    for fld in ("age", "gap_days", "n_prior_states"):
        a = assoc[(assoc.field == fld)].sort_values("auroc", key=lambda s: -np.abs(s - 0.5)).head(5)
        print(f"    single-feature AUROC, {fld}: " + ", ".join(f"{r['class']} {r.auroc:.3f}" for _, r in a.iterrows()))

    # ---- [2] multivariate, metadata only
    feat = pm.Featurizer().fit(states_tr["sex"], states_tr["age"], states_tr["acuity"])
    groups = image_design(tr, states_tr, feat)
    names = list(groups)
    variants = {"all": names, **{f"only {g}": [g] for g in names}, **{f"all - {g}": [x for x in names if x != g] for g in names}}
    multiv = {}
    print("\n[2] metadata-only logistic, patient-grouped 5-fold CV on train images (held-out, macro over 30 classes)")
    print(f"    prevalence baseline: macro AP {Yi.mean(0).mean():.4f}, AUROC 0.5")
    for k, use in variants.items():
        r = cv_metadata_only(groups, Yi, tr["pid"].to_numpy(), use)
        multiv[k] = r
        print(f"    {k:22s} macro AUROC {r['macro_auroc']:.4f} | macro AP {r['macro_ap']:.4f}")
    res["multivariate"] = {k: {kk: v[kk] for kk in ("groups", "macro_auroc", "macro_ap")} for k, v in multiv.items()}
    res["multivariate_per_class"] = {"classes": classes, "auroc_all": multiv["all"]["auroc"], "ap_all": multiv["all"]["ap"]}

    # ---- [3] transition modulation
    sn = states_tr
    has = sn["prev"].to_numpy() >= 0
    nxt_i = np.where(has)[0]
    prv_i = sn["prev"].to_numpy()[nxt_i]
    prev_c = pp.decode(sn["code"].to_numpy()[prv_i])
    next_c = pp.decode(sn["code"].to_numpy()[nxt_i])
    later = sn.iloc[nxt_i]
    a0, b0, n10 = chain_by(prev_c, next_c, np.ones(len(nxt_i), bool))
    strata = {"sex": later["sex"].to_numpy(),
              "age_band": pd.cut(later["age"], AGE_BANDS, right=False, labels=["<50", "50-69", "70-79", ">=80"]).astype(str).to_numpy(),
              "acuity": later["acuity"].map(dict(enumerate(pm.ACUITY_NAMES))).to_numpy(),
              "gap": np.select([later["gap"] <= 30, later["gap"] <= 365], ["<=30d", "31-365d"], ">365d")}
    print(f"\n[3] persistence a = P(1|1) and onset b = P(1|0) by stratum of the later state | pairs {len(nxt_i)}")
    print("    mean over classes with >= 30 prior positives in the stratum (pooled a over the same classes in brackets)")
    res["transition_strata"] = {}
    strata = {k: pd.Series(v, dtype=object).fillna("NA").astype(str).to_numpy() for k, v in strata.items()}
    for sname, lab in strata.items():
        for lv in sorted(set(lab)):
            m = lab == lv
            if m.sum() < 200:
                print(f"    {sname:8s} {lv:14s} pairs {int(m.sum()):6d} | skipped (< 200 pairs)")
                continue
            a, b, n1 = chain_by(prev_c, next_c, m)
            ok = n1 >= 30
            print(f"    {sname:8s} {lv:14s} pairs {int(m.sum()):6d} | classes {int(ok.sum()):2d} | mean a {a[ok].mean():.3f} "
                  f"[{a0[ok].mean():.3f}] | mean b {b.mean():.4f} [{b0.mean():.4f}]")
            res["transition_strata"][f"{sname}={lv}"] = {"pairs": int(m.sum()), "n_classes": int(ok.sum()),
                                                         "mean_a": float(a[ok].mean()), "mean_a_pooled": float(a0[ok].mean()),
                                                         "mean_b": float(b.mean()), "mean_b_pooled": float(b0.mean()),
                                                         "a": a.tolist(), "b": b.tolist(), "n1": n1.tolist()}
    acu = strata["acuity"]
    cls_ok = [c for c in np.argsort(-n10) if all(chain_by(prev_c, next_c, acu == lv)[2][c] >= 30 for lv in pm.ACUITY_NAMES)]
    persist = {"classes": [classes[c] for c in cls_ok],
               "a": {lv: [float(chain_by(prev_c, next_c, acu == lv)[0][c]) for c in cls_ok] for lv in pm.ACUITY_NAMES}}

    # ---- [4] figures
    age_assoc = assoc[assoc.field == "age"].sort_values("cramers_v", ascending=False)  # effect x prevalence
    top_age = age_assoc["class"].head(6).tolist()
    band = pd.cut(st["age"], [0, 30, 40, 50, 60, 70, 80, 90, 120], right=False)
    mids = np.array([25, 35, 45, 55, 65, 75, 85, 95], float)
    curves = {"top": top_age, "mid": mids.tolist(), "n": band.value_counts(sort=False).tolist(),
              "prev": {cn: pd.Series(Ys[:, classes.index(cn)]).groupby(band.to_numpy()).mean().reindex(band.cat.categories).tolist()
                       for cn in top_age}}
    figures(assoc, classes, prev, curves, persist, multiv, OUT / "figures")
    res["age_curves"] = curves
    res["persistence_by_acuity"] = persist

    # ---- [5] incremental value over the image model (val, cross-fit)
    if not args.limit:
        incremental(tr, va, feat, res)

    (OUT / "eda.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"\nwrote {OUT / 'eda.json'}, {OUT / 'assoc.csv'}, figures in {OUT / 'figures'}")


if __name__ == "__main__":
    main()
