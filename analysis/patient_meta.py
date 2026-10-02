"""Whitelisted PadChest patient / study metadata for the post-prediction patient HMM (2026-10-01).

Used by analysis/metadata_eda.py and analysis/markov_hmm_check.py
(plan: docs/plan/2026-10-01-markov-hmm-redesign-plan.md).

Rules:
  * internal train/val folds only -- every joined ImageID is a fold filename; never join
    dev/test/leaderboard images to the PadChest CSV (it carries the evaluation images' labels);
  * the CSV is read with a column whitelist; the label/report columns in FORBIDDEN are never read;
  * PatientBirth and PatientSex_DICOM are allowed for the internal folds (user decision, 2026-10-01;
    analysis/patient_prior.py still excludes them, so priors.pt / v5 stay reproducible);
  * labels come only from the 30-class label_list of data/{train,val}_fold_1.csv (shard order).

The chain / prior lookup itself is reused from analysis/patient_prior.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patient_prior as pp  # noqa: E402

FORBIDDEN = {"Report", "Labels", "Localizations", "LabelsLocalizationsBySentence", "labelCUIS",
             "LocalizationsCUIS", "MethodLabel", "ReportID", "ImageDir"}
PATIENT_COLS = ["PatientBirth", "PatientSex_DICOM", "ViewPosition_DICOM"]
ACQ_CATEGORICAL = ["Modality_DICOM", "Manufacturer_DICOM", "MethodProjection", "Pediatric",
                   "BitsStored_DICOM", "SpatialResolution_DICOM", "XRayTubeCurrent_DICOM"]
ACQ_NUMERIC = ["WindowCenter_DICOM", "WindowWidth_DICOM", "Rows_DICOM", "Columns_DICOM",
               "Exposure_DICOM", "ExposureInuAs_DICOM", "ExposureTime", "RelativeXRayExposure_DICOM"]
ACUITY = {"AP_horizontal": 2, "AP": 1}  # everything else (PA, L, COSTAL, EXCLUDE, NaN) -> 0
ACUITY_NAMES = ("PA/L", "AP", "AP_horizontal")


def read_meta(cols: list[str]) -> pd.DataFrame:
    """PadChest CSV restricted to `cols` (+ ImageID), indexed by ImageID, everything as str."""
    bad = FORBIDDEN & set(cols)
    assert not bad, f"forbidden PadChest columns requested: {sorted(bad)}"
    df = pd.read_csv(pp.PADCHEST, usecols=["ImageID", *cols], dtype=str, low_memory=False)
    assert not (FORBIDDEN & set(df.columns))
    return df.set_index("ImageID")


def load_folds_meta(classes: list[str], acq: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """pp.load_folds (sid / pid / date / proj / code) plus sex, birth year, age and acuity per image;
    with acq=True also the raw DICOM acquisition fields (EDA only)."""
    tr, va = pp.load_folds(classes)
    cols = PATIENT_COLS + (ACQ_CATEGORICAL + ACQ_NUMERIC if acq else [])
    meta = read_meta(cols)
    out = []
    for df in (tr, va):
        m = meta.reindex(df["filename"])  # fold filenames only
        df = df.copy()
        sex = m["PatientSex_DICOM"].to_numpy()
        df["sex"] = np.where(np.isin(sex, ["M", "F"]), sex, "U")
        df["birth"] = pd.to_numeric(m["PatientBirth"].to_numpy(), errors="coerce")
        df["age"] = (df["date"].dt.year - df["birth"]).clip(0, 110)
        df["acuity"] = df["proj"].map(ACUITY).fillna(0).astype(np.int64)
        df["view_pos"] = m["ViewPosition_DICOM"].to_numpy()
        if acq:
            for c in ACQ_CATEGORICAL + ACQ_NUMERIC:
                df[c] = m[c].to_numpy()
        out.append(df)
    return out[0], out[1]


def patient_sex_birth(images: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: modal known sex ('U' if none) and modal birth year."""
    known = images[images["sex"] != "U"]
    sex = known.groupby("pid")["sex"].agg(lambda s: s.mode().iloc[0])
    birth = images.dropna(subset=["birth"]).groupby("pid")["birth"].agg(lambda s: s.mode().iloc[0])
    pts = pd.DataFrame(index=pd.Index(images["pid"].unique(), name="pid"))
    pts["sex"] = sex.reindex(pts.index).fillna("U")
    pts["birth"] = birth.reindex(pts.index)
    return pts


def date_states_meta(sg: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    """pp.date_states(sg) (one state per (patient, date), OR of the day's study labels) plus, per
    state: sex, age at the state date, acuity (most acute projection among that day's fold images),
    rank within the patient, the index of the previous state (-1 if none) and the gap in days."""
    st = pp.date_states(sg).reset_index(drop=True)
    img = images[images["sid"].isin(sg.index)]
    acu = img.groupby(["pid", "date"])["acuity"].max()
    st["acuity"] = acu.reindex(pd.MultiIndex.from_frame(st[["pid", "date"]])).fillna(0).astype(np.int64).to_numpy()
    pts = patient_sex_birth(images)
    st["sex"] = pts["sex"].reindex(st["pid"]).to_numpy()
    st["age"] = (st["date"].dt.year - pts["birth"].reindex(st["pid"]).to_numpy()).clip(0, 110)
    pid = st["pid"].to_numpy()
    same = np.r_[False, pid[1:] == pid[:-1]]
    st["prev"] = np.where(same, np.arange(len(st)) - 1, -1)
    st["rank"] = st.groupby("pid").cumcount()
    gap = np.full(len(st), np.nan)
    d = st["date"].to_numpy()
    gap[same] = (d[1:][same[1:]] - d[:-1][same[1:]]) / np.timedelta64(1, "D")
    st["gap"] = gap
    return st


def state_index(states: pd.DataFrame, pid: np.ndarray, date: np.ndarray) -> np.ndarray:
    """Index of the (pid, date) state for each row, -1 if absent."""
    key = pd.MultiIndex.from_frame(states[["pid", "date"]])
    pos = pd.Series(np.arange(len(states)), index=key)
    q = pd.MultiIndex.from_arrays([pid, pd.to_datetime(date)])
    return pos.reindex(q).fillna(-1).astype(np.int64).to_numpy()


def prev_state_index(states: pd.DataFrame, pid: np.ndarray, date: np.ndarray) -> np.ndarray:
    """Index of the most recent strictly earlier state of the same patient, -1 if none
    (same rule as pp.lookup_prior)."""
    st = states.assign(sidx=np.arange(len(states)))[["pid", "date", "sidx"]].rename(columns={"date": "prior_date"})
    q = pd.DataFrame({"row": np.arange(len(pid)), "pid": pid, "date": pd.to_datetime(date)})
    ok = q["date"].notna()
    m = pd.merge_asof(q[ok].sort_values("date"), st.sort_values("prior_date"), left_on="date",
                      right_on="prior_date", by="pid", allow_exact_matches=False, direction="backward")
    out = np.full(len(pid), -1, np.int64)
    has = m["sidx"].notna().to_numpy()
    out[m["row"].to_numpy()[has]] = m["sidx"].to_numpy()[has].astype(np.int64)
    return out


# ---------------------------------------------------------------------- acquisition (DICOM) features
ACQ_FLAGS = (("Modality_DICOM", "DX"), ("Manufacturer_DICOM", "PhilipsMedicalSystems"),
             ("MethodProjection", "resnet-50.t7"), ("BitsStored_DICOM", "10"), ("Pediatric", "PED"))
ACQ_NUMERIC_ALL = ACQ_NUMERIC + ["XRayTubeCurrent_DICOM", "SpatialResolution_DICOM"]


def acquisition_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Raw (unstandardised) per-image acquisition features: 5 binary flags, then signed log1p of each
    numeric DICOM field with its missing indicator (missing -> median of `df`). Needs acq=True columns.
    These are care-setting / scanner proxies, not patient characteristics."""
    cols, names = [], []
    for c, v in ACQ_FLAGS:
        cols.append(df[c].eq(v).to_numpy(float))
        names.append(f"{c}={v}")
    for c in ACQ_NUMERIC_ALL:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        miss = np.isnan(x)
        x = np.sign(x) * np.log1p(np.abs(x))
        x[miss] = np.nanmedian(x) if (~miss).any() else 0.0
        cols += [x, miss.astype(float)]
        names += [c, c + "_missing"]
    return np.column_stack(cols), names


# ---------------------------------------------------------------------- features phi(m)
def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline (Harrell): x plus k-2 non-linear terms, linear beyond the end knots."""
    k = len(knots)
    t = knots
    scale = (t[-1] - t[0]) ** 2
    cols = [x]
    for j in range(k - 2):
        v = (np.clip(x - t[j], 0, None) ** 3
             - np.clip(x - t[k - 2], 0, None) ** 3 * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
             + np.clip(x - t[k - 1], 0, None) ** 3 * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]))
        cols.append(v / scale)
    return np.stack(cols, 1)


class Featurizer:
    """phi(m) = [sex M, sex unknown, age RCS (3), acuity AP, acuity AP_horizontal], standardised on
    the fitting states. The gap enters the transition separately as g = log1p(days)."""

    names = ["sex=M", "sex=U", "age", "age_rcs1", "age_rcs2", "acuity=AP", "acuity=AP_horizontal"]

    def __init__(self, knots=(5, 35, 65, 95)):
        self.knot_pct = knots

    def fit(self, sex, age, acuity) -> "Featurizer":
        a = np.asarray(age, float)
        self.age_fill = float(np.nanmedian(a))
        self.knots = np.percentile(a[~np.isnan(a)], self.knot_pct)
        raw = self._raw(sex, age, acuity)
        self.mu, self.sd = raw.mean(0), raw.std(0) + 1e-8
        return self

    def _raw(self, sex, age, acuity) -> np.ndarray:
        sex = np.asarray(sex)
        a = np.asarray(age, float)
        a = np.where(np.isnan(a), self.age_fill, a)
        acuity = np.asarray(acuity)
        return np.column_stack([sex == "M", sex == "U", rcs_basis(a, self.knots),
                                acuity == 1, acuity == 2]).astype(np.float64)

    def __call__(self, sex, age, acuity) -> np.ndarray:
        return (self._raw(sex, age, acuity) - self.mu) / self.sd

    def state_dict(self) -> dict:
        return {"knot_pct": self.knot_pct, "knots": self.knots, "age_fill": self.age_fill,
                "mu": self.mu, "sd": self.sd}

    @classmethod
    def from_state_dict(cls, d: dict) -> "Featurizer":
        f = cls(d["knot_pct"])
        f.knots, f.age_fill, f.mu, f.sd = d["knots"], d["age_fill"], d["mu"], d["sd"]
        return f


def gap_feature(days: np.ndarray) -> np.ndarray:
    return np.log1p(np.nan_to_num(np.asarray(days, float), nan=0.0))
