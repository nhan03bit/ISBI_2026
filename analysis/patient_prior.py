"""Patient-history Markov chain and prior lookup (2026-09-30).

Shared by analysis/patient_markov_check.py (offline HMM late fusion, "B-late") and, via the
priors file this module writes, by train/train_2_v5.py (label-query conditioning, "B-early").
Both arms therefore see exactly the same prior vector.

Rules (see docs/plan/2026-09-30-patient-markov-research.md):
  * internal train/val folds only -- never join dev/test/leaderboard images to the PadChest CSV
    (it also carries the report-derived labels of the evaluation images);
  * read only ImageID / StudyID / PatientID / StudyDate_DICOM / Projection from it
    (never Labels, Report, PatientBirth or sex);
  * labels come only from the 30-class label_list of data/{train,val}_fold_1.csv (shard order);
  * TRAIN priors use train-fold history only (no val labels ever reach a training input);
    VAL priors use train + val history (clinical simulation, as in the research study).

Prior vector per image, 31 values: r_c = logit pi_c(prior) - logit prevalence_c (c = 1..30, 0 if
the image has no strictly earlier study) plus a has_prior flag. pi_c = a_c if the prior state has
c else b_c, from a per-class 2-state chain fitted on consecutive train-fold (patient, date) states.

Usage
-----
    python analysis/patient_prior.py   # -> analysis/out/patient_markov/priors.pt
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluate"))
from class_groups import load_train_prevalence  # noqa: E402

LABEL_INFO = "/data/psytp7/wds_shards_train_raw/label_info.pt"  # shard/logit class order
PADCHEST = next(p for p in (ROOT / "data/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
                            Path("/data/psytp7/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"))
                if p.exists())
META_COLS = ["ImageID", "StudyID", "PatientID", "StudyDate_DICOM", "Projection"]
PRIORS = ROOT / "analysis/out/patient_markov/priors.pt"
BITS = np.int64(1) << np.arange(30, dtype=np.int64)


def label_matrix(df: pd.DataFrame, classes: list[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(df), len(classes)), np.uint8)
    for r, labels in enumerate(df["label_list"]):
        for c in ast.literal_eval(labels):
            Y[r, idx[c]] = 1
    return Y


def decode(codes: np.ndarray) -> np.ndarray:
    return ((codes[:, None] >> np.arange(30)) & 1).astype(np.uint8)


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float64), 1e-12, 1 - 1e-12)
    return np.log(x) - np.log1p(-x)


def load_folds(classes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and val fold rows with StudyID / PatientID / date / projection and a label bit-code."""
    meta = pd.read_csv(PADCHEST, usecols=META_COLS, dtype=str, low_memory=False).set_index("ImageID")
    folds = {}
    for k in ("train", "val"):
        df = pd.read_csv(ROOT / f"data/{k}_fold_1.csv", usecols=["filename", "label_list"])
        m = meta.reindex(df["filename"])
        df["sid"] = m["StudyID"].values
        df["pid"] = m["PatientID"].values
        df["date"] = pd.to_datetime(m["StudyDate_DICOM"].values, format="%Y%m%d", errors="coerce")
        df["proj"] = m["Projection"].values
        df["code"] = label_matrix(df, classes).astype(np.int64) @ BITS
        folds[k] = df
    tr, va = folds["train"], folds["val"]
    assert tr["pid"].notna().all() and va["pid"].notna().all(), "fold image missing from PadChest CSV"
    return tr, va


def study_table(tr: pd.DataFrame, va: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per StudyID; in_tr = the study has an image in the train fold."""
    st = pd.concat([tr.assign(in_tr=True), va.assign(in_tr=False)])
    st = st.dropna(subset=["date"])
    sg = st.groupby("sid").agg(pid=("pid", "first"), n_pid=("pid", "nunique"), date=("date", "first"),
                               n_date=("date", "nunique"), code=("code", "first"), n_code=("code", "nunique"),
                               in_tr=("in_tr", "any"))
    ambiguous = (sg["n_pid"] > 1) | (sg["n_date"] > 1)  # PadChest metadata errors: excluded from the chain
    stats = {"n_inconsistent": int((sg["n_code"] > 1).sum()), "n_ambiguous": int(ambiguous.sum()),
             "n_undated": int(len(tr) + len(va) - len(st))}
    return sg[~ambiguous], stats


def date_states(studies: pd.DataFrame) -> pd.DataFrame:
    """One state per (patient, date): OR of the label sets of that day's studies."""
    return (studies.groupby(["pid", "date"])
            .agg(code=("code", lambda c: np.bitwise_or.reduce(c.to_numpy())), any_tr=("in_tr", "any"),
                 n_studies=("code", "size"))
            .reset_index().sort_values(["pid", "date"]))


def fit_chain(ds_tr: pd.DataFrame) -> dict:
    """Per-class 2-state chain on consecutive (patient, date) states, Laplace-smoothed."""
    same_pid = ds_tr["pid"].to_numpy()[1:] == ds_tr["pid"].to_numpy()[:-1]
    prev_c = decode(ds_tr["code"].to_numpy()[:-1][same_pid])
    next_c = decode(ds_tr["code"].to_numpy()[1:][same_pid])
    gap = (ds_tr["date"].to_numpy()[1:][same_pid] - ds_tr["date"].to_numpy()[:-1][same_pid]) / np.timedelta64(1, "D")
    n1 = prev_c.sum(0)
    n0 = len(prev_c) - n1
    a = ((prev_c & next_c).sum(0) + 1) / (n1 + 2)
    b = (((1 - prev_c) & next_c).sum(0) + 1) / (n0 + 2)
    assert ((a > 0) & (a < 1) & (b > 0) & (b < 1)).all()
    return {"prev_c": prev_c, "next_c": next_c, "gap": gap, "n1": n1, "a": a, "b": b,
            "n_patients": len(set(ds_tr["pid"].to_numpy()[1:][same_pid]))}


def lookup_prior(pid: np.ndarray, date: np.ndarray, history: pd.DataFrame) -> pd.DataFrame:
    """Most recent strictly earlier (patient, date) state in `history` for each target row.

    Returns a frame in target-row order with prior_date / prior_code / any_tr (NaN when none).
    """
    rows = np.arange(len(pid))
    q = pd.DataFrame({"row": rows, "pid": pid, "date": date})
    undated = q["date"].isna()
    pri = history.rename(columns={"date": "prior_date", "code": "prior_code"}).sort_values("prior_date")
    q = pd.merge_asof(q[~undated].sort_values("date"), pri[["pid", "prior_date", "prior_code", "any_tr"]],
                      left_on="date", right_on="prior_date", by="pid", allow_exact_matches=False,
                      direction="backward")
    q = pd.concat([q, pd.DataFrame({"row": rows[undated.to_numpy()]})]).sort_values("row").reset_index(drop=True)
    assert (q["row"].to_numpy() == rows).all()
    has = q["prior_date"].notna().to_numpy()
    assert (q.loc[has, "prior_date"] < q.loc[has, "date"]).all(), "a prior is not strictly earlier"
    q.attrs["n_undated"] = int(undated.sum())
    return q


def prior_bits(q: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    has = q["prior_date"].notna().to_numpy()
    prior = np.zeros((len(q), 30), np.uint8)
    prior[has] = decode(q.loc[has, "prior_code"].astype(np.int64).to_numpy())
    return prior, has


def prior_log_ratio(prior: np.ndarray, has: np.ndarray, a: np.ndarray, b: np.ndarray,
                    prev: np.ndarray) -> np.ndarray:
    """r_c = logit pi_c(prior) - logit prevalence_c for rows with a prior, 0 otherwise."""
    pi_hist = np.where(prior == 1, a, b)
    log_ratio = np.zeros(prior.shape)
    log_ratio[has] = logit(pi_hist[has]) - logit(np.broadcast_to(prev, (int(has.sum()), 30)))
    return log_ratio


def build_priors() -> dict:
    classes = list(torch.load(LABEL_INFO, weights_only=False)["class_names"])
    _, prev, _ = load_train_prevalence(classes)
    tr, va = load_folds(classes)
    sg, stats = study_table(tr, va)
    chain = fit_chain(date_states(sg[sg["in_tr"]]))
    a, b = chain["a"], chain["b"]
    hist = {"train": date_states(sg[sg["in_tr"]]),  # train inputs never see val labels
            "val": date_states(sg)}                   # val: full history (clinical simulation)
    out = {"class_names": classes, "a": a, "b": b, "prevalence": prev, "chain_pairs": len(chain["prev_c"]),
           **stats}
    for split, df in (("train", tr), ("val", va)):
        q = lookup_prior(df["pid"].to_numpy(), df["date"].to_numpy(), hist[split])
        prior, has = prior_bits(q)
        r = np.concatenate([prior_log_ratio(prior, has, a, b, prev), has[:, None].astype(np.float64)], 1)
        out[split] = {"fnames": df["filename"].tolist(), "r": r.astype(np.float32), "has": has}
        print(f"{split:5s}: images {len(df)} | with a strictly earlier study {int(has.sum())} ({has.mean():.3f}) | "
              f"history = {'train fold only' if split == 'train' else 'train + val'} | "
              f"|r| max {np.abs(r[:, :30]).max():.2f}")
    return out


if __name__ == "__main__":
    PRIORS.parent.mkdir(parents=True, exist_ok=True)
    pri = build_priors()
    torch.save(pri, PRIORS)
    print(f"chain pairs {pri['chain_pairs']} | wrote {PRIORS}")
