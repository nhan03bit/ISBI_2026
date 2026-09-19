"""Canonical class metadata for the CXR-LT 2026 Task-1 30-class problem.

Two orderings exist in this repo and they are NOT the same:

  * "shard order"  -- ``MultiLabelBinarizer.classes_``, i.e. ``sorted(names)``.
    This is the order of the per-sample label tensors stored in the webdataset
    shards (``wds_shards_*/label_info.pt`` key ``class_names``) and therefore the
    order of every model's 30 output logits.
  * "csv order"    -- the column order of
    ``train/CXRLT_2026_training_filtered.csv`` (roughly frequency-descending).
    ``train/train_2_v2.py`` computes ``class_weights`` in this order.

Everything downstream (per-class AP, head/medium/tail grouping, LaTeX tables)
must be expressed in SHARD ORDER. This module is the single source of truth for
that mapping and for the frequency-group split.

Run directly to print the grouping table:

    python evaluate/class_groups.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

DEFAULT_LABEL_INFO = _REPO / "wds_shards_val_raw" / "label_info.pt"
DEFAULT_TRAIN_CSV = _REPO / "train" / "CXRLT_2026_training_filtered.csv"

# CXR-LT 2024 (Med. Image Anal. 2025) frequency-group thresholds, on *training*
# prevalence. head > 10%, 1% <= medium <= 10%, tail < 1%.
HEAD_MIN = 0.10
TAIL_MAX = 0.01
# "extreme tail" -- the sub-group where a long-tail method has to earn its claim.
EXTREME_TAIL_MAX = 0.005


def load_class_names(label_info_path: os.PathLike | str = DEFAULT_LABEL_INFO) -> list[str]:
    """The 30 class names in SHARD / logit order."""
    import torch

    info = torch.load(str(label_info_path), weights_only=False)
    names = list(info["class_names"])
    if len(names) != 30:
        raise ValueError(f"expected 30 class names, got {len(names)}")
    return names


def load_train_prevalence(
    class_names: list[str],
    train_csv: os.PathLike | str = DEFAULT_TRAIN_CSV,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (positive_counts, prevalence, N) aligned to ``class_names`` order."""
    import pandas as pd

    df = pd.read_csv(train_csv)
    label_cols = list(df.columns[3:])
    missing = set(class_names) - set(label_cols)
    if missing:
        raise ValueError(f"class names absent from {train_csv}: {sorted(missing)}")
    mat = df[list(class_names)].to_numpy(dtype=np.float64)  # reindex to shard order
    counts = mat.sum(axis=0)
    n = len(df)
    return counts.astype(np.int64), counts / n, n


def frequency_groups(prevalence: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean index masks (shard order) for head / medium / tail / extreme_tail."""
    head = prevalence > HEAD_MIN
    tail = prevalence < TAIL_MAX
    medium = ~head & ~tail
    extreme_tail = prevalence < EXTREME_TAIL_MAX
    return {"head": head, "medium": medium, "tail": tail, "extreme_tail": extreme_tail}


def build(
    label_info_path: os.PathLike | str = DEFAULT_LABEL_INFO,
    train_csv: os.PathLike | str = DEFAULT_TRAIN_CSV,
) -> dict:
    names = load_class_names(label_info_path)
    counts, prev, n = load_train_prevalence(names, train_csv)
    groups = frequency_groups(prev)
    return {
        "class_names": names,
        "train_counts": counts,
        "train_prevalence": prev,
        "train_n": n,
        "groups": groups,
    }


def group_of(prevalence_value: float) -> str:
    if prevalence_value > HEAD_MIN:
        return "head"
    if prevalence_value < TAIL_MAX:
        return "tail"
    return "medium"


def _main() -> None:
    meta = build()
    names = meta["class_names"]
    prev = meta["train_prevalence"]
    counts = meta["train_counts"]
    order = np.argsort(prev)  # ascending, rarest first
    print(f"{'idx':>3}  {'class':<32} {'train_pos':>9} {'prev':>8}  group")
    print("-" * 70)
    for i in order:
        print(f"{i:>3}  {names[i]:<32} {counts[i]:>9d} {prev[i]:>8.4f}  {group_of(prev[i])}")
    print("-" * 70)
    for g in ("head", "medium", "tail", "extreme_tail"):
        idx = np.where(meta["groups"][g])[0]
        print(f"{g:>12}: {len(idx):2d} classes  ->  {[names[j] for j in idx]}")
    print(f"\ntrain N = {meta['train_n']}")


if __name__ == "__main__":
    _main()
