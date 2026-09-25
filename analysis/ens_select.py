"""Greedy forward ensemble selection over dumped per-model val probabilities.

Consumes the ``*.npy`` files written by ``evaluate/evaluate_tta.py --dump-probs DIR``
(one [N,C] float32 array per checkpoint + ``labels.npy`` [N,C] uint8; all rows in
the same val-shard order). Scores every single model, then greedily adds the
model that most improves macro mAP of the uniform probability average, stopping
when no candidate improves by more than ``--min-gain``. Uniform averaging only
(no fitted weights) to limit overfitting to the internal val set, which is also
the early-stopping set - the resulting mAP is optimistically biased.

Usage
-----
    python analysis/ens_select.py analysis/out/probs --out analysis/out/ens_select.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchmetrics.functional.classification import multilabel_average_precision


def macro_map(probs: np.ndarray, labels: torch.Tensor) -> float:
    return float(multilabel_average_precision(
        torch.from_numpy(probs), labels, num_labels=labels.shape[1], average="macro"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump_dir")
    ap.add_argument("--min-gain", type=float, default=1e-4)
    ap.add_argument("--max-models", type=int, default=10)
    ap.add_argument("--exclude", nargs="*", default=[], help="substrings of file names to skip")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = Path(args.dump_dir)
    labels = torch.from_numpy(np.load(d / "labels.npy")).int()
    files = sorted(p for p in d.glob("*.npy")
                   if p.name != "labels.npy" and not any(s in p.name for s in args.exclude))
    probs = {p.name: np.load(p) for p in files}
    for k, v in probs.items():
        assert v.shape == tuple(labels.shape), f"{k}: {v.shape} vs labels {tuple(labels.shape)}"

    single = {k: macro_map(v, labels) for k, v in probs.items()}
    print("Single models:")
    for k, m in sorted(single.items(), key=lambda kv: -kv[1]):
        print(f"  {m:.4f}  {k}")

    chosen, cur, acc = [], -1.0, None
    while len(chosen) < args.max_models:
        best = None
        for k, v in probs.items():
            if k in chosen:
                continue
            cand = v if acc is None else (acc * len(chosen) + v) / (len(chosen) + 1)
            m = macro_map(cand, labels)
            if best is None or m > best[1]:
                best = (k, m, cand)
        if best is None or best[1] - cur <= args.min_gain:
            break
        chosen.append(best[0]); cur, acc = best[1], best[2]
        print(f"+ {best[0]}  -> ensemble mAP {cur:.4f} ({len(chosen)} models)")

    res = {"single": single, "chosen": chosen, "ensemble_mAP": cur, "n_val": int(labels.shape[0])}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
