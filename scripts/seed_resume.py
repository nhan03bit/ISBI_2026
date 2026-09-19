"""
Rebuild a resumable `last.pth` from a run that only ever wrote `model_best.pth`.

Needed for jobs launched before per-epoch checkpointing existed (e.g. 41730): if
such a run is interrupted, this reconstructs enough state for train.py's
_try_resume() to continue from the best epoch instead of restarting at 1.

What carries over: weights, cosine-LR position, early-stopping counters, history.
What does not: AdamW moments (never saved) and RNG state. Adam re-warms within a
few hundred steps, so the practical cost is small.

Usage:
    python scripts/seed_resume.py                      # writes checkpoint/stage1/last.pth
    python scripts/seed_resume.py --out-dir DIR        # read from / write to DIR
    python scripts/seed_resume.py --write-to DIR       # read real dir, write elsewhere (test)
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import Trainer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="./checkpoint/stage1")
    ap.add_argument("--write-to", default=None, help="write last.pth here instead of --out-dir")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--force", action="store_true", help="overwrite an existing last.pth")
    args = ap.parse_args()

    src = args.out_dir
    dst = args.write_to or args.out_dir

    best_path = os.path.join(src, "model_best.pth")
    hist_path = os.path.join(src, "history.json")
    for p in (best_path, hist_path):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")

    existing = os.path.join(dst, "last.pth")
    if os.path.exists(existing) and not args.force:
        sys.exit(f"{existing} already exists - refusing to overwrite without --force")

    with open(hist_path) as f:
        history = json.load(f)

    best_epoch = history.get("best_epoch")
    if not best_epoch:
        sys.exit("history.json has no best_epoch")

    # Same cfg train.py uses, so the rebuilt checkpoint matches what it expects.
    cfg = {
        "num_classes": 30,
        "lr": 4e-5,
        "weight_decay": 1e-2,
        "epochs": args.epochs,
        "patience": 3,
        "min_delta": 0.0,
        "monitor": "mAP",
        "out_dir": dst,
        "resume": True,
    }

    tr = Trainer(cfg)
    tr.model.load_state_dict(torch.load(best_path, map_location="cpu"))

    # Advance the cosine schedule to where the completed epochs left it.
    for _ in range(best_epoch):
        tr.sched.step()

    # Truncate history to the best epoch: anything after it was not saved with
    # weights, so replaying those epochs' metrics would misreport the curve.
    for k in ("train_loss", "val_loss", "lr", "mAP", "mAUC", "mF1", "mECE"):
        if isinstance(history.get(k), list):
            history[k] = history[k][:best_epoch]
    tr.history = history

    # bad_epochs=0 is deliberately optimistic: the true count is unknowable from
    # model_best.pth alone, and resetting it grants at most `patience` extra
    # epochs rather than stopping a run that still had room to improve.
    tr.es.best = history.get("best_metric")
    tr.es.bad_epochs = 0

    os.makedirs(dst, exist_ok=True)
    tr._save_ckpt(dst, best_epoch)

    out = os.path.join(dst, "last.pth")
    print(f"wrote {out}")
    print(f"  resumes at epoch {best_epoch + 1}/{args.epochs}")
    print(f"  best {cfg['monitor']}={tr.es.best} (epoch {best_epoch})")
    print(f"  size {os.path.getsize(out) / 1e6:.0f} MB")
    print("  NOTE: optimizer moments and RNG state are not recoverable")


if __name__ == "__main__":
    main()
