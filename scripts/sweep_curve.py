#!/usr/bin/env python3
"""
Read the accum=8 / ep30 convergence sweep logs and report:
  1. per-epoch mAP (+ mF1/mECE) for every seed, as a table
  2. retrospective early-stopping: for each patience P, which epoch it would
     stop at and what monitored metric it would select, per seed and on average

Usage:
    python scripts/sweep_curve.py                      # default 5 seeds, monitor mAP
    python scripts/sweep_curve.py --monitor mF1
    python scripts/sweep_curve.py --logdir /home/psytp7/logs --epochs-tag ep30
"""
import argparse
import glob
import re
import os

EPOCH_RE = re.compile(
    r"Epoch (\d+)/\d+ \| train_loss=([\d.]+) val_loss=([\d.]+) \| "
    r"mAP=([\d.]+) mAUC=([\d.]+) mF1=([\d.]+) mECE=([\d.]+)"
)


def load_curve(path):
    rows = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                ep = int(m.group(1))
                rows[ep] = dict(
                    train_loss=float(m.group(2)), val_loss=float(m.group(3)),
                    mAP=float(m.group(4)), mAUC=float(m.group(5)),
                    mF1=float(m.group(6)), mECE=float(m.group(7)),
                )
    return [rows[e] for e in sorted(rows)]


def stop_epoch(vals, patience, mode):
    """Replay EarlyStopping(patience, mode). Return (best_epoch, best_value, stopped_epoch)."""
    best = None
    best_ep = 0
    bad = 0
    for i, v in enumerate(vals, start=1):
        better = best is None or (v > best if mode == "max" else v < best)
        if better:
            best, best_ep, bad = v, i, 0
        else:
            bad += 1
            if bad >= patience:
                return best_ep, best, i
    return best_ep, best, len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="/home/psytp7/logs")
    ap.add_argument("--epochs-tag", default="ep30")
    ap.add_argument("--accum", default="8")
    ap.add_argument("--seeds", nargs="*", default=["42", "86", "123", "456", "789"])
    ap.add_argument("--monitor", default="mAP",
                    choices=["mAP", "mAUC", "mF1", "mECE", "val_loss"])
    args = ap.parse_args()
    mode = "min" if args.monitor in ("mECE", "val_loss") else "max"

    curves = {}
    for s in args.seeds:
        pat = f"{args.logdir}/isbi2026_train2v2_accum{args.accum}_seed{s}_{args.epochs_tag}_*.out"
        hits = sorted(glob.glob(pat))
        if not hits:
            print(f"seed {s}: no log ({pat})")
            continue
        curves[s] = (hits[-1], load_curve(hits[-1]))

    if not curves:
        print("no logs found yet")
        return

    maxep = max((len(c) for _, c in curves.values()), default=0)
    if maxep == 0:
        print("logs exist but no epoch has completed yet.")
        for s, (f, _) in curves.items():
            print(f"  seed {s}: {os.path.basename(f)}")
        return

    # --- per-epoch table ---
    print(f"\n{args.monitor} per epoch (seed -> epoch):")
    hdr = "  ep  " + "".join(f"{'s'+s:>9}" for s in curves)
    print(hdr)
    for ep in range(1, maxep + 1):
        cells = []
        for s in curves:
            c = curves[s][1]
            cells.append(f"{c[ep-1][args.monitor]:9.4f}" if ep <= len(c) else f"{'--':>9}")
        print(f"  {ep:2d}  " + "".join(cells))

    # --- retrospective patience ---
    print(f"\nretrospective early-stopping (monitor={args.monitor}, mode={mode}):")
    print("  P  " + "".join(f"{'s'+s:>16}" for s in curves) + f"{'mean_sel':>12}{'note':>8}")
    for P in range(1, 10):
        sels, parts, incomplete = [], [], False
        for s in curves:
            c = [r[args.monitor] for r in curves[s][1]]
            be, bv, se = stop_epoch(c, P, mode)
            if se == len(c) and len(c) < maxep:
                incomplete = True
            parts.append(f"ep{be}:{bv:.4f}(stop{se})".rjust(16))
            sels.append(bv)
        mean_sel = sum(sels) / len(sels)
        note = "partial" if incomplete else ""
        print(f"  {P}  " + "".join(parts) + f"{mean_sel:12.4f}{note:>8}")

    print("\nnote: 'partial' = at least one seed has not run enough epochs for that "
          "patience to trigger; its value is the best-so-far and may still change.")


if __name__ == "__main__":
    main()
