#!/usr/bin/env python3
"""Compile per-epoch metrics for the Wave-1 Stage-2 sweep (Tracks A + B).

Reads the Slurm .out logs directly (works while jobs are still running) and, when
present, the history.json best-epoch summary. No args.
"""
import glob
import json
import os
import re

LOGDIR = "/home/psytp7/logs"
EPOCH_RE = re.compile(
    r"Epoch (\d+)/\d+ \| train_loss=([\d.]+) val_loss=([\d.]+) \| "
    r"mAP=([\d.]+) mAUC=([\d.]+) mF1=([\d.]+) mECE=([\d.]+)"
)
STATE_RE = re.compile(r"=== Finished")


def curve(path):
    rows = {}
    finished = False
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.replace("\r", "\n")
            for m in EPOCH_RE.finditer(line):
                e = int(m.group(1))
                rows[e] = tuple(float(x) for x in m.groups()[1:])
            if STATE_RE.search(line):
                finished = True
    return rows, finished


def show(title, patterns, cols):
    print(f"\n### {title}\n")
    hdr = f"{'job / config':<42} {'ep':>3} " + " ".join(f"{c:>8}" for c in
        ["train", "val", "mAP", "mAUC", "mF1", "mECE"])
    print(hdr)
    print("-" * len(hdr))
    for pat in patterns:
        for path in sorted(glob.glob(os.path.join(LOGDIR, pat))):
            if "smoke" in os.path.basename(path):
                continue
            name = os.path.basename(path).replace("isbi2026_", "").replace(".out", "")
            name = re.sub(r"_\d{5}$", "", name)
            rows, finished = curve(path)
            if not rows:
                print(f"{name:<42} {'--':>3}  (no epoch logged yet)")
                continue
            best_e = max(rows, key=lambda e: rows[e][2])   # by mAP
            for e in sorted(rows):
                tr, va, ap, auc, f1, ece = rows[e]
                star = " *" if e == best_e else ""
                tag = name if e == min(rows) else ""
                print(f"{tag:<42} {e:>3} {tr:>8.4f} {va:>8.4f} {ap:>8.4f} "
                      f"{auc:>8.4f} {f1:>8.4f} {ece:>8.4f}{star}")
            flag = "DONE" if finished else "running"
            print(f"{'':<42}     -> best mAP {rows[best_e][2]:.4f} @ep{best_e}  [{flag}]")
    print()


show("Track A - train_2_v3 (new tricks: query-embed / XBM / head-excl / EMA)",
     ["isbi2026_v3_*.out"], None)
show("Track B - pure train_2_v2 hyperparameter sweep",
     ["isbi2026_v2pure_*.out"], None)

print("baseline references: prior Stage-2 plateau mAP ~0.41 (5-seed 0.414±0.002, "
      "mECE 0.143); tuned job 45702 = 0.4043; Stage-1 ckpt 0.385.")
