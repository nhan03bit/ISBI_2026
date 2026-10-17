"""Post-hoc paired image bootstrap between the top ensembles of analysis/entropy_best_config.py (2026-10-17).
NOT part of the pre-registered ranking: it adds the intervals the ranking itself does not give (E5 vs E3, E5 vs E4,
E3 vs E4, and each of them against the fair same-recipe control A5 and E1).

    python -I analysis/entropy_best_config_pairs.py [--boot 2000] [--workers 8] -> analysis/out/entropy_best_config_pairs.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edge_common as ec  # noqa: E402
import entropy_best_config as ebc  # noqa: E402

O = ec.O
PAIRS = [("E5", "E3"), ("E5", "E4"), ("E3", "E4"), ("E5", "E1"), ("E3", "E1"), ("E4", "E1"),
         ("E5", "A5"), ("E3", "A5"), ("E4", "A5")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--out", type=Path, default=O / "entropy_best_config_pairs.json")
    args = ap.parse_args()
    t0 = time.time()
    ctx = ec.load_context()
    L, tail = ctx["L"], ctx["groups"]["tail"]
    P = {k: np.load(O / ebc.SINGLES[k][0]).astype(np.float64) for k in {m for e in ebc.ENS.values() for m in e}}
    for k, members in ebc.ENS.items():
        P[k] = np.mean([P[m] for m in members], 0)
    P["A5"] = np.mean([np.load(O / ebc.ASL[f"A{s}"]).astype(np.float64) for s in ebc.SEEDS], 0)
    names = ["E1", "E3", "E4", "E5", "A5"]
    cols = ["all", "frontal", "frontal_unseen"]
    m3 = np.stack([ctx["masks"][c] for c in cols], 1).astype(np.float64)
    fa = [ec.FastAPM(P[k], L) for k in names]
    pt = {k: f(m3) for k, f in zip(names, fa)}
    AP = ec.boot_replicates(fa, m3, args.boot, args.seed, args.workers)
    ix = {k: i for i, k in enumerate(names)}
    bm = {"mAP": np.nanmean(AP[:, :, 0, :], -1), "frontal": np.nanmean(AP[:, :, 1, :], -1),
          "frontal_unseen": np.nanmean(AP[:, :, 2, :], -1), "tail": np.nanmean(AP[:, :, 0, :][..., tail], -1)}
    point = {k: {"mAP": float(np.nanmean(a[0])), "frontal": float(np.nanmean(a[1])),
                 "frontal_unseen": float(np.nanmean(a[2])), "tail": float(np.nanmean(a[0][tail]))} for k, a in pt.items()}
    out = {"boot": args.boot, "point": point, "pairs": {}}
    for x, y in PAIRS:
        out["pairs"][f"{x}-{y}"] = {}
        for met in bm:
            bs = ec.boot_summary(bm[met][:, ix[x]] - bm[met][:, ix[y]])
            out["pairs"][f"{x}-{y}"][met] = {"point": point[x][met] - point[y][met], "ci": bs["ci"], "p": bs["p"]}
    args.out.write_text(json.dumps(out, indent=1))
    print("| pair | all-view mAP | frontal mAP | frontal & unseen mAP | tail AP |")
    print("|---|---|---|---|---|")
    for k, v in out["pairs"].items():
        print(f"| {k} | " + " | ".join(f"{v[m]['point']:+.4f} [{v[m]['ci'][0]:+.4f}, {v[m]['ci'][1]:+.4f}]"
                                       for m in ("mAP", "frontal", "frontal_unseen", "tail")) + " |")
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
