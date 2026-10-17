"""Ensemble-level view of the edge-case contrasts (2026-10-17). POST HOC, not in the pre-registered decision.

Averaging the probabilities of all seeds of an arm removes most of the training-seed noise, so ensemble(C) -
ensemble(A) on a subset is the cleanest estimate of the systematic difference the validation images allow. Same
subsets, metric and paired image bootstrap as analysis/entropy_edge_cases.py (B resamples shared by all matrices); no
seed-level test exists for one ensemble pair, so p is the image-bootstrap p only and Holm is applied to it within the
families F1 (S1-S10) and F2 (7 tail classes).

  768 px, area:    E(C, 5 seeds) - E(A, 5 seeds)           (E1 and A5 of analysis/entropy_best_config.py, area dumps)
  768 px, lanczos: the same with the lanczos dumps
  512 px:          E(C, 6 seeds) - E(A, 6 seeds), E(C) - E(B)   (ablation preds, no TTA, lanczos)

    python -I analysis/entropy_edge_cases_ens.py [--boot 2000] [--workers 8] -> analysis/out/entropy_edge_cases_ens.json
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
import entropy_edge_cases as eec  # noqa: E402

O = ec.O


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--out", type=Path, default=O / "entropy_edge_cases_ens.json")
    args = ap.parse_args()
    t0 = time.time()
    ctx = ec.load_context()
    L, names = ctx["L"], ctx["names"]
    tail_keys = [f"C:{names[i]}" for i in ctx["tail_sorted"]]
    allkeys = ["ALL"] + eec.F1 + tail_keys

    mats = {}
    for it, tag in (("area", "768a"), ("lanczos", "768l")):
        for arm in ("A", "C"):
            mats[(tag, arm)] = np.mean([eec.load768(arm, s, it, L) for s in eec.SEEDS768], 0)
    for arm in ("A", "B", "C"):
        mats[("512", arm)] = np.mean([eec.load512(arm, s, L) for s in eec.SEEDS512], 0)
    keys = list(mats)
    fa = [ec.FastAPM(mats[k], L) for k in keys]
    point = {k: ec.subset_values(f(ctx["mmat"]), ctx) for k, f in zip(keys, fa)}
    AP = ec.boot_replicates(fa, ctx["mmat"], args.boot, args.seed, args.workers)
    boot = {k: {m: np.array([ec.subset_values(AP[b, i], ctx)[m] for b in range(AP.shape[0])]) for m in allkeys}
            for i, k in enumerate(keys)}
    print(f"bootstrap done ({time.time() - t0:.0f}s)", flush=True)

    out = {"boot": args.boot, "contrasts": {}}
    for name, x, y in (("768 area C-A", ("768a", "C"), ("768a", "A")), ("768 lanczos C-A", ("768l", "C"), ("768l", "A")),
                       ("512 C-A", ("512", "C"), ("512", "A")), ("512 C-B", ("512", "C"), ("512", "B"))):
        res = {}
        for m in allkeys:
            bs = ec.boot_summary(boot[x][m] - boot[y][m])
            res[m] = {"y": float(point[y][m]), "x": float(point[x][m]), "diff": float(point[x][m] - point[y][m]),
                      "ci": bs["ci"], "p": bs["p"]}
        for fam, ks in (("F1", eec.F1), ("F2", tail_keys)):
            adj = ec.holm({k: res[k]["p"] for k in ks})
            for k in ks:
                res[k]["holm"] = adj[k]
        out["contrasts"][name] = res
    args.out.write_text(json.dumps(out, indent=1, default=float))

    lab = dict(eec.SUBS)
    lab["ALL"] = "all views (reference)"
    lab.update({f"C:{names[i]}": names[i] for i in ctx["tail_sorted"]})
    for name, res in out["contrasts"].items():
        print(f"\n### ensemble contrast {name}")
        print("| ID | subset | ensemble ASL (or B) | ensemble Entropy | diff | image 95% CI | p (image) | Holm |")
        print("|---|---|---|---|---|---|---|---|")
        for k in allkeys:
            r = res[k]
            print(f"| {k} | {lab[k]} | {r['y']:.4f} | {r['x']:.4f} | {r['diff']:+.4f} | [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}] | "
                  f"{r['p']:.3f} | {r.get('holm', float('nan')):.3f} |")
    print(f"\nwrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
