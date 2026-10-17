"""Post-hoc optimism check of the old greedy-3 ensemble E5 (2026-10-17). NOT part of the pre-registered ranking.

E5 = push 768 s42 SWA + push 1024 s86 SWA + res_sweep 640 s86 best was picked greedily (analysis/ens_select.py) on the
17,270 internal-val images from the 10 flip-TTA dumps of analysis/out/probs_w01_flip, i.e. on the same images it is
scored on. A split-half check by patient cannot undo that (E5's members already saw both halves). So the selection
itself is repeated inside each split: greedy forward selection on the all-view mAP of half 1 only (same procedure,
min gain 1e-4), then the chosen ensemble is scored on half 2 and compared with fixed ensembles on half 2.

    python -I analysis/entropy_best_config_nested.py [--splits 100]  -> analysis/out/entropy_best_config_nested.json
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
import entropy_best_config as ebc  # noqa: E402  (candidate tables)

O = ec.O
POOL = ["P768-42", "P768-1024", "P896-86", "P1024-86", "R640-86", "R640-1024", "R768-86", "W512-42", "W512-1024", "D512-86"]
MIN_GAIN, MAX_MODELS = 1e-4, 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--out", type=Path, default=O / "entropy_best_config_nested.json")
    args = ap.parse_args()
    t0 = time.time()
    ctx = ec.load_context()
    L = ctx["L"]
    P = {k: np.load(O / ebc.SINGLES[k][0]).astype(np.float64) for k in set(POOL) | {m for e in ebc.ENS.values() for m in e}}
    for k, members in ebc.ENS.items():
        P[k] = np.mean([P[m] for m in members], 0)
    P["A5"] = np.mean([np.load(O / ebc.ASL[f"A{s}"]).astype(np.float64) for s in ebc.SEEDS], 0)
    fixed = ["E1", "E3", "E4", "E5", "A5"]
    fa = {k: ec.FastAPM(P[k], L) for k in fixed}
    codes, npat, fr = ctx["pid_codes"], ctx["n_patients"], ctx["masks"]["frontal"]

    def two_halves(probs: np.ndarray, h1: np.ndarray) -> np.ndarray:
        """(4,) = all-view mAP half 1, half 2, frontal mAP half 1, half 2."""
        W = np.stack([h1, ~h1, fr & h1, fr & ~h1], 1).astype(np.float64)
        return np.nanmean(ec.FastAPM(probs, L)(W), 1)

    rows = []
    for r in range(args.splits):
        rng = np.random.default_rng([args.seed + 11, r])
        h1 = (rng.random(npat) < 0.5)[codes]
        W4 = np.stack([h1, ~h1, fr & h1, fr & ~h1], 1).astype(np.float64)
        # greedy forward selection on half-1 all-view mAP
        chosen, acc, cur = [], None, -1.0
        while len(chosen) < MAX_MODELS:
            best = None
            for k in POOL:
                if k in chosen:
                    continue
                cand = P[k] if acc is None else (acc * len(chosen) + P[k]) / (len(chosen) + 1)
                m = float(np.nanmean(ec.FastAPM(cand, L)(W4[:, :1])[0]))
                if best is None or m > best[1]:
                    best = (k, m, cand)
            if best is None or best[1] - cur <= MIN_GAIN:
                break
            chosen.append(best[0])
            cur, acc = best[1], best[2]
        g = np.nanmean(ec.FastAPM(acc, L)(W4), 1)           # nested greedy: [all h1, all h2, frontal h1, frontal h2]
        ref = {k: np.nanmean(fa[k](W4), 1) for k in fixed}
        rows.append({"chosen": chosen, "greedy": g.tolist(), **{k: v.tolist() for k, v in ref.items()}})
        if (r + 1) % 20 == 0:
            print(f"split {r + 1}/{args.splits} ({time.time() - t0:.0f}s)", flush=True)

    def stat(f) -> dict:
        v = np.array([f(r) for r in rows])
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "share_positive": float((v > 0).mean())}

    out = {"splits": args.splits, "pool": POOL, "rule": f"greedy forward, all-view mAP on half 1, min gain {MIN_GAIN}",
           "chosen_counts": {}, "sizes": [len(r["chosen"]) for r in rows]}
    for r in rows:
        key = "+".join(sorted(r["chosen"]))
        out["chosen_counts"][key] = out["chosen_counts"].get(key, 0) + 1
    H2A, H2F = 1, 3                                         # held-out half: all-view, frontal
    for ref in fixed:
        out[f"nested_greedy_minus_{ref}_heldout_all"] = stat(lambda r, ref=ref: r["greedy"][H2A] - r[ref][H2A])
        out[f"nested_greedy_minus_{ref}_heldout_frontal"] = stat(lambda r, ref=ref: r["greedy"][H2F] - r[ref][H2F])
    out["E5_minus_E1_heldout_all_contaminated"] = stat(lambda r: r["E5"][H2A] - r["E1"][H2A])
    out["E5_minus_E1_heldout_frontal_contaminated"] = stat(lambda r: r["E5"][H2F] - r["E1"][H2F])
    out["greedy_selection_half_minus_heldout_all"] = stat(lambda r: r["greedy"][0] - r["greedy"][1])
    out["E5_E1_gap_selection_vs_heldout_halves_all"] = stat(lambda r: (r["E5"][0] - r["E1"][0]) - (r["E5"][1] - r["E1"][1]))
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "chosen_counts"}, indent=1))
    print("chosen sets:", json.dumps(out["chosen_counts"], indent=1))
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
