"""Route 1 replication over the 5 Table 1 Entropy seeds (2026-10-17, CPU).

Reads the analysis/memory_readout_eval.py output for the tags entropy_c_s<S> (probes 53219-53223) and applies the
rule pre-registered in docs/plan/2026-10-16-beat-asl-plan.md section 2c before any of these probes finished:
  * primary: binary-entropy-gated blend - classifier alone, all-view mAP, no TTA. Holds only if the mean over the
    5 seeds has a 95% t-interval over seeds above 0 AND at least 4 of 5 seeds are positive;
  * the gate counts only if, in at least 4 of 5 seeds, binary beats both the random gate and the constant blend;
  * reported, not decisive: frontal and tail deltas, the configs each fold chose.
Exploratory (labelled so): each seed's fold-chosen binary config applied to its flip-TTA area probabilities
(analysis/out/probs_unified_flip_area), against the classifier alone and against ASL's flip-TTA area run of the
same seed index (analysis/out/probs_v0_flip_area, the Table 1 pairing). ASL gets no readout here.

    python -I analysis/memory_readout_eval.py --tags entropy_c_s42 entropy_c_s123 entropy_c_s456 \
        entropy_c_s789 entropy_c_s1024 --out analysis/out/memory_readout_eval_5seed.json
    python -I analysis/memory_readout_seeds.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entropy_ablation as ea  # noqa: E402  (FastAP)
import memory_readout_eval as mre  # noqa: E402  (knn_readout, binary_entropy, the same PadChest columns)

ROOT = ea.ROOT
O = ROOT / "analysis/out"
SEEDS = (42, 123, 456, 789, 1024)


def seed_stats(d: np.ndarray) -> dict:
    """Mean, 95% t-interval over seeds, wins and the exact two-sided sign-flip p of per-seed differences."""
    n = len(d)
    half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    flips = np.array([np.mean(d * s) for s in itertools.product((1, -1), repeat=n)])
    return {"diffs": d.tolist(), "mean": float(d.mean()), "ci": [float(d.mean() - half), float(d.mean() + half)],
            "wins": int((d > 0).sum()), "n": n, "p_signflip": float(np.mean(np.abs(flips) >= abs(d.mean()) - 1e-12))}


def fmt(s: dict) -> str:
    return (f"{s['mean']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] wins {s['wins']}/{s['n']} "
            f"p {s['p_signflip']:.3f} | per seed {' '.join(f'{x:+.4f}' for x in s['diffs'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", type=Path, default=O / "memory_readout_eval_5seed.json")
    ap.add_argument("--root", default="/data/psytp7/memprobe")
    ap.add_argument("--out", type=Path, default=O / "memory_readout_seeds.json")
    args = ap.parse_args()

    ev = json.loads(args.eval_json.read_text())["tags"]
    tags = [f"entropy_c_s{s}" for s in SEEDS]
    missing = [t for t in tags if t not in ev]
    if missing:
        raise SystemExit(f"{args.eval_json} lacks {missing}")
    fam = {f: np.array([ev[t]["families"][f]["mAP"] for t in tags]) for f in mre.FAMILIES}
    cls = np.array([ev[t]["cls"]["mAP"] for t in tags])
    res = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "eval_json": str(args.eval_json),
           "cls_mAP": cls.tolist(), "families": {}, "chosen_binary": {t: ev[t]["families"]["binary"]["chosen"] for t in tags}}
    print(f"classifier alone (no TTA, area): {' '.join(f'{x:.4f}' for x in cls)} | mean {cls.mean():.4f}")
    for f in mre.FAMILIES:
        r = {k: seed_stats(np.array([ev[t]["families"][f][k] for t in tags])) for k in ("d_mAP", "d_frontal", "d_tail")}
        res["families"][f] = r
        print(f"{f:13s} mAP {fmt(r['d_mAP'])}\n{'':13s} frontal {r['d_frontal']['mean']:+.4f} "
              f"[{r['d_frontal']['ci'][0]:+.4f}, {r['d_frontal']['ci'][1]:+.4f}] | tail {r['d_tail']['mean']:+.4f} "
              f"[{r['d_tail']['ci'][0]:+.4f}, {r['d_tail']['ci'][1]:+.4f}]")
    g_rand, g_none = fam["binary"] - fam["random"], fam["binary"] - fam["none"]
    res["binary_minus_random"], res["binary_minus_none"] = seed_stats(g_rand), seed_stats(g_none)
    print(f"binary - random {fmt(res['binary_minus_random'])}\nbinary - none   {fmt(res['binary_minus_none'])}")

    prim = res["families"]["binary"]["d_mAP"]
    holds = prim["ci"][0] > 0 and prim["wins"] >= 4
    gate = int(((g_rand > 0) & (g_none > 0)).sum()) >= 4
    res["verdict"] = {"primary_holds": bool(holds), "gate_counts": bool(gate)}
    print(f"\nPRE-REGISTERED VERDICT: primary {'HOLDS' if holds else 'FAILS'} | gate {'COUNTS' if gate else 'DOES NOT COUNT'}"
          f" (binary beats random and none in {int(((g_rand > 0) & (g_none > 0)).sum())}/5 seeds)")

    # Exploratory: the fold-chosen binary config on the flip-TTA area probabilities, against ASL of the same seed.
    pc = pd.read_csv(mre.PADCHEST_CSV, usecols=["ImageID", "PatientID"], dtype=str,
                     low_memory=False).drop_duplicates("ImageID").set_index("ImageID")
    dkeys = (O / "probs_unified_flip_area/keys.txt").read_text().split("\n")
    assert dkeys == (O / "probs_v0_flip_area/keys.txt").read_text().split("\n")
    rows = {"tta_cls": [], "tta_blend": [], "asl": []}
    for s, t in zip(SEEDS, tags):
        d = Path(args.root) / t
        keys = (d / "val_keys.txt").read_text().split("\n")
        order = pd.Index(dkeys).get_indexer(keys)               # probe rows -> dump rows
        assert (order >= 0).all(), f"{t}: probe keys missing from the dumps"
        L = np.load(d / "val_labels.npy").astype(np.int64)
        assert (L == np.load(O / "probs_unified_flip_area/labels.npy")[order]).all(), f"{t}: labels differ"
        ytr = np.load(d / "train_labels.npy").astype(np.uint8)
        kci, kcs = np.load(d / "knn_class_idx.npy"), np.load(d / "knn_class_sim.npy")
        kpi, kps = np.load(d / "knn_pool_idx.npy"), np.load(d / "knn_pool_sim.npy")
        pid = pc["PatientID"].reindex(keys).fillna("").values
        fold = np.array([zlib.crc32((p or k).encode()) % 2 for p, k in zip(pid, keys)])  # memory_readout_eval's split
        Pc = np.load(O / f"probs_unified_flip_area/c__seed_{s}__model_swa@768_flip_area.npy")[order].astype(np.float64)
        Pa = np.load(O / f"probs_v0_flip_area/v0_768__seed_{s}__model_swa@768_flip_area.npy")[order].astype(np.float64)
        hb = mre.binary_entropy(Pc)
        Pb = Pc.copy()
        for f, cfg in enumerate(ev[t]["families"]["binary"]["chosen"]):  # chosen[f] was tuned on the other half
            idx, sim = (kci, kcs) if cfg["embed"] == "class" else (kpi, kps)
            knn = mre.knn_readout(idx, sim, ytr, cfg["k"], cfg["tau"])
            a = cfg["a"] * hb
            sel = fold == f
            Pb[sel] = ((1 - a) * Pc + a * knn)[sel]
        ones = np.ones(len(keys))
        for k, P in (("tta_cls", Pc), ("tta_blend", Pb), ("asl", Pa)):
            rows[k].append(float(np.nanmean(ea.FastAP(P, L)(ones)[0])))
    R = {k: np.array(v) for k, v in rows.items()}
    res["exploratory_tta"] = {"mAP": {k: v.tolist() for k, v in R.items()},
                              "blend_minus_cls": seed_stats(R["tta_blend"] - R["tta_cls"]),
                              "blend_minus_asl": seed_stats(R["tta_blend"] - R["asl"]),
                              "cls_minus_asl": seed_stats(R["tta_cls"] - R["asl"])}
    print("\nEXPLORATORY (flip-TTA area; ASL has no readout):")
    for k in ("tta_cls", "tta_blend", "asl"):
        print(f"  {k:9s} {' '.join(f'{x:.4f}' for x in R[k])} | mean {R[k].mean():.4f}")
    for k in ("blend_minus_cls", "blend_minus_asl", "cls_minus_asl"):
        print(f"  {k:15s} {fmt(res['exploratory_tta'][k])}")
    args.out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
