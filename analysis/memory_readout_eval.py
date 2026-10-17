"""Memory readout evaluation (2026-10-16, CPU) of analysis/memory_readout_probe.py outputs.

Question: does a readout of the training-set memory, blended with the classifier and optionally gated by the
classifier's entropy, beat the classifier alone on internal val; and is the Entropy (Triplet Memory) checkpoint's
memory better than the ASL checkpoint's? Plan: docs/plan/2026-10-16-beat-asl-plan.md.

Protocol (fixed before any result was seen):
  * probe: no TTA, area resize, same-patient train images excluded from every neighbour list;
  * kNN readout p_knn[i,c] = sum_j softmax(s_ij / tau) * y_j[c] over the top-K neighbours, from the per-class
    h_c lists ("class") or the pooled-embedding list ("pool"); prototype score as a logit-space blend;
  * blend p = (1 - a_ic) p_cls + a_ic p_knn, with gate families
      none          a_ic = a
      binary        a_ic = a * Hb(p_cls[i,c]) / ln 2        (per image-class Bernoulli entropy)
      softmax-hard  a_ic = a * 1[H_softmax(z_i) >= mean]    (the training gate's statistic and rule)
      random        a_ic = a for the same number of images as softmax-hard, chosen at random (control)
      knn-only      p = p_knn
      proto         logit(p) = z_cls + b * proto / sd(proto on the tuning half)
  * grid K {10, 25, 50, 100} x tau {0.01, 0.03, 0.1} x a {0.1, 0.2, 0.3, 0.5} x embed {class, pool};
    b {0.25, 0.5, 1, 2};
  * 2-fold cross-fitting by patient: within each family the config is chosen on one half of the val patients by
    mAP and scored on the other; the two scored halves are pooled and compared with the classifier on the same
    images (paired bootstrap over images).

    python -I analysis/memory_readout_eval.py --tags asl_seeded entropy_seeded
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import entropy_ablation as ea  # noqa: E402  (FastAP, class groups)

ROOT = ea.ROOT
PADCHEST_CSV = ROOT / "data/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
FRONTAL = ("PA", "AP", "AP_horizontal")
KS, TAUS, ALPHAS, BETAS = (10, 25, 50, 100), (0.01, 0.03, 0.1), (0.1, 0.2, 0.3, 0.5), (0.25, 0.5, 1.0, 2.0)
FAMILIES = ("none", "binary", "softmax-hard", "random", "knn-only", "proto")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def knn_readout(idx, sim, ytr, k, tau):
    """idx/sim (Nv,K') for one list -> weights over the first k; returns the weighted label average (Nv,30)
    or, for a per-class list (30,Nv,K'), the per-class weighted average of that class's label."""
    def w(s):
        s = s[..., :k] / tau
        e = np.exp(s - s.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
    if idx.ndim == 2:
        return np.einsum("nk,nkc->nc", w(sim), ytr[idx[:, :k]].astype(np.float32))
    return np.stack([(w(sim[c]) * ytr[idx[c, :, :k], c]).sum(1) for c in range(idx.shape[0])], 1)


def binary_entropy(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p)) / np.log(2)


def softmax_entropy(z):
    e = np.exp(z - z.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["asl_seeded", "entropy_seeded"])
    ap.add_argument("--root", default="/data/psytp7/memprobe")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--out", type=Path, default=ROOT / "analysis/out/memory_readout_eval.json")
    args = ap.parse_args()

    meta_g = ea.class_groups.build()
    groups = {g: np.where(meta_g["groups"][g])[0] for g in ("head", "medium", "tail")}
    pc = pd.read_csv(PADCHEST_CSV, usecols=["ImageID", "PatientID", "Projection"], dtype=str,
                     low_memory=False).drop_duplicates("ImageID").set_index("ImageID")
    res = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "protocol": __doc__.split("Protocol")[1],
           "tags": {}}
    ref_keys = None
    for tag in args.tags:
        d = Path(args.root) / tag
        keys = (d / "val_keys.txt").read_text().split("\n")
        z = np.load(d / "val_logits.npy").astype(np.float64)
        L = np.load(d / "val_labels.npy").astype(np.int64)
        ytr = np.load(d / "train_labels.npy").astype(np.uint8)
        kci, kcs = np.load(d / "knn_class_idx.npy"), np.load(d / "knn_class_sim.npy")
        kpi, kps = np.load(d / "knn_pool_idx.npy"), np.load(d / "knn_pool_sim.npy")
        proto = np.load(d / "proto_score.npy").astype(np.float64)
        if ref_keys is None:
            ref_keys = keys
        order = pd.Index(keys).get_indexer(ref_keys)       # align every tag to the first tag's row order
        assert (order >= 0).all() and len(order) == len(keys), f"{tag}: val keys differ"
        z, L, proto, kci, kcs, kpi, kps = z[order], L[order], proto[order], kci[:, order], kcs[:, order], \
            kpi[order], kps[order]
        n = len(ref_keys)
        pid = pc["PatientID"].reindex(ref_keys).fillna("").values
        fold = np.array([zlib.crc32((p or k).encode()) % 2 for p, k in zip(pid, ref_keys)])
        fr = pc["Projection"].reindex(ref_keys).isin(FRONTAL).values
        p_cls = sigmoid(z)

        def mAP(P, w):
            return float(np.nanmean(ea.FastAP(P, L)(w)[0]))

        hb = binary_entropy(p_cls)
        hs = softmax_entropy(z)
        hard = (hs >= hs.mean()).astype(np.float64)       # val-mean threshold: unlabeled statistic only
        rnd = np.zeros(n)
        rnd[np.random.default_rng(0).permutation(n)[:int(hard.sum())]] = 1.0
        knn = {(emb, k, tau): (knn_readout(kci, kcs, ytr, k, tau) if emb == "class"
                               else knn_readout(kpi, kps, ytr, k, tau)).astype(np.float32)
               for emb, k, tau in itertools.product(("class", "pool"), KS, TAUS)}
        gates = {"none": lambda a: a, "binary": lambda a: a * hb,
                 "softmax-hard": lambda a: a * hard[:, None], "random": lambda a: a * rnd[:, None]}

        def configs(fam, f):
            """(config, builder) pairs of a family; builders are lazy so candidates are not all held in memory."""
            if fam == "proto":
                sd = proto[fold == 1 - f].std(0) + 1e-9   # scale from the tuning half only
                return [({"b": b}, lambda b=b: sigmoid(z + b * proto / sd)) for b in BETAS]
            if fam == "knn-only":
                return [({"embed": e, "k": k, "tau": t}, lambda key=(e, k, t): knn[key].astype(np.float64))
                        for e, k, t in knn]
            return [({"embed": e, "k": k, "tau": t, "a": a},
                     lambda key=(e, k, t), a=a: (1 - gates[fam](a)) * p_cls + gates[fam](a) * knn[key])
                    for e, k, t in knn for a in ALPHAS]

        out = {"n": n, "n_frontal": int(fr.sum()), "fold_sizes": [int((fold == f).sum()) for f in (0, 1)],
               "softmax_hard_share": float(hard.mean()), "cls": {}, "families": {}}
        ones = np.ones(n)
        fa_cls = ea.FastAP(p_cls, L)
        ap_cls = fa_cls(ones)[0]
        out["cls"] = {"mAP": float(np.nanmean(ap_cls)), "frontal": mAP(p_cls, fr.astype(float)),
                      **{g: float(np.nanmean(ap_cls[gi])) for g, gi in groups.items()}}
        rng = np.random.default_rng(1)
        boots = [np.bincount(rng.integers(0, n, n), minlength=n).astype(float) for _ in range(args.boot)]
        cls_b = [(fa_cls(w)[0], fa_cls(w * fr)[0]) for w in boots]          # cached once, reused per family
        for fam in FAMILIES:
            Pfin, chosen = np.zeros_like(p_cls), []
            for f in (0, 1):
                tune, score = (fold == 1 - f).astype(float), fold == f
                best_c, best_m, best_P = None, -1.0, None
                for cfg, build in configs(fam, f):
                    P = build()
                    m = mAP(P, tune)
                    if m > best_m:
                        best_c, best_m, best_P = cfg, m, P
                chosen.append({**best_c, "tune_mAP": best_m})
                Pfin[score] = best_P[score]
            fa = ea.FastAP(Pfin, L)
            ap_all = fa(ones)[0]
            r = {"chosen": chosen, "mAP": float(np.nanmean(ap_all)), "frontal": mAP(Pfin, fr.astype(float)),
                 **{g: float(np.nanmean(ap_all[gi])) for g, gi in groups.items()}}
            r["d_mAP"] = r["mAP"] - out["cls"]["mAP"]
            r["d_frontal"] = r["frontal"] - out["cls"]["frontal"]
            bd = []
            for w, (cb, cbf) in zip(boots, cls_b):
                a_w, a_f = fa(w)[0], fa(w * fr)[0]
                bd.append((np.nanmean(a_w) - np.nanmean(cb), np.nanmean(a_f) - np.nanmean(cbf),
                           np.nanmean(a_w[groups["tail"]]) - np.nanmean(cb[groups["tail"]])))
            bd = np.array(bd)
            r["ci_mAP"], r["ci_frontal"], r["ci_tail"] = [np.percentile(bd[:, j], [2.5, 97.5]).tolist() for j in range(3)]
            r["d_tail"] = r["tail"] - out["cls"]["tail"]
            out["families"][fam] = r
        res["tags"][tag] = out

        print(f"\n== {tag}: n={n} (frontal {int(fr.sum())}), classifier alone mAP {out['cls']['mAP']:.4f} "
              f"frontal {out['cls']['frontal']:.4f} | head {out['cls']['head']:.4f} medium {out['cls']['medium']:.4f} "
              f"tail {out['cls']['tail']:.4f} | softmax-hard gate opens on {hard.mean():.0%} of images")
        print(f"{'family':13s} {'mAP':>7s} {'d':>8s} {'95% CI':>19s} | {'frontal d':>9s} {'95% CI':>19s} | "
              f"{'tail d':>8s} {'95% CI':>19s} | chosen (fold 0 | fold 1)")
        for fam, r in out["families"].items():
            print(f"{fam:13s} {r['mAP']:.4f} {r['d_mAP']:+.4f} [{r['ci_mAP'][0]:+.4f}, {r['ci_mAP'][1]:+.4f}] | "
                  f"{r['d_frontal']:+9.4f} [{r['ci_frontal'][0]:+.4f}, {r['ci_frontal'][1]:+.4f}] | "
                  f"{r['d_tail']:+.4f} [{r['ci_tail'][0]:+.4f}, {r['ci_tail'][1]:+.4f}] | {r['chosen'][0]} | {r['chosen'][1]}")

    if len(res["tags"]) == 2:
        (a, ra), (b, rb) = res["tags"].items()
        print(f"\n== {b} - {a} (memory quality: knn-only; best blends) ==")
        for fam in FAMILIES:
            print(f"{fam:13s} mAP {rb['families'][fam]['mAP'] - ra['families'][fam]['mAP']:+.4f} | "
                  f"classifier {rb['cls']['mAP'] - ra['cls']['mAP']:+.4f}")
    args.out.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
