"""Read-only probe: how much does the Triplet Memory term in train/train_2_v5.py actually do?

Plan: docs/plan/2026-10-15-triplet-memory-debug-plan.md. For each checkpoint it replays the v5
training step (train mode, 768 px, batch 4, seeded train stream, 2048-entry cross-batch memory,
--metric-embed query, --head-classes 5, --anchor-rule mean) WITHOUT optimizer steps and logs:
  H1  triplet loss + fraction of active hinges under easy / hard / semihard mining (same anchors)
  H2  ||lambda * grad L_tri|| / ||grad L_cls|| and their cosine, over backbone and head params
  H3  embedding geometry: mean pairwise cosine in the bank, d(anchor,pos), d(anchor,neg)
  H4  memory-bank consistency: written labels/embeddings read back, fill level
  H5  anchors picked by softmax vs binary (multi-label) entropy

Run from train/ (imports train_2_v5); see scripts/submit_triplet_probe.sh.
  python ../analysis/triplet_probe.py --ckpt c=PATH --ckpt asl=PATH --ckpt stage1=stage1
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

TRAIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train")
sys.path.insert(0, TRAIN_DIR)
import utils_update as U  # noqa: E402
import train_2_v5 as V  # noqa: E402

STAGE1 = "checkpoint3/Model_20260119_062652/model_best.pth"
TRAIN_SHARDS = "/data/psytp7/wds_shards_train_raw"
VAL_SHARDS = "/data/psytp7/wds_shards_val_raw"
LOSS_SCALE = 2.0 ** 12   # fp16 autocast: scale before autograd.grad so tiny grads do not underflow
MINING = ("easy", "hard", "semihard")


def class_stats(k_head=5):
    """Shard-order class weights and head ids, copied from train_2_v5.main (--class-weight-order shard)."""
    df = pd.read_csv(os.path.join(os.path.dirname(TRAIN_DIR), "data", "CXRLT_2026_training_filtered.csv"))
    order = list(torch.load(os.path.join(TRAIN_SHARDS, "label_info.pt"), weights_only=False)["class_names"])
    lcounts = df[order].values.astype(np.float32).sum(axis=0)
    cw = np.log(1.0 / (lcounts / len(df) + 1e-8))
    cw = np.clip(cw / cw.mean(), 0.5, 2.0)
    head_ids = [int(i) for i in np.argsort(lcounts)[::-1][:k_head]]
    return order, cw, head_ids


def load_model(path, device, drop_path):
    model = V.create_model(30, STAGE1, drop_path_rate=drop_path)   # Stage-1 init (strict=False, as in training)
    if path != "stage1":
        sd = torch.load(path if os.path.isabs(path) else os.path.join(TRAIN_DIR, path), map_location="cpu")
        model.load_state_dict(sd, strict=True)                    # EMA state dict saved by _save_best
    return model.to(device).train()


def triplet_terms(trips):
    an, po, ne = (F.normalize(t.float(), dim=-1) for t in trips)
    dap, dan = F.pairwise_distance(an, po), F.pairwise_distance(an, ne)
    return an, po, ne, dap, dan


def flat_grad(loss, params):
    gs = torch.autograd.grad(loss * LOSS_SCALE, params, retain_graph=True, allow_unused=True)
    return [None if g is None else g.float() / LOSS_SCALE for g in gs]


def group_stats(g_cls, g_tri, groups, lam):
    out = {}
    for name, idx in groups.items():
        dot = n_c = n_t = 0.0
        for i in idx:
            c, t = g_cls[i], g_tri[i]
            if c is not None:
                n_c += float((c * c).sum())
            if t is not None:
                n_t += float((t * t).sum())
            if c is not None and t is not None:
                dot += float((c * t).sum())
        n_c, n_t = n_c ** 0.5, n_t ** 0.5
        out[name] = {"cls_norm": n_c, "tri_norm": n_t,
                     "ratio_lam": lam * n_t / max(n_c, 1e-30),
                     "cos": dot / max(n_c * n_t, 1e-30)}
    return out


def probe(tag, path, loader_fn, args, cw, head_ids, device):
    torch.manual_seed(args.seed)
    model = load_model(path, device, args.drop_path)
    crit = V.AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=0.05,
                            reduction="mean", class_weights=cw)
    params = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    groups = {"all": list(range(len(params))),
              "head": [i for i, n in enumerate(names) if n.startswith("head.")],
              "backbone": [i for i, n in enumerate(names) if not n.startswith("head.")]}
    mem = U.CrossBatchMemoryV2(embedding_dim=args.embedding_dim, memory_size=args.memory_size,
                               num_classes=30, device=device)

    def embed(x):
        logits, _ = model(x)
        return logits, F.normalize(model.head.last_h.mean(dim=1), dim=1)

    rec = {m: {"steps_with_triplet": 0, "triplets": 0, "loss": [], "active": [], "dap": [], "dan": []} for m in MINING}
    grad_rows = {m: [] for m in ("easy", "semihard")}
    gate_jacc, n_anchor_soft, n_anchor_bin, h4_fail, cls_losses = [], 0, 0, 0, 0.0
    t0, step = time.time(), 0
    for x, y, _r in loader_fn():
        step += 1
        x = V.preprocess_batch(x.to(device, non_blocking=True))
        y = y.to(device).float()
        if step <= args.warmup:                                 # fill the bank, as training would have
            with torch.no_grad(), autocast("cuda"):
                _, e = embed(x)
            mem.update(e.detach(), y)
            continue
        if step > args.warmup + args.batches:
            break
        with autocast("cuda"):
            logits, e = embed(x)
            l_cls = crit(logits, y)
        cls_losses += float(l_cls)
        g_cls = None
        # H5: anchor gate under softmax (what v13 uses) vs binary entropy
        gs = U._anchor_gate(U.calculate_entropy_from_logits(logits.float(), mode="softmax"), "mean")
        gb = U._anchor_gate(U.calculate_entropy_from_logits(logits.float(), mode="binary"), "mean")
        n_anchor_soft += int(gs.sum()); n_anchor_bin += int(gb.sum())
        gate_jacc.append(float((gs & gb).sum()) / max(1, int((gs | gb).sum())))
        for m in MINING:
            trips = U.sample_triplets_v13(e, logits, y, mem, margin=args.margin, head_class_ids=head_ids,
                                          anchor_rule="mean", mining=m)
            r = rec[m]
            if trips is None:
                continue
            an, po, ne, dap, dan = triplet_terms(trips)
            l_tri = nn.TripletMarginLoss(margin=args.margin, p=2)(an, po, ne)
            hinge = (dap - dan + args.margin).clamp(min=0)
            assert abs(float(hinge.mean()) - float(l_tri)) < 1e-3, "hinge recompute != TripletMarginLoss"
            r["steps_with_triplet"] += 1; r["triplets"] += len(an)
            r["loss"].append(float(l_tri)); r["active"].append(float((hinge > 0).float().mean()))
            r["dap"] += dap.tolist(); r["dan"] += dan.tolist()
            if m in grad_rows and step - args.warmup <= args.grad_batches:
                if g_cls is None:
                    g_cls = flat_grad(l_cls, params)
                grad_rows[m].append(group_stats(g_cls, flat_grad(l_tri, params), groups, args.lam))
        # H4: the bank stores exactly what was written (labels exact, embeddings to fp16 precision)
        ptr0 = mem.ptr
        mem.update(e.detach(), y)
        idx = [(ptr0 + i) % mem.memory_size for i in range(y.size(0))]
        if not (torch.equal(mem.labels[idx], y)
                and torch.allclose(mem.embeddings[idx], e.detach().float(), atol=2e-3)):
            h4_fail += 1
        model.zero_grad(set_to_none=True)
        if (step - args.warmup) % 50 == 0:
            print(f"[{tag}] {step - args.warmup}/{args.batches} batches ({time.time() - t0:.0f}s)", flush=True)

    n = max(1, step - 1 - args.warmup)
    with torch.no_grad():                                        # H3 on the final bank
        k = mem.filled
        B = mem.embeddings[:k].float()
        cos = B @ B.T
        off = (cos.sum() - cos.diag().sum()) / max(1, k * (k - 1))
    def mean(v):
        return float(np.mean(v)) if len(v) else float("nan")
    out = {"tag": tag, "ckpt": path, "batches": n, "warmup": args.warmup, "lam": args.lam,
           "cls_loss_mean": cls_losses / n,
           "mining": {m: {"steps_with_triplet_frac": r["steps_with_triplet"] / n,
                          "triplets_per_step": r["triplets"] / n,
                          "loss_mean_when_present": mean(r["loss"]),
                          "loss_mean_per_step": float(np.sum(r["loss"])) / n,
                          "active_hinge_frac": mean(r["active"]),
                          "d_ap_mean": mean(r["dap"]), "d_an_mean": mean(r["dan"]),
                          "d_an_minus_d_ap": mean(r["dan"]) - mean(r["dap"])} for m, r in rec.items()},
           "grad": {m: {g: {s: mean([row[g][s] for row in rows]) for s in ("cls_norm", "tri_norm", "ratio_lam", "cos")}
                        for g in ("all", "head", "backbone")} | {"n": len(rows)}
                    for m, rows in grad_rows.items()},
           "bank": {"filled": int(mem.filled), "mean_offdiag_cos": float(off), "h4_fail_batches": h4_fail},
           "gate": {"anchors_per_step_softmax": n_anchor_soft / n, "anchors_per_step_binary": n_anchor_bin / n,
                    "jaccard_softmax_vs_binary": mean(gate_jacc)}}
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="tag=path (path 'stage1' = Stage-1 init)")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(TRAIN_DIR), "analysis/out/triplet_probe"))
    ap.add_argument("--warmup", type=int, default=512, help="batches to fill the 2048 bank (512 x 4)")
    ap.add_argument("--batches", type=int, default=300)
    ap.add_argument("--grad-batches", type=int, default=150, help="batches with the two extra backward passes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img-size", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--memory-size", type=int, default=2048)
    ap.add_argument("--embedding-dim", type=int, default=768)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--gamma-neg", type=float, default=None)
    ap.add_argument("--gamma-pos", type=float, default=None)
    args = ap.parse_args()
    # ASL gammas: take the v5 defaults so the probe matches training exactly
    sys_argv, sys.argv = sys.argv, [sys.argv[0]]
    dflt = V.parse_args()
    sys.argv = sys_argv
    args.gamma_neg = dflt.gamma_neg if args.gamma_neg is None else args.gamma_neg
    args.gamma_pos = dflt.gamma_pos if args.gamma_pos is None else args.gamma_pos
    assert dflt.margin == args.margin and dflt.embedding_dim == args.embedding_dim

    device = torch.device("cuda")
    order, cw, head_ids = class_stats()
    print("head classes:", [order[i] for i in head_ids], flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    def loader_fn():   # identical seeded stream for every checkpoint -> paired comparison
        V.seed_everything(args.seed)
        train_ds, _, _ = V.make_webdataset(TRAIN_SHARDS, VAL_SHARDS, seed=args.seed, strict_repro=True,
                                           img_size=args.img_size, shuffle_buf=V.SAMPLE_SHUFFLE_BUFSIZE)
        g = torch.Generator(); g.manual_seed(args.seed)
        return DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                          pin_memory=True, generator=g, persistent_workers=False)

    results = []
    for spec in args.ckpt:
        tag, path = spec.split("=", 1)
        res = probe(tag, path, loader_fn, args, cw, head_ids, device)
        res["args"] = vars(args)
        with open(os.path.join(args.out_dir, f"{tag}.json"), "w") as f:
            json.dump(res, f, indent=2)
        results.append(res)

    print("\n| ckpt | mining | trip/step | L_tri/step | active hinge | d_an-d_ap | lam*|gTri|/|gCls| all | backbone | cos |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        for m in MINING:
            mm, gg = r["mining"][m], r["grad"].get(m, {})
            ga = gg.get("all", {}); gb = gg.get("backbone", {})
            print(f"| {r['tag']} | {m} | {mm['triplets_per_step']:.2f} | {mm['loss_mean_per_step']:.4f} | "
                  f"{mm['active_hinge_frac']:.3f} | {mm['d_an_minus_d_ap']:.3f} | "
                  f"{ga.get('ratio_lam', float('nan')):.2e} | {gb.get('ratio_lam', float('nan')):.2e} | "
                  f"{ga.get('cos', float('nan')):.3f} |")
        print(f"| {r['tag']} | bank cos {r['bank']['mean_offdiag_cos']:.3f}, H4 fails {r['bank']['h4_fail_batches']}, "
              f"gate jaccard {r['gate']['jaccard_softmax_vs_binary']:.2f}, L_cls {r['cls_loss_mean']:.4f} |")


if __name__ == "__main__":
    main()
