"""Probe 3: why does --triplet-lambda 1.0 make Entropy worse, if the triplet gradient looks negligible?

Plan: docs/plan/2026-10-15-triplet-memory-debug-plan.md, follow-up after probes 1-2. Probes 1-2 measured the
triplet term at the END of training, where its easy-mined hinge is ~1% active. Weight space says that is not the
whole story: the lambda-1.0 run (p) moved 15% less from Stage-1 than ASL and every lambda-0.1 run, mostly in
backbone stages 2-3, and its difference from ASL points back toward Stage-1. Hypothesis: the triplet term is a
restoring constraint - ASL training drifts the mean-pooled embedding until easy-mined hinges break, the triplet
gradient then pushes back, and the end state looks inactive because the constraint is satisfied there.

  A  logits   : for the SWA checkpoints (Stage-1, ASL, c, p), eval mode on fixed val images, split each logit
                into the readout h_c . w_c and the bias b_c, and h_c . w_c into the dataset-shared part (mean over
                images) and the image-specific part; mean logit on negatives / positives, val ASL loss, geometry
  D  dynamics : train from Stage-1 exactly like train_one_epoch (AdamW, GradScaler, accum 12, bank filled
                during training, easy mining, anchor-rule mean) at each lambda, on the same seeded stream, at a
                constant LR, and log per span: active hinge fraction, L_tri, bank cosine; once per window on a
                micro-batch with an active hinge, lambda*|g_tri|/|g_cls| and cos(-g, theta - theta_stage1) for
                each loss; every --eval-every steps the A statistics plus the projection of this arm's drift on
                the lambda=0 arm's drift at the same step
  B  static   : the same instrumented replay at a trained checkpoint with lr 0 (bank warmed up first)

Run from train/; launcher scripts/submit_triplet_probe.sh MODE=3.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triplet_probe as TP  # noqa: E402  (also puts train/ on sys.path)
from triplet_probe2 import cos, flat  # noqa: E402
U, V = TP.U, TP.V

SWA = {"stage1": "stage1",
       "asl": "checkpoint_ASL/v0_768_seeded/seed_42/Model_run/model_swa.pth",
       "c": "checkpoint_Triplet_3/unified768_seeded/c/seed_42/Model_run/model_swa.pth",
       "p": "checkpoint_Triplet_3/unified768_seeded/p/seed_42/Model_run/model_swa.pth"}


def pvec(params):
    return torch.cat([p.detach().reshape(-1).float() for p in params])


@torch.no_grad()
def logit_stats(model, val, crit, device):
    was_training = model.training
    model.eval()
    W = model.head.duplicate_pooling.detach()[:, :, 0].float()
    b = model.head.duplicate_pooling_bias.detach().float()
    L, S, Hm, Y = [], [], [], []
    for x, y in val:
        with autocast("cuda"):
            lg, _ = model(V.preprocess_batch(x.to(device)))
        h = model.head.last_h.float()
        L.append(lg.float()); S.append(torch.einsum("bcd,cd->bc", h, W)); Hm.append(h.mean(1)); Y.append(y.to(device).float())
    L, S, Hm, Y = (torch.cat(t) for t in (L, S, Hm, Y))
    neg, pos = Y == 0, Y == 1
    shared = S.mean(0)
    Mn = F.normalize(Hm, dim=1)
    n = Mn.shape[0]
    model.train(was_training)
    return {"images": n, "val_asl_loss": float(crit(L, Y)),
            "mean_logit_neg": float(L[neg].mean()), "mean_logit_pos": float(L[pos].mean()),
            "logit_gap_pos_neg": float(L[pos].mean() - L[neg].mean()),
            "mean_prob_neg": float(torch.sigmoid(L[neg]).mean()), "mean_prob_pos": float(torch.sigmoid(L[pos]).mean()),
            "bias_mean": float(b.mean()),
            "readout_shared_mean": float(shared.mean()),                 # mean_c of (mean over images of h_c . w_c)
            "readout_image_std": float((S - shared).std(0).mean()),      # mean_c of image std of h_c . w_c
            "recon_max_err": float((S + b - L).abs().max()),
            "meanh_norm": float(Hm.norm(dim=1).mean()),
            "meanh_pairwise_cos": float(((Mn @ Mn.T).sum() - n) / (n * (n - 1))),
            "meanh_shared_norm_over_image_std": float(Hm.mean(0).norm() / Hm.var(0).sum().sqrt())}


def replay(tag, init, lam, lr, opt_steps, warmup, args, cw, head_ids, device, loader_fn, val, theta_ref, ref_drifts):
    """Train (lr > 0) or replay (lr == 0) the v5 step from `init`; theta_ref = Stage-1 params (flat, GPU)."""
    torch.manual_seed(args.seed)
    model = TP.load_model(init, device, args.drop_path)
    crit = V.AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=0.05,
                            reduction="mean", class_weights=cw)
    tri = nn.TripletMarginLoss(margin=args.margin, p=2)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = AdamW(params, lr=lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, s / args.warmup_opt) if args.warmup_opt > 0 else 1.0)
    scaler = GradScaler("cuda")
    pnames = [n for n, p in model.named_parameters() if p.requires_grad]
    pgroup = ["head" if n.startswith("head.") else
              ("bb_s23" if n.startswith(("stages.2", "stages.3", "downsample_layers.2", "downsample_layers.3"))
               else "bb_s01") for n in pnames]

    def adam_stats():
        """rms of sqrt(v_hat) (Adam's per-weight step denominator) and of the last update, per group."""
        acc = {}
        for p, g in zip(params, pgroup):
            st = optim.state.get(p, {})
            if "exp_avg_sq" not in st:
                continue
            t = float(st["step"])
            vh = st["exp_avg_sq"] / (1 - 0.999 ** t)
            a = acc.setdefault(g, [0.0, 0])
            a[0] += float(vh.sum()); a[1] += vh.numel()
        return {f"rms_sqrt_vhat_{g}": (s / n) ** 0.5 for g, (s, n) in acc.items()}
    mem = U.CrossBatchMemoryV2(embedding_dim=768, memory_size=args.memory_size, num_classes=30, device=device)

    def fwd(x):
        logits, _ = model(x)
        return logits, F.normalize(model.head.last_h.mean(dim=1), dim=1)

    spans, evals, inst = [], [], []
    span = {"micro": 0, "with_tri": 0, "triplets": 0, "active": 0, "l_tri_sum": 0.0, "l_cls_sum": 0.0}
    instrumented_this_window = False
    k = opt = 0
    t0 = time.time()

    def do_eval():
        row = {"opt_step": opt} | logit_stats(model, val, crit, device)
        d = pvec(params) - theta_ref
        row["drift_norm"] = float(d.norm())
        if lam == 0 and lr > 0:
            ref_drifts[opt] = d.cpu()
        elif opt in ref_drifts and float(ref_drifts[opt].norm()) > 0:
            r = ref_drifts[opt].to(device)
            row["proj_on_lam0_drift"] = float(d @ r / (r @ r))
            row["cos_with_lam0_drift"] = cos(d, r)
            del r
        del d
        evals.append(row)
        print(f"[{tag}] eval @opt {opt}: " + json.dumps({k_: round(v, 4) for k_, v in row.items()}), flush=True)

    if lr > 0:
        do_eval()
    step = 0
    for x, y, _r in loader_fn():
        step += 1
        x = V.preprocess_batch(x.to(device, non_blocking=True)); y = y.to(device).float()
        if step <= warmup:
            with torch.no_grad(), autocast("cuda"):
                _, e = fwd(x)
            mem.update(e.detach(), y)
            continue
        with autocast("cuda"):
            logits, e = fwd(x)
            trips = U.sample_triplets_v13(e, logits, y, mem, margin=args.margin, head_class_ids=head_ids,
                                          anchor_rule="mean", mining="easy")
            l_cls = crit(logits, y)
            if trips is None:
                l_tri = e.sum() * 0
                active = 0
            else:
                an, po, ne, dap, dan = TP.triplet_terms(trips)
                l_tri = tri(an, po, ne)
                active = int(((dap - dan + args.margin) > 0).sum())
                span["with_tri"] += 1; span["triplets"] += len(an); span["active"] += active
        span["micro"] += 1; span["l_tri_sum"] += float(l_tri); span["l_cls_sum"] += float(l_cls)

        # once per optimizer window: separate gradients on a micro-batch whose hinge is active
        if active and not instrumented_this_window and (lr == 0 or opt > 0):
            gs = torch.autograd.grad(l_cls * TP.LOSS_SCALE, params, retain_graph=True, allow_unused=True)
            g_cls = flat(gs, params) / TP.LOSS_SCALE
            gs = torch.autograd.grad(l_tri * TP.LOSS_SCALE, params, retain_graph=True, allow_unused=True)
            g_tri = flat(gs, params) / TP.LOSS_SCALE
            del gs
            d = pvec(params) - theta_ref
            inst.append({"opt_step": opt, "active": active,
                         "lam1_ratio_tri_over_cls": float(g_tri.norm() / g_cls.norm().clamp(min=1e-30)),
                         "cos_tri_cls": cos(g_tri, g_cls),
                         "cos_neg_gtri_drift": cos(-g_tri, d) if float(d.norm()) > 0 else float("nan"),
                         "cos_neg_gcls_drift": cos(-g_cls, d) if float(d.norm()) > 0 else float("nan")})
            del g_cls, g_tri, d
            instrumented_this_window = True

        if lr > 0:
            loss = (l_cls + lam * l_tri) / args.accum
            scaler.scale(loss).backward()
        mem.update(e.detach(), y)
        k += 1
        if k % args.accum == 0:
            upd = None
            if lr > 0:
                before = pvec(params) if (opt + 1) % args.log_every == 0 else None
                scaler.step(optim); scaler.update(); sched.step()
                optim.zero_grad(set_to_none=True)
                if before is not None:
                    upd = float((pvec(params) - before).norm()); del before
            opt += 1
            instrumented_this_window = False
            if opt % args.log_every == 0:
                kk = mem.filled
                Bk = mem.embeddings[:kk].float()
                bank_cos = float(((Bk @ Bk.T).sum() - kk) / max(1, kk * (kk - 1)))
                spans.append({"opt_step": opt, "micro": span["micro"],
                              "frac_micro_with_triplet": span["with_tri"] / span["micro"],
                              "active_hinge_frac": span["active"] / max(1, span["triplets"]),
                              "active_triplets_per_window": span["active"] / (span["micro"] / args.accum),
                              "l_tri_per_micro": span["l_tri_sum"] / span["micro"],
                              "l_cls_per_micro": span["l_cls_sum"] / span["micro"],
                              "bank_cos": bank_cos, "bank_filled": kk,
                              "lr": optim.param_groups[0]["lr"], "update_norm": upd} | (adam_stats() if lr > 0 else {}))
                print(f"[{tag}] opt {opt}/{opt_steps} ({time.time() - t0:.0f}s) " +
                      json.dumps({k_: (float(f"{v:.4g}") if isinstance(v, float) else v)
                                  for k_, v in spans[-1].items()}), flush=True)
                span = {"micro": 0, "with_tri": 0, "triplets": 0, "active": 0, "l_tri_sum": 0.0, "l_cls_sum": 0.0}
            if lr > 0 and opt % args.eval_every == 0:
                do_eval()
            if opt >= opt_steps:
                break
        if not (lr > 0):
            model.zero_grad(set_to_none=True)
    res = {"tag": tag, "init": init, "lam": lam, "lr": lr, "opt_steps": opt, "warmup": warmup,
           "spans": spans, "evals": evals, "instrumented": inst}
    if inst:
        res["instrumented_mean"] = {k_: float(np.nanmean([r[k_] for r in inst])) for k_ in inst[0] if k_ != "opt_step"}
        res["instrumented_mean"]["n"] = len(inst)
        res["random_cos_ref"] = float(1 / np.sqrt(sum(p.numel() for p in params)))
    del model, optim
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="A,D,B")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(TP.TRAIN_DIR), "analysis/out/triplet_probe3"))
    ap.add_argument("--lams", default="0,0.1,1.0")
    ap.add_argument("--lr", type=float, default=1e-4, help="constant LR for D (the real schedule peaks at 1e-4)")
    ap.add_argument("--opt-steps", type=int, default=150)
    ap.add_argument("--warmup-opt", type=int, default=0,
                    help="linear LR warm-up in optimizer steps (0 = constant; the real 4-of-8-epoch run uses 258)")
    ap.add_argument("--static-windows", type=int, default=16)
    ap.add_argument("--static-ckpts", default="asl,p")
    ap.add_argument("--warmup", type=int, default=512, help="bank warm-up batches for B (D fills it while training)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--val-batches", type=int, default=64)
    ap.add_argument("--accum", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--memory-size", type=int, default=2048)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=6)
    args = ap.parse_args()
    sys_argv, sys.argv = sys.argv, [sys.argv[0]]
    dflt = V.parse_args()
    sys.argv = sys_argv
    args.gamma_neg, args.gamma_pos = dflt.gamma_neg, dflt.gamma_pos
    assert dflt.margin == args.margin and dflt.weight_decay == args.weight_decay

    device = torch.device("cuda")
    _, cw, head_ids = TP.class_stats()
    os.makedirs(args.out_dir, exist_ok=True)
    parts = set(args.parts.split(","))

    def datasets():
        V.seed_everything(args.seed)
        return V.make_webdataset(TP.TRAIN_SHARDS, TP.VAL_SHARDS, seed=args.seed, strict_repro=True,
                                 img_size=768, shuffle_buf=V.SAMPLE_SHUFFLE_BUFSIZE)

    def loader_fn():
        tr, _, _ = datasets()
        g = torch.Generator(); g.manual_seed(args.seed)
        return DataLoader(tr, batch_size=4, num_workers=args.num_workers, pin_memory=True, generator=g)

    _, va, _ = datasets()
    val = []
    for i, (x, y, _r) in enumerate(DataLoader(va, batch_size=4, num_workers=2)):
        if i >= args.val_batches:
            break
        val.append((x, y))
    crit = V.AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=0.05,
                            reduction="mean", class_weights=cw)

    m0 = TP.load_model("stage1", device, args.drop_path)
    theta_ref = pvec([p for p in m0.parameters() if p.requires_grad])
    del m0
    torch.cuda.empty_cache()

    def dump(name, obj):
        obj["args"] = vars(args)
        with open(os.path.join(args.out_dir, f"{name}.json"), "w") as f:
            json.dump(obj, f, indent=2)

    if "A" in parts:
        out = {}
        for tag, path in SWA.items():
            m = TP.load_model(path, device, args.drop_path)
            out[tag] = logit_stats(m, val, crit, device)
            print(f"[A] {tag}: " + json.dumps({k: round(v, 4) for k, v in out[tag].items()}), flush=True)
            del m
            torch.cuda.empty_cache()
        dump("A_logits", out)

    if "D" in parts:
        ref_drifts = {}
        for lam in sorted(float(v) for v in args.lams.split(",")):
            tag = f"D_lam{lam:g}"
            dump(tag, replay(tag, "stage1", lam, args.lr, args.opt_steps, 0, args, cw, head_ids, device,
                             loader_fn, val, theta_ref, ref_drifts))
        del ref_drifts

    if "B" in parts:
        for tag in args.static_ckpts.split(","):
            dump(f"B_{tag}", replay(f"B_{tag}", SWA[tag], 1.0, 0.0, args.static_windows, args.warmup, args, cw,
                                    head_ids, device, loader_fn, val, theta_ref, {}))


if __name__ == "__main__":
    main()
