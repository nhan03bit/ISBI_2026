"""Probe 2: WHY does the Triplet Memory gradient not change the classifier? (read-only, no training)

Plan: docs/plan/2026-10-15-triplet-memory-debug-plan.md, follow-up "why is the Entropy gradient mitigated".
Same replay as analysis/triplet_probe.py (v5 train step, 768 px, batch 4, seeded stream, 2048 bank,
easy mining, --anchor-rule mean, lambda 0.1), but it measures the candidate damping mechanisms:

  M1 accumulation: per optimizer window of 12 micro-batches, coherence ||sum g|| / sum ||g|| of the
     triplet and ASL gradients, and the effective ratio lambda*||sum g_tri|| / ||sum g_cls||
  M2 across windows (what Adam momentum + EMA 0.999 + SWA average over): cosine between consecutive
     window sums and coherence of the sum over all windows
  M3 readout: logit_c = h_c . w_c, so the part of dL/dh that moves logits is (dL/dh_c . w_c).
     Ratio of the triplet's logit-moving push to its total push on h, vs the same for ASL
  M4 function space: on fixed val images (eval mode, fp32), d(logits)/d(step) and d(embedding)/d(step)
     along the accumulated triplet direction vs ASL direction vs a random direction (same step norm)
  M5 embedding: share of the mean-pooled embedding that is the frozen, image-independent mean query
     vector, and image variance of mean_c h_c vs the per-query image variance
  M6 precision: triplet gradient under fp16 autocast (training path) vs fp32, eval mode, same batch

Run from train/ (imports train_2_v5). Launcher: scripts/submit_triplet_probe.sh MODE=2.
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
from torch.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triplet_probe as TP  # noqa: E402  (also puts train/ on sys.path)
U, V = TP.U, TP.V


def flat(gs, params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).float() for g, p in zip(gs, params)])


def cos(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp(min=1e-30))


def add_(params, vec, scale):
    off = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.add_(vec[off:off + n].view_as(p).to(p.dtype), alpha=scale)
            off += n


def run(tag, path, args, cw, head_ids, device, loader_fn, val_fn):
    torch.manual_seed(args.seed)
    model = TP.load_model(path, device, args.drop_path)
    crit = V.AsymmetricLoss(gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos, clip=0.05,
                            reduction="mean", class_weights=cw)
    tri = nn.TripletMarginLoss(margin=args.margin, p=2)
    params = [p for p in model.parameters() if p.requires_grad]
    mem = U.CrossBatchMemoryV2(embedding_dim=768, memory_size=args.memory_size, num_classes=30, device=device)
    W = model.head.duplicate_pooling.detach()[:, :, 0].float()          # (30, D): logit_c = h_c . w_c + b_c
    m_q = model.head.query_embed.weight.detach().float().mean(0)          # frozen, image-independent

    def fwd(x):
        logits, _ = model(x)
        h = model.head.last_h
        return logits, h, F.normalize(h.mean(dim=1), dim=1)

    def grads(loss, wrt):
        return torch.autograd.grad(loss * TP.LOSS_SCALE, wrt, retain_graph=True, allow_unused=True)

    win = {"coh_cls": [], "coh_tri": [], "ratio": [], "cos_cls_tri": [], "xwin_cos_cls": [], "xwin_cos_tri": [],
           "micro_ratio": [], "micro_with_tri": []}
    m3 = {"tri_logit_share": [], "cls_logit_share": [], "ratio_h": [], "ratio_logit": []}
    S_cls = S_tri = P_cls = P_tri = T_cls = T_tri = None
    sum_n_cls = sum_n_tri = 0.0
    win_sums_cls, win_sums_tri = [], []          # norms of window sums, for all-window coherence
    m_all, h_sum, h_sq, n_img = [], None, None, 0
    step, k = 0, 0
    t0 = time.time()
    for x, y, _r in loader_fn():
        step += 1
        x = V.preprocess_batch(x.to(device, non_blocking=True)); y = y.to(device).float()
        if step <= args.warmup:
            with torch.no_grad(), autocast("cuda"):
                _, _, e = fwd(x)
            mem.update(e.detach(), y)
            continue
        if k >= args.windows * args.accum:
            break
        with autocast("cuda"):
            logits, h, e = fwd(x)
            l_cls = crit(logits, y)
        trips = U.sample_triplets_v13(e, logits, y, mem, margin=args.margin, head_class_ids=head_ids,
                                      anchor_rule="mean", mining="easy")
        g_cls_all = grads(l_cls, params + [h])
        g_cls = flat(g_cls_all[:-1], params) / TP.LOSS_SCALE
        gh_cls = g_cls_all[-1].float() / TP.LOSS_SCALE
        if trips is not None:
            an, po, ne = (F.normalize(t.float(), dim=-1) for t in trips)
            l_tri = args.lam * tri(an, po, ne)
            g_tri_all = grads(l_tri, params + [h])
            g_tri = flat(g_tri_all[:-1], params) / TP.LOSS_SCALE
            gh_tri = (g_tri_all[-1].float() / TP.LOSS_SCALE) if g_tri_all[-1] is not None else torch.zeros_like(gh_cls)
        else:
            g_tri = torch.zeros_like(g_cls); gh_tri = torch.zeros_like(gh_cls)
        win["micro_with_tri"].append(trips is not None)
        if trips is not None:
            win["micro_ratio"].append(float(g_tri.norm() / g_cls.norm().clamp(min=1e-30)))
            # M3: share of the push on h that moves logits: per class, the component along w_c
            def logit_push(gh):   # (B,30,D) . (30,D) -> (B,30)
                return torch.einsum("bcd,cd->bc", gh, W)
            lt, lc = logit_push(gh_tri), logit_push(gh_cls)
            wn = W.norm(dim=1)[None, :]
            m3["tri_logit_share"].append(float((lt / wn).norm() / gh_tri.norm().clamp(min=1e-30)))
            m3["cls_logit_share"].append(float((lc / wn).norm() / gh_cls.norm().clamp(min=1e-30)))
            m3["ratio_h"].append(float(gh_tri.norm() / gh_cls.norm().clamp(min=1e-30)))
            m3["ratio_logit"].append(float(lt.norm() / lc.norm().clamp(min=1e-30)))
        # M5 statistics (detached)
        with torch.no_grad():
            hm = h.detach().float().mean(1)
            m_all.append(hm.cpu())
            hh = h.detach().float()
            h_sum = hh.sum(0) if h_sum is None else h_sum + hh.sum(0)
            h_sq = (hh ** 2).sum(0) if h_sq is None else h_sq + (hh ** 2).sum(0)
            n_img += hh.size(0)
        mem.update(e.detach(), y)
        model.zero_grad(set_to_none=True)

        if S_cls is None:
            S_cls, S_tri = torch.zeros_like(g_cls), torch.zeros_like(g_cls)
        S_cls += g_cls; S_tri += g_tri
        sum_n_cls += float(g_cls.norm()); sum_n_tri += float(g_tri.norm())
        k += 1
        if k % args.accum == 0:              # one optimizer window
            win["coh_cls"].append(float(S_cls.norm()) / max(sum_n_cls, 1e-30))
            win["coh_tri"].append(float(S_tri.norm()) / max(sum_n_tri, 1e-30))
            win["ratio"].append(float(S_tri.norm() / S_cls.norm().clamp(min=1e-30)))
            win["cos_cls_tri"].append(cos(S_cls, S_tri))
            if P_cls is not None:
                win["xwin_cos_cls"].append(cos(S_cls, P_cls)); win["xwin_cos_tri"].append(cos(S_tri, P_tri))
            win_sums_cls.append(float(S_cls.norm())); win_sums_tri.append(float(S_tri.norm()))
            T_cls = S_cls.clone() if T_cls is None else T_cls + S_cls
            T_tri = S_tri.clone() if T_tri is None else T_tri + S_tri
            P_cls, P_tri = S_cls.clone(), S_tri.clone()
            S_cls.zero_(); S_tri.zero_(); sum_n_cls = sum_n_tri = 0.0
            print(f"[{tag}] window {k // args.accum}/{args.windows} ({time.time() - t0:.0f}s)", flush=True)
    del S_cls, S_tri, P_cls, P_tri
    torch.cuda.empty_cache()

    def mean(v):
        return float(np.mean(v)) if len(v) else float("nan")
    res = {"tag": tag, "ckpt": path, "lam": args.lam, "accum": args.accum, "windows": args.windows,
           "M1_accumulation": {"micro_ratio_lam_tri_over_cls": mean(win["micro_ratio"]),
                               "micro_frac_with_triplet": mean(win["micro_with_tri"]),
                               "window_coherence_cls": mean(win["coh_cls"]), "window_coherence_tri": mean(win["coh_tri"]),
                               "random_coherence_ref": 1 / np.sqrt(args.accum),
                               "window_ratio_lam_tri_over_cls": mean(win["ratio"]),
                               "window_cos_cls_tri": mean(win["cos_cls_tri"])},
           "M2_across_windows": {"consecutive_cos_cls": mean(win["xwin_cos_cls"]),
                                 "consecutive_cos_tri": mean(win["xwin_cos_tri"]),
                                 "all_window_coherence_cls": float(T_cls.norm()) / max(sum(win_sums_cls), 1e-30),
                                 "all_window_coherence_tri": float(T_tri.norm()) / max(sum(win_sums_tri), 1e-30),
                                 "random_coherence_ref": 1 / np.sqrt(len(win_sums_cls)),
                                 "total_ratio_lam_tri_over_cls": float(T_tri.norm() / T_cls.norm().clamp(min=1e-30)),
                                 "total_cos_cls_tri": cos(T_cls, T_tri)},
           "M3_readout": {k_: mean(v) for k_, v in m3.items()} | {"random_dir_logit_share_ref": float(1 / np.sqrt(768))}}

    # M5: embedding decomposition
    M = torch.cat(m_all)                                         # (N, D) mean_c h_c, pre-normalisation
    Mn = F.normalize(M, dim=1)
    off = lambda A: float((A @ A.T).fill_diagonal_(0).sum() / (A.shape[0] * (A.shape[0] - 1)))  # noqa: E731
    var_m = float(M.var(0).sum())
    h_mean = h_sum / n_img
    var_h = float(((h_sq / n_img) - h_mean ** 2).sum(-1).mean())   # mean over queries of per-query image variance
    res["M5_embedding"] = {"norm_mean_query_const": float(m_q.norm()), "norm_mean_h": float(M.norm(dim=1).mean()),
                           "cos_mean_h_vs_const": float((Mn @ F.normalize(m_q.cpu(), dim=0)).mean()),
                           "norm_batchmean_of_mean_h": float(M.mean(0).norm()),
                           "image_std_of_mean_h": var_m ** 0.5,
                           "mean_pairwise_cos_mean_h": off(Mn),
                           "mean_pairwise_cos_after_removing_dataset_mean": off(F.normalize(M - M.mean(0), dim=1)),
                           "image_var_mean_h_over_mean_per_query_var": var_m / max(var_h, 1e-30),
                           "independent_queries_ref": 1 / 30}

    # M4: function-space effect of a step along each accumulated direction (eval mode, fp32)
    model.eval()
    Xv = [V.preprocess_batch(xv.to(device)) for xv, *_ in val_fn()]
    g = torch.Generator(device="cpu").manual_seed(0)
    dirs = {"tri": T_tri / T_tri.norm(), "cls": T_cls / T_cls.norm(),
            "rand": F.normalize(torch.randn(T_cls.numel(), generator=g), dim=0).to(device)}
    m4 = {}
    with torch.no_grad():
        for name, u in dirs.items():
            outs = []
            for s in (+1, -1):
                add_(params, u, s * args.step)
                L_, E_ = [], []
                for xv in Xv:
                    lg, _ = model(xv)
                    L_.append(lg.float()); E_.append(F.normalize(model.head.last_h.float().mean(1), dim=1))
                outs.append((torch.cat(L_), torch.cat(E_)))
                add_(params, u, -s * args.step)
            dl = (outs[0][0] - outs[1][0]) / (2 * args.step)
            de = (outs[0][1] - outs[1][1]) / (2 * args.step)
            m4[name] = {"rms_dlogit_per_unit_step": float(dl.pow(2).mean().sqrt()),
                        "rms_dembed_per_unit_step": float(de.pow(2).mean().sqrt())}
    for name in m4:
        m4[name]["dlogit_over_dembed"] = m4[name]["rms_dlogit_per_unit_step"] / max(m4[name]["rms_dembed_per_unit_step"], 1e-30)
    res["M4_function_space"] = m4 | {"step_l2": args.step, "val_images": sum(x.size(0) for x in Xv)}

    # M6: fp16 autocast vs fp32 triplet gradient on the same eval-mode batches
    m6 = []
    for i, xv in enumerate(Xv[:4]):
        yv = torch.zeros(xv.size(0), 30, device=device)
        gs = {}
        for prec in ("fp16", "fp32"):
            with autocast("cuda", enabled=(prec == "fp16")):
                lg, h, e = fwd(xv)
            # deterministic pseudo-labels so triplets exist: model's own top-3 predictions
            yv = torch.zeros_like(lg, dtype=torch.float32).scatter_(1, lg.float().topk(3, dim=1).indices, 1.0) if prec == "fp16" else yv
            t = U.sample_triplets_v13(e, lg, yv, mem, margin=args.margin, head_class_ids=head_ids,
                                      anchor_rule="all", mining="hard")
            if t is None:
                break
            an, po, ne = (F.normalize(z.float(), dim=-1) for z in t)
            gs[prec] = flat(grads(args.lam * tri(an, po, ne), params), params) / TP.LOSS_SCALE
            model.zero_grad(set_to_none=True)
        if len(gs) == 2:
            m6.append({"cos": cos(gs["fp16"], gs["fp32"]), "norm_ratio": float(gs["fp16"].norm() / gs["fp32"].norm())})
    res["M6_precision"] = m6
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(TP.TRAIN_DIR), "analysis/out/triplet_probe2"))
    ap.add_argument("--warmup", type=int, default=512)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--accum", type=int, default=12)
    ap.add_argument("--val-batches", type=int, default=16)
    ap.add_argument("--step", type=float, default=0.5, help="L2 norm of the finite-difference step (M4)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--memory-size", type=int, default=2048)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=6)
    args = ap.parse_args()
    sys_argv, sys.argv = sys.argv, [sys.argv[0]]
    dflt = V.parse_args()
    sys.argv = sys_argv
    args.gamma_neg, args.gamma_pos = dflt.gamma_neg, dflt.gamma_pos

    device = torch.device("cuda")
    _, cw, head_ids = TP.class_stats()
    os.makedirs(args.out_dir, exist_ok=True)

    def datasets():
        V.seed_everything(args.seed)
        return V.make_webdataset(TP.TRAIN_SHARDS, TP.VAL_SHARDS, seed=args.seed, strict_repro=True,
                                 img_size=768, shuffle_buf=V.SAMPLE_SHUFFLE_BUFSIZE)

    def loader_fn():
        tr, _, _ = datasets()
        g = torch.Generator(); g.manual_seed(args.seed)
        return DataLoader(tr, batch_size=4, num_workers=args.num_workers, pin_memory=True, generator=g)

    def val_fn():
        _, va, _ = datasets()
        out = []
        for i, b in enumerate(DataLoader(va, batch_size=4, num_workers=2)):
            if i >= args.val_batches:
                break
            out.append(b)
        return out

    for spec in args.ckpt:
        tag, path = spec.split("=", 1)
        res = run(tag, path, args, cw, head_ids, device, loader_fn, val_fn)
        res["args"] = vars(args)
        with open(os.path.join(args.out_dir, f"{tag}.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps({k: v for k, v in res.items() if k != "args"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
