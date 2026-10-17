"""Memory readout probe (2026-10-16): can a memory of training-set embeddings add information beyond the
classifier at test time, and does Triplet Memory training make that memory better?

Why: Triplet Memory as trained in train_2_v5.py is near-inactive (docs/plan/2026-10-15-triplet-memory-debug.md),
so a test-time readout of the memory is the cheapest way to ask whether "memory" has any value on top of ASL
before spending training GPU time (docs/plan/2026-10-16-beat-asl-plan.md).

For one checkpoint, one forward pass (no TTA, --interp resize, default area) over every train-shard and val-shard
image, keeping the logits and the decoded per-class query embeddings h (N, 30, 768) that the readout reads
(logit_c = h_c . w_c + b_c). Then, for every val image:
  * per-class kNN: the top-K train images by cosine of h_c, for each class c;
  * pooled kNN: the top-K train images by cosine of normalize(mean_c h_c), the Triplet Memory's metric embedding;
  * prototype score: cos(h_c, mean positive h_c) - cos(h_c, mean negative h_c) over the train images.
Train images of the SAME PATIENT are excluded from every neighbour list: 77% of val images share a patient
with the train fold and labels are per study, so a same-patient neighbour would leak the label (the leaderboard
test set is assumed patient-disjoint; the Markov-with-history gain shows the size of that leak).

Outputs in OUT/<tag>/ (analysis: analysis/memory_readout_eval.py):
  {train,val}_keys.txt  {train,val}_labels.npy (N,30) u8  {train,val}_logits.npy (N,30) f32
  knn_class_idx.npy (30,Nv,K) i32  knn_class_sim.npy (30,Nv,K) f32   per-class neighbours (train row indices)
  knn_pool_idx.npy (Nv,K) i32      knn_pool_sim.npy (Nv,K) f32       pooled-embedding neighbours
  proto_score.npy (Nv,30) f32
  emb_{train,val}_h.npy (30,N,768) f16, emb_{train,val}_pool.npy (N,768) f16   (unless --no-save-emb)
  meta.json

    sbatch scripts/memory_probe.sh --tag asl_seeded train/checkpoint_ASL/v0_768_seeded/seed_42/Model_run/model_swa.pth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "evaluate"))
import evaluate_tta as base  # noqa: E402  (decode/resize, the eval transform)
import evaluate_tta_v5 as v5e  # noqa: E402  (load_model: ConvNeXt2 / prior / MoE by state dict)
import convnext  # noqa: E402
import webdataset as wds  # noqa: E402
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision.transforms.functional import normalize  # noqa: E402

PADCHEST_CSV = ROOT / "data/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
C, D = 30, 768


def decode(sample):
    out = base.decode_and_resize(sample)
    if out is None:
        return None
    xs, y = out
    return xs[str(base._LOAD_SIZES[0])], y, sample["fname"].decode()


def loader(shard_dir: str, batch_size: int, num_workers: int) -> DataLoader:
    uris = ["file:" + p.resolve().as_posix() for p in sorted(Path(shard_dir).glob("shard_*.tar"))]
    ds = wds.WebDataset(uris, shardshuffle=False, empty_check=False).map(decode).select(lambda x: x is not None)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


@torch.no_grad()
def extract(model, shard_dir, nmax, args, device):
    """One pass: class-major h (C, nmax, D) f16 and pooled (nmax, D) f16 on the GPU; logits, labels, keys on CPU."""
    H = torch.zeros((C, nmax, D), dtype=torch.float16, device=device)
    P = torch.zeros((nmax, D), dtype=torch.float16, device=device)
    logits, labels, keys, n, t0 = [], [], [], 0, time.time()
    for b, (x, y, fn) in enumerate(loader(shard_dir, args.batch_size, args.num_workers)):
        if args.limit_batches and b >= args.limit_batches:
            break
        x = normalize(x.float().to(device, non_blocking=True), IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
        h = model.head.last_h                                   # (B, C, D), the tensor the readout reads
        B = h.size(0)
        if n + B > nmax:
            raise RuntimeError(f"more than nmax={nmax} images in {shard_dir}")
        H[:, n:n + B] = h.transpose(0, 1).half()
        P[n:n + B] = h.float().mean(1).half()                  # the Triplet Memory's embedding, before normalize
        logits.append((out[0] if isinstance(out, tuple) else out).float().cpu())
        labels.append((y > 0.5).to(torch.uint8))
        keys += list(fn)
        n += B
        if (b + 1) % 500 == 0:
            print(f"  {shard_dir}: {n} images, {n / (time.time() - t0):.1f} img/s", flush=True)
    assert len(set(keys)) == n, "duplicate keys: shards were read twice"
    return H, P, torch.cat(logits).numpy(), torch.cat(labels).numpy(), keys, n


@torch.no_grad()
def topk_excluding(V, T, pid_v, pid_t, k, chunk):
    """Top-k cosine neighbours of each row of V (unit rows) among the rows of T, excluding equal patient ids."""
    idx = np.zeros((V.size(0), k), np.int32)
    sim = np.zeros((V.size(0), k), np.float32)
    for s in range(0, V.size(0), chunk):
        S = V[s:s + chunk] @ T.T                                # (chunk, Nt) f32
        S.masked_fill_(pid_v[s:s + chunk, None] == pid_t[None, :], -2.0)
        v, i = S.topk(k, dim=1)
        idx[s:s + chunk], sim[s:s + chunk] = i.int().cpu().numpy(), v.cpu().numpy()
    return idx, sim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="/data/psytp7/memprobe")
    ap.add_argument("--img-size", type=int, default=768)
    ap.add_argument("--interp", choices=list(base.INTERP), default="area")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--train-shards-dir", default="/data/psytp7/wds_shards_train_raw")
    ap.add_argument("--val-shards-dir", default="/data/psytp7/wds_shards_val_raw")
    ap.add_argument("--train-nmax", type=int, default=106000)
    ap.add_argument("--val-nmax", type=int, default=18000)
    ap.add_argument("--limit-batches", type=int, default=0, help="smoke test: batches per split")
    ap.add_argument("--no-save-emb", dest="save_emb", action="store_false", default=True)
    args = ap.parse_args()

    base._LOAD_SIZES, base._INTERP = (args.img_size,), args.interp
    device = torch.device("cuda")
    out = Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"convnext module: {convnext.__file__} | out {out}", flush=True)
    model = v5e.load_model(args.checkpoint, device)
    if v5e.is_prior_model(model):
        raise ValueError("prior checkpoints need a prior input; this probe is for ASL / Entropy checkpoints")

    t0 = time.time()
    Hv, Pv, lv, yv, kv, nv = extract(model, args.val_shards_dir, args.val_nmax, args, device)
    Ht, Pt, lt, yt, kt, nt = extract(model, args.train_shards_dir, args.train_nmax, args, device)
    print(f"extracted val {nv} / train {nt} in {(time.time() - t0) / 60:.1f} min", flush=True)
    del model
    torch.cuda.empty_cache()
    assert not set(kv) & set(kt), "train and val share images"

    pid = pd.read_csv(PADCHEST_CSV, usecols=["ImageID", "PatientID"], dtype=str, low_memory=False) \
        .drop_duplicates("ImageID").set_index("ImageID")["PatientID"]
    codes = {p: i for i, p in enumerate(pd.unique(pid.dropna()))}
    pv_ids = [codes.get(p, -1 - i) for i, p in enumerate(pid.reindex(kv))]            # unknown -> unique id
    pt_ids = [codes.get(p, -10_000_000 - i) for i, p in enumerate(pid.reindex(kt))]
    pid_v = torch.as_tensor(pv_ids, dtype=torch.long, device=device)
    pid_t = torch.as_tensor(pt_ids, dtype=torch.long, device=device)

    t0 = time.time()
    k_idx = np.zeros((C, nv, args.k), np.int32)
    k_sim = np.zeros((C, nv, args.k), np.float32)
    proto = np.zeros((nv, C), np.float32)
    ytt = torch.as_tensor(yt, device=device).bool()
    for c in range(C):
        T = F.normalize(Ht[c, :nt].float(), dim=1)
        V = F.normalize(Hv[c, :nv].float(), dim=1)
        k_idx[c], k_sim[c] = topk_excluding(V, T, pid_v, pid_t, args.k, args.chunk)
        mu_pos = F.normalize(T[ytt[:, c]].mean(0), dim=0)
        mu_neg = F.normalize(T[~ytt[:, c]].mean(0), dim=0)
        proto[:, c] = (V @ mu_pos - V @ mu_neg).cpu().numpy()
        del T, V
    Tp = F.normalize(Pt[:nt].float(), dim=1)
    Vp = F.normalize(Pv[:nv].float(), dim=1)
    p_idx, p_sim = topk_excluding(Vp, Tp, pid_v, pid_t, args.k, args.chunk)
    print(f"kNN done in {(time.time() - t0) / 60:.1f} min", flush=True)

    (out / "val_keys.txt").write_text("\n".join(kv))
    (out / "train_keys.txt").write_text("\n".join(kt))
    for name, arr in (("val_labels", yv), ("train_labels", yt), ("val_logits", lv), ("train_logits", lt),
                      ("knn_class_idx", k_idx), ("knn_class_sim", k_sim), ("knn_pool_idx", p_idx),
                      ("knn_pool_sim", p_sim), ("proto_score", proto)):
        np.save(out / f"{name}.npy", arr)
    if args.save_emb:
        for name, H, n in (("emb_val_h", Hv, nv), ("emb_train_h", Ht, nt)):
            mm = np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=np.float16, shape=(C, n, D))
            for c in range(C):
                mm[c] = H[c, :n].cpu().numpy()
            mm.flush()
            del mm
        np.save(out / "emb_val_pool.npy", Pv[:nv].cpu().numpy())
        np.save(out / "emb_train_pool.npy", Pt[:nt].cpu().numpy())
    train_patients = set(pt_ids)
    same_patient_share = float(np.mean([p >= 0 and p in train_patients for p in pv_ids]))
    meta = {"checkpoint": str(Path(args.checkpoint).resolve()), "tag": args.tag, "img_size": args.img_size,
            "interp": args.interp, "tta": False, "k": args.k, "n_val": nv, "n_train": nt,
            "val_share_with_train_patient": same_patient_share, "excluded": "same PatientID (PadChest CSV)",
            "convnext": convnext.__file__, "limit_batches": args.limit_batches,
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
