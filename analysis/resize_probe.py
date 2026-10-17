"""Eval-only probe: does the train/val resampling mismatch cost mAP?

train_2_v0.py / train_2_v5.py downscale the ~3000 px originals to --img-size with cv2.INTER_LINEAR inside
deterministic_random_resized_crop (no antialiasing at a ~4x downscale), while validation (deterministic_val_resize)
and evaluate/evaluate_tta.py use cv2.INTER_LANCZOS4. This scores one checkpoint on the internal val shards with
each image decoded once and resized three ways: lanczos (current eval), linear (the training filter, no crop)
and area (antialiased). No TTA, fp32, eval mode. Writes analysis/out/resize_probe/<tag>.json.

Run from evaluate/:  python ../analysis/resize_probe.py --ckpt TAG=PATH [--ckpt ...]
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import webdataset as wds
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryCalibrationError, MultilabelAUROC, MultilabelAveragePrecision
from torchvision.transforms.functional import normalize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluate"))
sys.path.insert(0, os.path.join(ROOT, "train"))
import evaluate_tta_v5 as E  # noqa: E402

METHODS = {"lanczos": cv2.INTER_LANCZOS4, "linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA}
SIZE = 768


def decode(sample):
    try:
        y = torch.load(io.BytesIO(sample["cls"]))
        img = cv2.imdecode(np.frombuffer(sample["img"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        xs = {m: torch.from_numpy(np.ascontiguousarray(cv2.resize(img, (SIZE, SIZE), interpolation=f)))
              .permute(2, 0, 1).float() / 255.0 for m, f in METHODS.items()}
        return xs, y, sample["fname"].decode()
    except Exception as e:  # noqa: BLE001
        print("decode error:", e, flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="tag=path")
    ap.add_argument("--val-dir", default="/data/psytp7/wds_shards_val_raw")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "analysis/out/resize_probe"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many batches (smoke)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")
    specs = [s.split("=", 1) for s in args.ckpt]
    models = {tag: E.load_model(path if os.path.isabs(path) else os.path.join(ROOT, path), device) for tag, path in specs}
    shards = sorted(Path(args.val_dir).glob("shard_*.tar"))
    ds = wds.WebDataset(["file:" + p.resolve().as_posix() for p in shards], shardshuffle=False,
                        empty_check=False).map(decode).select(lambda x: x is not None)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    probs = {(t, m): [] for t in models for m in METHODS}
    labels, keys = [], []
    t0 = time.time()
    with torch.no_grad():
        for b, (xs, y, fn) in enumerate(loader, 1):
            if args.limit and b > args.limit:
                break
            for m in METHODS:
                x = normalize(xs[m].to(device, non_blocking=True), IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
                for t, model in models.items():
                    out = model(x)
                    probs[(t, m)].append(torch.sigmoid((out[0] if isinstance(out, tuple) else out).float()).cpu())
            labels.append((y > 0.5).int()); keys.extend(fn)
            if b % 200 == 0:
                print(f"{b} batches ({time.time() - t0:.0f}s)", flush=True)
    Y = torch.cat(labels)
    for t in models:
        res = {"tag": t, "ckpt": dict(specs)[t], "n_rows": len(keys), "tta": "none", "size": SIZE}
        for m in METHODS:
            P = torch.cat(probs[(t, m)])
            mp = MultilabelAveragePrecision(num_labels=30, average="none")
            per = mp(P, Y)
            res[m] = {"mAP": float(per.mean()),
                      "mAUC": float(MultilabelAUROC(num_labels=30, average="macro")(P, Y)),
                      "mECE": float(BinaryCalibrationError(n_bins=15, norm="l1")(P.reshape(-1), Y.reshape(-1))),
                      "per_class_AP": [float(v) for v in per]}
            np.save(os.path.join(args.out_dir, f"{t}_{m}.npy"), P.numpy().astype(np.float32))
        with open(os.path.join(args.out_dir, f"{t}.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(t, json.dumps({m: {k: round(v, 4) for k, v in res[m].items() if k != "per_class_AP"} for m in METHODS}),
              flush=True)
    with open(os.path.join(args.out_dir, "keys.txt"), "w") as f:
        f.write("\n".join(keys))
    np.save(os.path.join(args.out_dir, "labels.npy"), Y.numpy().astype(np.uint8))


if __name__ == "__main__":
    main()
