"""Task-1 image-shift probe on internal val (2026-10-17, GPU).

The CXR-LT Task-1 PNGs are 8-bit RGB, and a share of them carry independent noise in each channel (measured by
/data/psytp7/isbi_task1/tools/noise_probe.py), while every training and val image is grayscale replicated to 3
identical channels. Task-1 has no labels here, so this asks the question on internal val, before any choice is
made for the Task-1 predictions: how much does such noise cost, and does averaging the channels
(evaluate/predict_task1.py --channel-mean) recover it?

Conditions per val image (decoded the eval's way: IMREAD_COLOR 8-bit, 3 identical channels), one noise field per
image seeded by its key so every condition sees the same draw:
  clean          as is
  rgbnoise       + N(0, sigma) independently per channel and pixel at full resolution, clipped, rounded to uint8
  rgbnoise_mean  rgbnoise, then the equal-weight channel mean replicated to 3 channels
  graynoise      + the first channel's N(0, sigma) field in all 3 channels (noise that a channel mean cannot remove)
Then the eval transform (area resize to --img-size, /255, ImageNet normalisation), flip TTA, the checkpoint's
probabilities. Reports mAP per condition (all views and frontal) and paired image-bootstrap differences.

    sbatch scripts/task1_shift_probe.sh --sigma 20 --limit 6000 \
        train/checkpoint_Triplet_3/unified768/c/seed_42/Model_run/model_swa.pth
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zlib
from datetime import datetime
from pathlib import Path

import cv2  # type: ignore
import numpy as np
import pandas as pd
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader
from torchvision.transforms.functional import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "evaluate"))
sys.path.insert(0, str(ROOT / "analysis"))
import evaluate_tta as base  # noqa: E402  (resize_square)
import evaluate_tta_v5 as v5e  # noqa: E402  (load_model, model_output)
import convnext  # noqa: E402
import entropy_ablation as ea  # noqa: E402  (FastAP)
import webdataset as wds  # noqa: E402

assert Path(convnext.__file__).resolve() == (ROOT / "train/convnext.py").resolve(), convnext.__file__

CONDS = ("clean", "rgbnoise", "rgbnoise_mean", "graynoise")
PADCHEST_CSV = ROOT / "data/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
FRONTAL = ("PA", "AP", "AP_horizontal")
ARGS = None  # set in main before the loader forks


def perturb(sample):
    try:
        img = cv2.imdecode(np.frombuffer(sample["img"], np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        y = torch.load(io.BytesIO(sample["cls"]))
        fname = sample["fname"].decode()
        rng = np.random.default_rng(zlib.crc32(fname.encode()))
        noise = rng.normal(0.0, ARGS.sigma, img.shape).astype(np.float32)
        f = img.astype(np.float32)
        variants = {
            "clean": img,
            "rgbnoise": np.clip(np.rint(f + noise), 0, 255).astype(np.uint8),
            "graynoise": np.clip(np.rint(f + noise[..., :1]), 0, 255).astype(np.uint8),
        }
        variants["rgbnoise_mean"] = np.repeat(np.rint(variants["rgbnoise"].mean(2, dtype=np.float32))[..., None],
                                              3, 2).astype(np.uint8)
        xs = {}
        for c in CONDS:
            r = base.resize_square(cv2.cvtColor(variants[c], cv2.COLOR_BGR2RGB), ARGS.img_size, ARGS.interp)
            xs[c] = torch.from_numpy(np.ascontiguousarray(r)).permute(2, 0, 1).float() / 255.0
        return xs, y, fname
    except Exception as e:  # noqa: BLE001
        print("decode error:", e, flush=True)
        return None


def main() -> None:
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--sigma", type=float, required=True, help="per-pixel noise s.d. in 8-bit units, full resolution")
    ap.add_argument("--img-size", type=int, default=768)
    ap.add_argument("--interp", choices=list(base.INTERP), default="area")
    ap.add_argument("--limit", type=int, default=6000, help="first N val images in shard order")
    ap.add_argument("--val-shards-dir", default="/data/psytp7/wds_shards_val_raw")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=ROOT / "analysis/out/task1_shift_probe.json")
    ARGS = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = v5e.load_model(ARGS.checkpoint, device)
    uris = ["file:" + p.resolve().as_posix() for p in sorted(Path(ARGS.val_shards_dir).glob("shard_*.tar"))]
    ds = wds.WebDataset(uris, shardshuffle=False, empty_check=False).map(perturb).select(lambda x: x is not None)
    loader = DataLoader(ds, batch_size=ARGS.batch_size, num_workers=ARGS.num_workers, pin_memory=True)

    probs = {c: [] for c in CONDS}
    labels, keys, n = [], [], 0
    with torch.no_grad():
        for b, (xs, y, fn) in enumerate(loader):
            for c in CONDS:
                x = normalize(xs[c].to(device, non_blocking=True), IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
                probs[c].append(v5e.model_output(model, {str(ARGS.img_size): x}, True, [ARGS.img_size], False, None).cpu())
            labels.append((y > 0.5).int())
            keys += list(fn)
            n += len(fn)
            if b % 200 == 0:
                print(f"batch {b} | images {n}", flush=True)
            if n >= ARGS.limit:
                break
    P = {c: torch.cat(v).numpy().astype(np.float64)[: ARGS.limit] for c, v in probs.items()}
    L = torch.cat(labels).numpy().astype(np.int64)[: ARGS.limit]
    keys = keys[: ARGS.limit]
    pc = pd.read_csv(PADCHEST_CSV, usecols=["ImageID", "Projection"], dtype=str,
                     low_memory=False).drop_duplicates("ImageID").set_index("ImageID")
    fr = pc["Projection"].reindex(keys).isin(FRONTAL).values.astype(float)

    nimg = len(keys)
    ones = np.ones(nimg)
    fa = {c: ea.FastAP(P[c], L) for c in CONDS}
    res = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "checkpoint": ARGS.checkpoint,
           "sigma": ARGS.sigma, "img_size": ARGS.img_size, "interp": ARGS.interp, "n": nimg,
           "n_frontal": int(fr.sum()), "mAP": {}, "frontal": {}, "diff": {}}
    for c in CONDS:
        res["mAP"][c] = float(np.nanmean(fa[c](ones)[0]))
        res["frontal"][c] = float(np.nanmean(fa[c](fr)[0]))
    rng = np.random.default_rng(0)
    pairs = [("rgbnoise", "clean"), ("rgbnoise_mean", "rgbnoise"), ("rgbnoise_mean", "clean"), ("graynoise", "clean")]
    bd = {p: [] for p in pairs}
    for _ in range(ARGS.boot):
        w = np.bincount(rng.integers(0, nimg, nimg), minlength=nimg).astype(float)
        m = {c: np.nanmean(fa[c](w)[0]) for c in CONDS}
        for a, b in pairs:
            bd[(a, b)].append(m[a] - m[b])
    print(f"\n{ARGS.checkpoint} | sigma {ARGS.sigma} | n {nimg} (frontal {int(fr.sum())}) | {ARGS.interp} {ARGS.img_size}")
    for c in CONDS:
        print(f"  {c:14s} mAP {res['mAP'][c]:.4f} | frontal {res['frontal'][c]:.4f}")
    for a, b in pairs:
        d = res["mAP"][a] - res["mAP"][b]
        ci = np.percentile(bd[(a, b)], [2.5, 97.5]).tolist()
        res["diff"][f"{a} - {b}"] = {"d": d, "ci": ci, "d_frontal": res["frontal"][a] - res["frontal"][b]}
        print(f"  {a:14s} - {b:9s} {d:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | frontal {res['frontal'][a] - res['frontal'][b]:+.4f}")
    ARGS.out.parent.mkdir(parents=True, exist_ok=True)
    out = ARGS.out.with_name(f"{ARGS.out.stem}_sigma{ARGS.sigma:g}{ARGS.out.suffix}")
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
