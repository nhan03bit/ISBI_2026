import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import io
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import webdataset as wds
from PIL import ImageFile

from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelAUROC,
    MultilabelF1Score,
)
from torchmetrics.classification import BinaryCalibrationError

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
from convnext import ConvNeXt2

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SIZE = 512
_LOAD_SIZE = 512  # loader resize; scales are GPU-downsampled from this (set in main)


def decode_and_resize(sample):
    try:
        img_bytes = sample["img"]
        y = torch.load(io.BytesIO(sample["cls"]))
        img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_np is None:
            return None
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_np = cv2.resize(img_np, (_LOAD_SIZE, _LOAD_SIZE), interpolation=cv2.INTER_LANCZOS4)
        x = torch.from_numpy(np.ascontiguousarray(img_np)).permute(2, 0, 1).float() / 255.0
        return x, y
    except Exception as e:  # noqa: BLE001
        print("decode error:", e)
        return None


def make_val_loader(val_dir, batch_size, num_workers):
    shards = sorted(Path(val_dir).glob("shard_*.tar"))
    uris = ["file:" + p.resolve().as_posix() for p in shards]
    logger.info(f"Val shards: {len(uris)} | load_size={_LOAD_SIZE}")
    ds = (
        wds.WebDataset(uris, shardshuffle=False, empty_check=False)
        .map(decode_and_resize)
        .select(lambda x: x is not None)
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


def load_model(ckpt_path, device):
    model = ConvNeXt2(depths=[3, 3, 27, 3, 2], dims=[128, 256, 512, 1024, 1024], num_classes=30)
    sd = torch.load(ckpt_path, map_location="cpu")
    res = model.load_state_dict(sd, strict=False)
    logger.info(f"loaded {ckpt_path} (missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)})")
    return model.to(device).eval()


def _fwd(model, x):
    out = model(x)
    return (out[0] if isinstance(out, tuple) else out).float()


@torch.no_grad()
def model_output(model, x, tta_flip, scales, logit_avg):
    """One model's mean output over {hflip} x {scales}, in logit space if
    logit_avg else probability space. x is (B,3,load,load), imagenet-normalized."""
    views = [x] + ([torch.flip(x, dims=[3])] if tta_flip else [])
    terms = []
    for v in views:
        for s in scales:
            vs = v if v.shape[-1] == s else F.interpolate(
                v, size=(s, s), mode="bilinear", align_corners=False)
            lg = _fwd(model, vs)
            terms.append(lg if logit_avg else torch.sigmoid(lg))
    return torch.stack(terms).mean(0)


@torch.no_grad()
def run(checkpoints, val_dir, batch_size, num_workers, tta_flip, scales, logit_avg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(c, device) for c in checkpoints]

    C = 30
    m_map = MultilabelAveragePrecision(num_labels=C, average="macro").to(device)
    m_auc = MultilabelAUROC(num_labels=C, average="macro").to(device)
    m_f1 = MultilabelF1Score(num_labels=C, average="macro", threshold=0.5).to(device)
    m_ece = BinaryCalibrationError(n_bins=15, norm="l1").to(device)

    loader = make_val_loader(val_dir, batch_size, num_workers)
    for x, y in loader:
        x = x.float().to(device, non_blocking=True)
        x = normalize(x, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        y_int = (y.to(device) > 0.5).int()

        acc = None
        for model in models:
            o = model_output(model, x, tta_flip, scales, logit_avg)
            acc = o if acc is None else acc + o
        acc = acc / len(models)
        probs = torch.sigmoid(acc) if logit_avg else acc

        m_map.update(probs, y_int)
        m_auc.update(probs, y_int)
        m_f1.update(probs, y_int)
        m_ece.update(probs.reshape(-1), y_int.reshape(-1))

    return {
        "val_mAP": float(m_map.compute().item()),
        "val_mAUC": float(m_auc.compute().item()),
        "val_mF1": float(m_f1.compute().item()),
        "val_mECE": float(m_ece.compute().item()),
    }


def parse_args():
    ap = argparse.ArgumentParser(description="TTA (hflip) + multi-checkpoint ensemble evaluation")
    ap.add_argument("checkpoints", nargs="+", help="one or more model_best.pth paths; probs are averaged")
    ap.add_argument("--val-shards-dir", default="/data/psytp7/wds_shards_val_raw")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-tta", dest="tta_flip", action="store_false", default=True,
                    help="disable horizontal-flip TTA (note: a few CXR-LT classes are "
                         "side-specific, e.g. azygos lobe - flip TTA still helps macro mAP "
                         "in practice but this lets you check).")
    ap.add_argument("--scales", default="512",
                    help="comma-separated square input sizes to average over, e.g. "
                         "512,640,768. Loader resizes to max(scale, 512) with LANCZOS; "
                         "smaller scales are bilinear-downsampled on GPU. Default 512 "
                         "(= original behaviour).")
    ap.add_argument("--logit-avg", action="store_true",
                    help="average in logit space (mean of logits, then one sigmoid) "
                         "instead of probability space. Often better for ensembles.")
    ap.add_argument("--out-json", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    global _LOAD_SIZE
    scales = [int(s) for s in args.scales.split(",") if s.strip()]
    _LOAD_SIZE = max(scales + [512])

    res = run(args.checkpoints, args.val_shards_dir, args.batch_size, args.num_workers,
              args.tta_flip, scales, args.logit_avg)
    res.update({
        "checkpoints": [str(Path(c).resolve()) for c in args.checkpoints],
        "n_models": len(args.checkpoints),
        "tta_flip": args.tta_flip,
        "scales": scales,
        "logit_avg": args.logit_avg,
        "evaluated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    })
    print(json.dumps(res, indent=2))

    out = Path(args.out_json) if args.out_json else (
        Path(args.checkpoints[0]).resolve().parent / "eval_tta_result.json"
        if len(args.checkpoints) == 1 else Path("eval_tta_ensemble_result.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
