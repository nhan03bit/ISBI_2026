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

# Sizes the loader must produce; each is an independent LANCZOS resize of the
# decoded original, i.e. exactly train_2_v3.deterministic_val_resize at that
# size (set in main from the union of all per-checkpoint scales).
_LOAD_SIZES = (512,)


def decode_and_resize(sample):
    try:
        img_bytes = sample["img"]
        y = torch.load(io.BytesIO(sample["cls"]))
        img_np = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img_np is None:
            return None
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        xs = {}
        for s in _LOAD_SIZES:
            r = cv2.resize(img_np, (s, s), interpolation=cv2.INTER_LANCZOS4)
            xs[str(s)] = torch.from_numpy(np.ascontiguousarray(r)).permute(2, 0, 1).float() / 255.0
        return xs, y
    except Exception as e:  # noqa: BLE001
        print("decode error:", e)
        return None


def make_val_loader(val_dir, batch_size, num_workers):
    shards = sorted(Path(val_dir).glob("shard_*.tar"))
    uris = ["file:" + p.resolve().as_posix() for p in shards]
    logger.info(f"Val shards: {len(uris)} | load_sizes={list(_LOAD_SIZES)}")
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
def model_output(model, xs, tta_flip, scales, logit_avg):
    """One model's mean output over {hflip} x {scales}, in logit space if
    logit_avg else probability space. xs maps str(size) -> (B,3,size,size),
    imagenet-normalized."""
    terms = []
    for s in scales:
        x = xs[str(s)]
        for v in [x] + ([torch.flip(x, dims=[3])] if tta_flip else []):
            lg = _fwd(model, v)
            terms.append(lg if logit_avg else torch.sigmoid(lg))
    return torch.stack(terms).mean(0)


def parse_ckpt_spec(spec, default_scales):
    """'path' -> (path, default_scales); 'path@768' or 'path@768,896' -> (path, [768, 896])."""
    if "@" in spec:
        path, sc = spec.rsplit("@", 1)
        return path, [int(s) for s in sc.split(",") if s.strip()]
    return spec, list(default_scales)


@torch.no_grad()
def run(specs, val_dir, batch_size, num_workers, tta_flip, logit_avg, dump_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(p, device) for p, _ in specs]

    C = 30
    m_map = MultilabelAveragePrecision(num_labels=C, average="macro").to(device)
    m_auc = MultilabelAUROC(num_labels=C, average="macro").to(device)
    m_f1 = MultilabelF1Score(num_labels=C, average="macro", threshold=0.5).to(device)
    m_ece = BinaryCalibrationError(n_bins=15, norm="l1").to(device)

    per_model = [[] for _ in specs]   # --dump-probs: per-model probabilities
    labels = []

    loader = make_val_loader(val_dir, batch_size, num_workers)
    for xs, y in loader:
        xs = {k: normalize(v.float().to(device, non_blocking=True), IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
              for k, v in xs.items()}
        y_int = (y.to(device) > 0.5).int()

        acc = None
        for i, (model, (_, scales)) in enumerate(zip(models, specs)):
            o = model_output(model, xs, tta_flip, scales, logit_avg)
            if dump_dir is not None:
                per_model[i].append((torch.sigmoid(o) if logit_avg else o).cpu())
            acc = o if acc is None else acc + o
        acc = acc / len(models)
        probs = torch.sigmoid(acc) if logit_avg else acc
        if dump_dir is not None:
            labels.append(y_int.cpu())

        m_map.update(probs, y_int)
        m_auc.update(probs, y_int)
        m_f1.update(probs, y_int)
        m_ece.update(probs.reshape(-1), y_int.reshape(-1))

    if dump_dir is not None:
        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        np.save(dump_dir / "labels.npy", torch.cat(labels).numpy().astype(np.uint8))
        manifest = []
        for (path, scales), chunks in zip(specs, per_model):
            rp = Path(path).resolve()
            # e.g. res_sweep__img768_seed86__model_best@768.npy
            name = "__".join(rp.parts[-4:-2] + (rp.stem,)).replace("/", "_")
            name = f"{name}@{'-'.join(map(str, scales))}{'_flip' if tta_flip else ''}.npy"
            np.save(dump_dir / name, torch.cat(chunks).numpy().astype(np.float32))
            manifest.append({"file": name, "checkpoint": str(rp), "scales": scales, "tta_flip": tta_flip})
        with open(dump_dir / f"manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"dumped per-model probs for {len(specs)} model(s) to {dump_dir}")

    return {
        "val_mAP": float(m_map.compute().item()),
        "val_mAUC": float(m_auc.compute().item()),
        "val_mF1": float(m_f1.compute().item()),
        "val_mECE": float(m_ece.compute().item()),
    }


def parse_args():
    ap = argparse.ArgumentParser(description="TTA (hflip) + multi-checkpoint ensemble evaluation")
    ap.add_argument("checkpoints", nargs="+",
                    help="one or more model_best.pth paths; probs are averaged. Append "
                         "@SIZE[,SIZE...] to evaluate that checkpoint at its own input "
                         "size(s), e.g. .../img768_seed86/Model_run/model_best.pth@768 - "
                         "otherwise --scales applies.")
    ap.add_argument("--val-shards-dir", default="/data/psytp7/wds_shards_val_raw")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-tta", dest="tta_flip", action="store_false", default=True,
                    help="disable horizontal-flip TTA (note: a few CXR-LT classes are "
                         "side-specific, e.g. azygos lobe - flip TTA still helps macro mAP "
                         "in practice but this lets you check).")
    ap.add_argument("--scales", default="512",
                    help="default comma-separated square input sizes to average over for "
                         "checkpoints given without @SIZE, e.g. 512,640,768. Every size is "
                         "an independent LANCZOS resize of the original (= training val "
                         "transform). Default 512 (= original behaviour).")
    ap.add_argument("--logit-avg", action="store_true",
                    help="average in logit space (mean of logits, then one sigmoid) "
                         "instead of probability space. Often better for ensembles.")
    ap.add_argument("--dump-probs", default=None, metavar="DIR",
                    help="also save each checkpoint's val probabilities (after its own "
                         "TTA) + labels.npy to DIR, for offline ensemble selection "
                         "(analysis/ens_select.py).")
    ap.add_argument("--out-json", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    global _LOAD_SIZES
    default_scales = [int(s) for s in args.scales.split(",") if s.strip()]
    specs = [parse_ckpt_spec(c, default_scales) for c in args.checkpoints]
    _LOAD_SIZES = tuple(sorted({s for _, sc in specs for s in sc}))

    res = run(specs, args.val_shards_dir, args.batch_size, args.num_workers,
              args.tta_flip, args.logit_avg, args.dump_probs)
    res.update({
        "checkpoints": [str(Path(p).resolve()) for p, _ in specs],
        "scales_per_checkpoint": [sc for _, sc in specs],
        "n_models": len(specs),
        "tta_flip": args.tta_flip,
        "scales": list(_LOAD_SIZES),
        "logit_avg": args.logit_avg,
        "evaluated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    })
    print(json.dumps(res, indent=2))

    out = Path(args.out_json) if args.out_json else (
        Path(specs[0][0]).resolve().parent / "eval_tta_result.json"
        if len(specs) == 1 else Path("eval_tta_ensemble_result.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
