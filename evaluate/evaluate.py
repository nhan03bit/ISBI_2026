import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import io
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch

print(torch.__version__)
print(torch.version.cuda)
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import webdataset as wds

from torchvision.transforms.functional import normalize, rgb_to_grayscale
import torchvision.transforms.functional as TF
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random
import numpy as np
from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelAUROC,
    MultilabelF1Score,
)
from torchmetrics.classification import BinaryCalibrationError

# convnext.py/ml_decoder.py aren't in this directory - train/'s copy is the one
# whose ConvNeXt2.forward returns (logits, embeddings), which this script's
# evaluate() loop expects.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "train"))
sys.path.insert(0, _THIS_DIR)  # so `import class_groups` works regardless of cwd
from convnext import ConvNeXt2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import albumentations as A  # type: ignore
import cv2  # type: ignore
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform(size=224):
    return A.Compose([
        A.RandomResizedCrop(size=(size, size), scale=(0.9, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            p=0.5
        ),
        A.OneOf([
            A.OpticalDistortion(),
            A.GridDistortion(),
            A.ElasticTransform(),
        ], p=0.2),

        A.RandomBrightnessContrast(p=0.5),
        A.OneOf([
            A.GaussNoise(),
            A.GaussianBlur(),
            A.MotionBlur(),
            A.MedianBlur(),
        ], p=0.2),
        # A.Compose([A.Resize(size, size, interpolation=cv2.INTER_LANCZOS4)])
    ])


def val_transform(size=224):
    return A.Compose([A.Resize(size, size, interpolation=cv2.INTER_LANCZOS4)])


train_aug = train_transform(512)
val_aug = val_transform(512)  # rebound in main() from --img-size


def set_val_img_size(size: int):
    """Rebind the module-level val transform. Call before building the dataset."""
    global val_aug
    val_aug = val_transform(size)
    logger.info(f"val transform resize = {size}x{size}")


def decode_and_transform(sample, is_train=True):
    try:
        img_bytes = sample["img"]
        y = torch.load(io.BytesIO(sample["cls"]))

        # Decode using OpenCV (more tolerant than PIL)
        img_np = cv2.imdecode(
            np.frombuffer(img_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        if img_np is None:
            pass

        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        if is_train:
            out = train_aug(image=img_np)["image"]
        else:
            out = val_aug(image=img_np)["image"]

        x = torch.from_numpy(out).permute(2, 0, 1).float() / 255.0
        return x, y

    except Exception as e:
        return None


def load_model(checkpoint_path):
    model = ConvNeXt2(depths=[3, 3, 27, 3, 2],
                      dims=[128, 256, 512, 1024, 1024],
                      num_classes=30)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint and all(
        not k.startswith(("downsample_layers", "stages", "head", "pos_encoding"))
        for k in list(checkpoint.keys())[:5]
    ):
        checkpoint = checkpoint["model"]
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    # A genuine architecture mismatch would show up as many missing backbone keys.
    real_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    if real_missing:
        logger.warning(f"{len(real_missing)} missing state_dict keys, e.g. {real_missing[:8]}")
    if unexpected:
        logger.warning(f"{len(unexpected)} unexpected state_dict keys, e.g. {unexpected[:8]}")
    model.eval()
    return model


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8,
                 disable_torch_grad_focal_loss=True, reduction="mean"):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.reduction = reduction

    def forward(self, x, y):
        y = y.float()

        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg  # log-likelihood

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    pt = xs_pos * y + xs_neg * (1.0 - y)
                    gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
                    w = (1.0 - pt).pow(gamma)
            else:
                pt = xs_pos * y + xs_neg * (1.0 - y)
                gamma = self.gamma_pos * y + self.gamma_neg * (1.0 - y)
                w = (1.0 - pt).pow(gamma)

            loss = loss * w

        loss = -loss  # convert to positive loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 0.0):
        assert mode in ("min", "max")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Return True if improved, False otherwise."""
        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            return True

        improved = (value > self.best + self.min_delta) if self.mode == "max" else (value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.bad_epochs = 0
            return True

        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def preprocess_batch(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, 3, H, W] (RGB tensor)
    returns: [B, 3, H, W] grayscale replicated to 3 channels
    """
    # # Convert RGB → grayscale (keeps channel dim)
    # x = rgb_to_grayscale(x, num_output_channels=1)  # [B,1,H,W]

    # Normalize
    x = normalize(x, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    return x

class EmbeddingClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.net(x)

import torch.nn.functional as F

class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        # self.model = create_model(cfg["num_classes"]).to(self.device)

        #self.optim = AdamW(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        #self.sched = CosineAnnealingLR(self.optim, T_max=cfg["epochs"])
        #self.crit = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.05, reduction="mean")
        #self.crit_triplet = nn.TripletMarginLoss(margin=1.0, p=2)

        C = cfg["num_classes"]
        self.map = MultilabelAveragePrecision(num_labels=C).to(self.device)
        self.auc = MultilabelAUROC(num_labels=C).to(self.device)
        self.f1 = MultilabelF1Score(num_labels=C).to(self.device)
        self.ece = BinaryCalibrationError(n_bins=15, norm="l1").to(self.device)

        self.history = {
            "train_loss": [], "train_triplet_loss": [], "val_loss": [], "lr": [],
            "train_mAP": [], "train_mAUC": [], "train_mF1": [], "train_mECE": [],
            "val_mAP": [], "val_mAUC": [], "val_mF1": [], "val_mECE": [],
            "best_epoch": None, "best_triplet_epoch": None, "best_metric": None, "best_triplet_metric": None,
        }

        # Early stopping config
        # self.monitor = cfg.get("monitor", "mAP")  # one of: "mAP", "mAUC", "mF1", "mECE", "val_loss"
        # self.patience = cfg.get("patience", 10)

        # if self.monitor == "val_loss":
        #     self.es = EarlyStopping(patience=self.patience, mode="min", min_delta=cfg.get("min_delta", 0.0))
        # elif self.monitor == "train_loss":
        #     self.es = EarlyStopping(patience=self.patience, mode="min", min_delta=cfg.get("min_delta", 0.0))
        # elif self.monitor == "mECE":
        #     self.es = EarlyStopping(patience=self.patience, mode="min", min_delta=cfg.get("min_delta", 0.0))
        # else:
        #     self.es = EarlyStopping(patience=self.patience, mode="max", min_delta=cfg.get("min_delta", 0.0))


    @torch.no_grad()
    def evaluate(self, loader, checkpoint_path):
        # Use trainer device
        device = self.device
        C = self.cfg["num_classes"]

        # Load model and move to device
        self.model = load_model(checkpoint_path)
        self.model = self.model.to(device)
        self.model.eval()

        # Metrics (macro, kept for backwards compatibility with old eval_result.json)
        map_metric = MultilabelAveragePrecision(num_labels=C).to(device)
        auc_metric = MultilabelAUROC(num_labels=C).to(device)
        f1_metric = MultilabelF1Score(num_labels=C, average="macro", threshold=0.5).to(device)
        ece_metric = BinaryCalibrationError(n_bins=15, norm="l1").to(device)
        for m in (map_metric, auc_metric, f1_metric, ece_metric):
            m.reset()

        # Accumulate raw predictions so per-class AP / co-occurrence analysis is a
        # recompute rather than a re-run.
        all_probs: list[torch.Tensor] = []
        all_y: list[torch.Tensor] = []

        with torch.no_grad():
            for x, y in tqdm(loader, desc="eval"):
                x = x.float().to(self.device, non_blocking=True)
                x = normalize(x, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
                y = y.float().to(device)

                logits, embeddings = self.model(x)

                probs = torch.sigmoid(logits)
                y_int = (y > 0.5).int()

                map_metric.update(probs, y_int)
                auc_metric.update(probs, y_int)
                f1_metric.update(probs, y_int)
                ece_metric.update(probs.reshape(-1), y_int.reshape(-1))

                all_probs.append(probs.detach().float().cpu())
                all_y.append(y_int.detach().cpu())

        probs_all = torch.cat(all_probs, dim=0)          # [N, C]
        y_all = torch.cat(all_y, dim=0).int()            # [N, C]
        self._probs_np = probs_all.numpy()
        self._y_np = y_all.numpy()

        val_mAP = map_metric.compute().item()
        val_mAUC = auc_metric.compute().item()
        val_mF1 = f1_metric.compute().item()
        val_mECE = ece_metric.compute().item()

        # ---- per-class + frequency-group breakdown ----
        ap_pc = MultilabelAveragePrecision(num_labels=C, average=None).to(device)
        auc_pc = MultilabelAUROC(num_labels=C, average=None).to(device)
        ap_pc.update(probs_all.to(device), y_all.to(device))
        auc_pc.update(probs_all.to(device), y_all.to(device))
        per_class_ap = ap_pc.compute().cpu().numpy()
        per_class_auc = auc_pc.compute().cpu().numpy()
        val_pos = self._y_np.sum(axis=0).astype(int)

        try:
            import class_groups
            meta = class_groups.build(
                label_info_path=self.cfg.get("label_info_path", class_groups.DEFAULT_LABEL_INFO)
            )
            names = meta["class_names"]
            groups = meta["groups"]
            group_map = {
                g: float(np.nanmean(per_class_ap[np.where(groups[g])[0]]))
                for g in ("head", "medium", "tail", "extreme_tail")
            }
            group_map["category_avg"] = float(
                np.mean([group_map["head"], group_map["medium"], group_map["tail"]])
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"frequency-group breakdown skipped: {e}")
            names = [f"class_{i}" for i in range(C)]
            group_map = {}

        self._per_class = {
            "class_names": names,
            "ap": per_class_ap.tolist(),
            "auc": per_class_auc.tolist(),
            "val_pos": val_pos.tolist(),
        }
        self._group_map = group_map

        print(f"val_mAP:  {val_mAP:.4f}   val_mAUC: {val_mAUC:.4f}")
        print(f"val_mF1:  {val_mF1:.4f}   val_mECE: {val_mECE:.4f}")
        if group_map:
            print("  group mAP: " + "  ".join(f"{g}={group_map[g]:.4f}" for g in
                  ("head", "medium", "tail", "extreme_tail", "category_avg")))
            print(f"{'class':<32} {'val_pos':>7} {'AP':>7} {'AUC':>7}")
            for i in range(C):
                print(f"{names[i]:<32} {val_pos[i]:>7d} {per_class_ap[i]:>7.4f} {per_class_auc[i]:>7.4f}")

        self.history["val_mAP"].append(val_mAP)
        self.history["val_mAUC"].append(val_mAUC)
        self.history["val_mF1"].append(val_mF1)
        self.history["val_mECE"].append(val_mECE)

        return self.history


def seed_everything(seed: int):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Enforce deterministic algorithms (will error if an op has no det. impl)
    torch.use_deterministic_algorithms(True)


def make_val_webdataset(val_dir: str):
    val_shards = sorted(Path(val_dir).glob("shard_*.tar"))
    uri_val = ["file:" + p.resolve().as_posix() for p in val_shards]

    logger.info(f"Val shards: {len(uri_val)}")

    val_ds = (
        wds.WebDataset(uri_val, shardshuffle=False, empty_check=False)
            .map(lambda s: decode_and_transform(s, is_train=False))
            .select(lambda x: x is not None)
    )

    return val_ds


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="path to a model_best.pth (or any state_dict) to evaluate")
    ap.add_argument("--val-shards-dir", default="/data/psytp7/wds_shards_val_raw")
    ap.add_argument("--num-classes", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--img-size", type=int, default=512,
                    help="val resize; Stage-1 [b12] used 384, Stage-2 uses 512 "
                         "(A2 resolution control)")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output files, e.g. 'res512' or 'seed86'")
    ap.add_argument("--out-json", default=None,
                    help="where to write the result JSON; defaults to "
                         "eval_result[.<tag>].json next to --checkpoint")
    ap.add_argument("--save-preds", action="store_true",
                    help="also dump preds[.<tag>].npz (probs, labels, class_names) "
                         "next to --checkpoint for per-class / co-occurrence analysis")
    ap.add_argument("--no-save", action="store_true",
                    help="skip writing the result JSON, print-only")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = {
        "num_classes": args.num_classes,
        "label_info_path": str(Path(args.val_shards_dir) / "label_info.pt"),
    }

    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    set_val_img_size(args.img_size)

    val_ds = make_val_webdataset(args.val_shards_dir)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True
    )

    logger.info(f"Evaluating checkpoint: {args.checkpoint} @ {args.img_size}px")

    trainer = Trainer(cfg)
    trainer.history = trainer.evaluate(val_loader, checkpoint_path=args.checkpoint)

    ckpt_dir = Path(args.checkpoint).resolve().parent
    sfx = f".{args.tag}" if args.tag else ""

    if args.save_preds:
        npz_path = ckpt_dir / f"preds{sfx}.npz"
        np.savez_compressed(
            npz_path,
            probs=trainer._probs_np.astype(np.float32),
            labels=trainer._y_np.astype(np.int8),
            class_names=np.array(trainer._per_class["class_names"]),
            img_size=args.img_size,
            checkpoint=str(Path(args.checkpoint).resolve()),
        )
        logger.info(f"Wrote predictions to {npz_path}  {trainer._probs_np.shape}")

    if not args.no_save:
        result = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "evaluated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "val_shards_dir": args.val_shards_dir,
            "img_size": args.img_size,
            "val_mAP": trainer.history["val_mAP"][-1],
            "val_mAUC": trainer.history["val_mAUC"][-1],
            "val_mF1": trainer.history["val_mF1"][-1],
            "val_mECE": trainer.history["val_mECE"][-1],
            "group_mAP": trainer._group_map,
            "per_class": trainer._per_class,
        }
        out_json = Path(args.out_json) if args.out_json else ckpt_dir / f"eval_result{sfx}.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Wrote eval result to {out_json}")


if __name__ == "__main__":
    main()