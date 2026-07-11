import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import io
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
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

from convnext import ConvNeXt2, model_urls

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import cv2 # type: ignore
from PIL import Image
import hashlib

SIZE = 384
AUG_BASE_SEED = 42
AUG_LOG = {}
ENABLE_AUG_LOG = False


def stable_seed_from_key(key: str, epoch: int, base_seed: int = AUG_BASE_SEED) -> int:
    """
    Stable seed that does not depend on DataLoader worker order,
    global numpy state, or sample loading order.
    """
    s = f"{base_seed}_{key}_{epoch}"
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (2**32)


def deterministic_random_resized_crop(
    img: np.ndarray,
    rng: np.random.RandomState,
    size: int = SIZE,
    scale=(0.9, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
) -> np.ndarray:
    """
    Deterministic OpenCV replacement for Albumentations RandomResizedCrop.

    This keeps the original spirit of:
        A.RandomResizedCrop(size=(size, size), scale=(0.9, 1.0), p=1.0)
    """
    h, w = img.shape[:2]
    area = h * w

    log_ratio = (np.log(ratio[0]), np.log(ratio[1]))

    for _ in range(10):
        target_area = rng.uniform(scale[0], scale[1]) * area
        aspect_ratio = np.exp(rng.uniform(log_ratio[0], log_ratio[1]))

        crop_w = int(round(np.sqrt(target_area * aspect_ratio)))
        crop_h = int(round(np.sqrt(target_area / aspect_ratio)))

        if 0 < crop_w <= w and 0 < crop_h <= h:
            top = rng.randint(0, h - crop_h + 1)
            left = rng.randint(0, w - crop_w + 1)

            crop = img[top:top + crop_h, left:left + crop_w]
            return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

    # Fallback: deterministic center crop
    short = min(h, w)
    top = (h - short) // 2
    left = (w - short) // 2
    crop = img[top:top + short, left:left + short]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def deterministic_brightness_contrast(
    img: np.ndarray,
    rng: np.random.RandomState,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
) -> np.ndarray:
    """
    Deterministic approximation of A.RandomBrightnessContrast(p=0.5).
    """
    alpha = 1.0 + rng.uniform(-contrast_limit, contrast_limit)
    beta = rng.uniform(-brightness_limit, brightness_limit) * 255.0

    img = img.astype(np.float32) * alpha + beta
    return np.clip(img, 0, 255).astype(np.uint8)


def deterministic_aug(
    img: np.ndarray,
    key: str,
    epoch: int,
    size: int = SIZE,
    log_dict: dict | None = None,
) -> np.ndarray:
    seed = stable_seed_from_key(key, epoch)
    rng = np.random.RandomState(seed)

    decisions = {
        "key": key,
        "epoch": epoch,
        "seed": int(seed),
        "random_resized_crop": 1,
        "flip": 0,
        "shift_scale_rotate": 0,
        "brightness_contrast": 0,
        "distortion": "none",
        "noise_blur": "none",
    }

    # ---------------------------------------------------
    # 1. RandomResizedCrop, deterministic version
    # ---------------------------------------------------
    img = deterministic_random_resized_crop(
        img=img,
        rng=rng,
        size=size,
        scale=(0.9, 1.0),
    )

    h, w = img.shape[:2]

    # ---------------------------------------------------
    # 2. HorizontalFlip(p=0.5)
    # ---------------------------------------------------
    if rng.rand() < 0.5:
        img = cv2.flip(img, 1)
        decisions["flip"] = 1

    # ---------------------------------------------------
    # 3. ShiftScaleRotate(p=0.5)
    # ---------------------------------------------------
    if rng.rand() < 0.5:
        decisions["shift_scale_rotate"] = 1

        angle = rng.uniform(-15, 15)
        scale_factor = rng.uniform(0.9, 1.1)
        tx = rng.uniform(-0.05, 0.05) * size
        ty = rng.uniform(-0.05, 0.05) * size

        center = (size // 2, size // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale_factor)
        M[:, 2] += [tx, ty]

        img = cv2.warpAffine(
            img,
            M,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    # ---------------------------------------------------
    # 4. OneOf([OpticalDistortion, GridDistortion, ElasticTransform], p=0.2)
    # ---------------------------------------------------
    if rng.rand() < 0.2:
        choice = rng.randint(0, 3)

        if choice == 0:
            decisions["distortion"] = "optical"

            k1 = rng.uniform(-0.3, 0.3)
            k2 = rng.uniform(-0.1, 0.1)

            fx = w
            fy = h
            cx = w / 2
            cy = h / 2

            mapx, mapy = np.meshgrid(np.arange(w), np.arange(h))
            x = (mapx - cx) / fx
            y = (mapy - cy) / fy

            r2 = x**2 + y**2
            radial = 1 + k1 * r2 + k2 * r2**2

            x_distorted = x * radial
            y_distorted = y * radial

            mapx = (x_distorted * fx + cx).astype(np.float32)
            mapy = (y_distorted * fy + cy).astype(np.float32)

            img = cv2.remap(
                img,
                mapx,
                mapy,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        elif choice == 1:
            decisions["distortion"] = "grid"

            num_steps = 5
            x_steps = np.linspace(0, w, num_steps)
            y_steps = np.linspace(0, h, num_steps)

            x_shifts = rng.uniform(-10, 10, size=num_steps)
            y_shifts = rng.uniform(-10, 10, size=num_steps)

            mapx, mapy = np.meshgrid(np.arange(w), np.arange(h))
            mapx = mapx.astype(np.float32)
            mapy = mapy.astype(np.float32)

            for i in range(num_steps - 1):
                for j in range(num_steps - 1):
                    x0, x1 = int(x_steps[i]), int(x_steps[i + 1])
                    y0, y1 = int(y_steps[j]), int(y_steps[j + 1])

                    if x1 <= x0 or y1 <= y0:
                        continue

                    mapx[y0:y1, x0:x1] += x_shifts[i]
                    mapy[y0:y1, x0:x1] += y_shifts[j]

            img = cv2.remap(
                img,
                mapx,
                mapy,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        else:
            decisions["distortion"] = "elastic"

            alpha = rng.uniform(15, 25)
            sigma = rng.uniform(3, 5)

            dx = rng.randn(h, w).astype(np.float32)
            dy = rng.randn(h, w).astype(np.float32)

            dx = cv2.GaussianBlur(dx, (0, 0), sigma) * alpha
            dy = cv2.GaussianBlur(dy, (0, 0), sigma) * alpha

            x, y = np.meshgrid(np.arange(w), np.arange(h))
            mapx = (x + dx).astype(np.float32)
            mapy = (y + dy).astype(np.float32)

            img = cv2.remap(
                img,
                mapx,
                mapy,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

    # ---------------------------------------------------
    # 5. RandomBrightnessContrast(p=0.5)
    # ---------------------------------------------------
    if rng.rand() < 0.5:
        decisions["brightness_contrast"] = 1
        img = deterministic_brightness_contrast(img, rng)

    # ---------------------------------------------------
    # 6. OneOf([GaussNoise, GaussianBlur, MotionBlur, MedianBlur], p=0.2)
    # ---------------------------------------------------
    if rng.rand() < 0.2:
        choice = rng.randint(0, 4)

        if choice == 0:
            decisions["noise_blur"] = "gaussian_noise"

            std = rng.uniform(5, 15)
            noise = rng.normal(0, std, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        elif choice == 1:
            decisions["noise_blur"] = "gaussian_blur"

            ksize = int(rng.choice([3, 5]))
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        elif choice == 2:
            decisions["noise_blur"] = "motion_blur"

            ksize = int(rng.choice([3, 5, 7]))
            kernel = np.zeros((ksize, ksize), dtype=np.float32)

            center = ksize // 2
            angle = rng.uniform(0, np.pi)

            for i in range(ksize):
                x = int(center + (i - center) * np.cos(angle))
                y = int(center + (i - center) * np.sin(angle))

                if 0 <= x < ksize and 0 <= y < ksize:
                    kernel[y, x] = 1

            if kernel.sum() == 0:
                kernel[center, center] = 1

            kernel /= kernel.sum()
            img = cv2.filter2D(img, -1, kernel)

        else:
            decisions["noise_blur"] = "median_blur"

            ksize = int(rng.choice([3, 5]))
            img = cv2.medianBlur(img, ksize)

    if log_dict is not None:
        log_dict[(key, epoch)] = decisions

    return img


def deterministic_val_resize(img: np.ndarray, size: int = SIZE) -> np.ndarray:
    """
    Validation transform. This mirrors your previous val_transform:
        A.Resize(size, size, interpolation=cv2.INTER_LANCZOS4)
    """
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)


def decode_and_transform(
    sample,
    is_train: bool = True,
    epoch: int = 0,
    size: int = SIZE,
):
    try:
        key = str(sample.get("__key__", "unknown_key"))

        img_bytes = sample["img"]
        y = torch.load(io.BytesIO(sample["cls"]))

        img_np = cv2.imdecode(
            np.frombuffer(img_bytes, np.uint8),
            cv2.IMREAD_COLOR,
        )

        if img_np is None:
            raise ValueError("cv2.imdecode returned None")

        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        if is_train:
            log_dict = AUG_LOG if ENABLE_AUG_LOG else None
            img_np = deterministic_aug(
                img=img_np,
                key=key,
                epoch=epoch,
                size=size,
                log_dict=log_dict,
            )
        else:
            img_np = deterministic_val_resize(img_np, size=size)

        img_np = np.ascontiguousarray(img_np)
        x = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        return x, y

    except Exception as e:
        print("Decode error:", e, "key:", sample.get("__key__", None))
        return None


class DecodeAndTransform:
    """
    Epoch-aware callable wrapper for WebDataset.

    Important:
    - DataLoader currently uses persistent_workers=False by default.
    - Therefore workers are recreated every epoch, and the updated epoch value
      is visible when the dataset is copied to workers.
    """

    def __init__(self, is_train: bool, size: int = SIZE):
        self.is_train = is_train
        self.size = size
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __call__(self, sample):
        return decode_and_transform(
            sample,
            is_train=self.is_train,
            epoch=self.epoch,
            size=self.size,
        )

def create_model(
    num_classes: int = 30
):

    model = ConvNeXt2(depths=[3, 3, 27, 3, 2], 
                      dims=[128, 256, 512, 1024, 1024], 
                      num_classes=num_classes)

    # checkpoint = torch.load("/root/disk_3tb/Padchest_dataset/checkpoint3/Model_20260119_062652/model_best.pth", map_location="cpu")

    url = model_urls['convnext_base_1k']
    checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")["model"]
    remove = ['norm.weight', 'norm.bias', 'head.weight', 'head.bias']

    # print("Checkpoint: \n", checkpoint.keys())

    for k in list(checkpoint.keys()):
        if k in remove:
            del checkpoint[k]

    model.load_state_dict(checkpoint, strict=False)

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

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


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        self.model = create_model(cfg["num_classes"]).to(self.device)
        # set_params(self.model)
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")


        self.optim = AdamW(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        self.sched = CosineAnnealingLR(self.optim, T_max=cfg["epochs"])
        self.crit = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.05, reduction="mean")


        C = cfg["num_classes"]
        self.map = MultilabelAveragePrecision(num_labels=C, average="macro").to(self.device)
        self.auc = MultilabelAUROC(num_labels=C, average="macro").to(self.device)
        self.f1  = MultilabelF1Score(num_labels=C, average="macro", threshold=0.5).to(self.device)
        self.ece = BinaryCalibrationError(n_bins=15, norm="l1").to(self.device)

        self.history = {
            "train_loss": [], "val_loss": [], "lr": [],
            "mAP": [], "mAUC": [], "mF1": [], "mECE": [],
            "best_epoch": None, "best_metric": None,
        }

        # Early stopping config
        self.monitor = cfg.get("monitor", "mAP")     # one of: "mAP", "mAUC", "mF1", "mECE", "val_loss"
        self.patience = cfg.get("patience", 10)

        if self.monitor == "val_loss":
            self.es = EarlyStopping(patience=self.patience, mode="min", min_delta=cfg.get("min_delta", 0.0))
        elif self.monitor == "mECE":
            self.es = EarlyStopping(patience=self.patience, mode="min", min_delta=cfg.get("min_delta", 0.0))
        else:
            self.es = EarlyStopping(patience=self.patience, mode="max", min_delta=cfg.get("min_delta", 0.0))

    @staticmethod
    def _save_json(obj, path):
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    def _save_best(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(out_dir, "model_best.pth"))
        self._save_json(self.history, os.path.join(out_dir, "history.json"))
        logger.info(f"Saved BEST checkpoint to: {out_dir}")

    # def _normalize(self, x: torch.Tensor) -> torch.Tensor:
    #     return normalize(x, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)

    def train_one_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        running = 0.0

        pbar = tqdm(loader, desc=f"Train {epoch}/{self.cfg['epochs']}", leave=False)
        for step, (x, y) in enumerate(pbar, start=1):
            x = preprocess_batch(x.to(self.device, non_blocking=True))
            y = y.to(self.device, non_blocking=True)
            if not torch.is_floating_point(y):
                y = y.float()

            self.optim.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.crit(logits, y)

            loss.backward()
            self.optim.step()

            running += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.sched.get_last_lr()[0]:.2e}")

            if step % self.cfg["log_every"] == 0:
                logger.info(f"[Epoch {epoch} Step {step}] train_loss(avg)={running/step:.4f} last_loss={loss.item():.4f}")

        return running / max(1, step)

    @torch.no_grad()
    def evaluate(self, loader, epoch: int):
        self.model.eval()
        self.map.reset(); self.auc.reset(); self.f1.reset(); self.ece.reset()

        running = 0.0
        pbar = tqdm(loader, desc=f"Eval  {epoch}/{self.cfg['epochs']}", leave=False)
        for step, (x, y) in enumerate(pbar, start=1):
            x = preprocess_batch(x.to(self.device, non_blocking=True))
            y = y.to(self.device, non_blocking=True)
            if not torch.is_floating_point(y):
                y = y.float()

            logits = self.model(x)
            loss = self.crit(logits, y)
            running += float(loss.item())

            probs = torch.sigmoid(logits)
            y_int = (y > 0.5).int()
            self.map.update(probs, y_int)
            self.auc.update(probs, y_int)
            self.f1.update(probs, y_int)
            self.ece.update(probs.reshape(-1), y_int.reshape(-1))

        val_loss = running / max(1, step)
        metrics = {
            "mAP": float(self.map.compute().item()),
            "mAUC": float(self.auc.compute().item()),
            "mF1": float(self.f1.compute().item()),
            "mECE": float(self.ece.compute().item()),
        }
        return val_loss, metrics

    def fit(self, train_loader, val_loader, train_mapper=None):
        os.makedirs(self.cfg["out_dir"], exist_ok=True)

        best_epoch = None
        for epoch in range(1, self.cfg["epochs"] + 1):
            if train_mapper is not None:
                train_mapper.set_epoch(epoch)

            tr_loss = self.train_one_epoch(train_loader, epoch)
            va_loss, m = self.evaluate(val_loader, epoch)
            self.sched.step()

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["lr"].append(self.sched.get_last_lr()[0])
            self.history["mAP"].append(m["mAP"])
            self.history["mAUC"].append(m["mAUC"])
            self.history["mF1"].append(m["mF1"])
            self.history["mECE"].append(m["mECE"])

            logger.info(
                f"Epoch {epoch:02d}/{self.cfg['epochs']} | "
                f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} | "
                f"mAP={m['mAP']:.4f} mAUC={m['mAUC']:.4f} "
                f"mF1={m['mF1']:.4f} mECE={m['mECE']:.4f}"
            )

            monitor_value = va_loss if self.monitor == "val_loss" else m[self.monitor]
            improved = self.es.step(monitor_value)

            if improved:
                best_epoch = epoch
                self.history["best_epoch"] = best_epoch
                self.history["best_metric"] = float(self.es.best)
                self._save_best(self.cfg["out_dir"])

            if self.es.should_stop:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best {self.monitor}={self.es.best:.4f} at epoch {best_epoch}."
                )
                break

        logger.info(
            f"Done. Best epoch: {self.history['best_epoch']} | "
            f"Best {self.monitor}: {self.history['best_metric']}"
        )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def make_webdataset(train_dir: str, val_dir: str, seed: int = 42, strict_repro: bool = True):
    train_shards = sorted(Path(train_dir).glob("shard_*.tar"))
    val_shards = sorted(Path(val_dir).glob("shard_*.tar"))

    # Optional deterministic shard-level shuffle.
    # This gives a fixed pseudo-random shard order across runs.
    if strict_repro:
        rng = random.Random(seed)
        train_shards = list(train_shards)
        rng.shuffle(train_shards)

    uri_train = ["file:" + p.resolve().as_posix() for p in train_shards]
    uri_val = ["file:" + p.resolve().as_posix() for p in val_shards]

    logger.info(f"Train shards: {len(uri_train)} | Val shards: {len(uri_val)}")

    train_mapper = DecodeAndTransform(is_train=True, size=SIZE)
    val_mapper = DecodeAndTransform(is_train=False, size=SIZE)

    if strict_repro:
        # Strict mode:
        # - no WebDataset shardshuffle
        # - no WebDataset sample-buffer shuffle
        # - deterministic shard order controlled above
        train_ds = (
            wds.WebDataset(uri_train, shardshuffle=False)
            .map(train_mapper)
            .select(lambda x: x is not None)
        )
    else:
        train_ds = (
            wds.WebDataset(uri_train, shardshuffle=True)
            .shuffle(32)
            .map(train_mapper)
            .select(lambda x: x is not None)
        )

    val_ds = (
        wds.WebDataset(uri_val, shardshuffle=False)
        .map(val_mapper)
        .select(lambda x: x is not None)
    )

    return train_ds, val_ds, train_mapper


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = {
        "num_classes": 30,
        "lr": 4e-5,
        "weight_decay": 1e-2,
        "epochs": 30,         
        "patience": 3,        
        "min_delta": 0.0,
        "monitor": "mAP",     
        "batch_size": 32,
        "num_workers": 8,
        "val_num_workers": 2,
        "log_every": 500,
        "out_dir": f"./checkpoint/stage1",
        "train_shards_dir": "./wds_shards_train_raw",
        "val_shards_dir": "./wds_shards_val_raw",
        "seed": 42,
        "strict_repro": True,
    }

    SEED = cfg["seed"]

    seed_everything(SEED)

    g_torch = torch.Generator()
    g_torch.manual_seed(SEED)

    train_ds, val_ds, train_mapper = make_webdataset(
        cfg["train_shards_dir"],
        cfg["val_shards_dir"],
        seed=SEED,
        strict_repro=cfg["strict_repro"],
    )

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        pin_memory=True,
        generator=g_torch,
        worker_init_fn=seed_worker if cfg["num_workers"] > 0 else None,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        num_workers=cfg["val_num_workers"],
        pin_memory=True,
        generator=g_torch,
        worker_init_fn=seed_worker if cfg["val_num_workers"] > 0 else None,
        persistent_workers=False,
    )

    trainer = Trainer(cfg)
    trainer.fit(train_loader, val_loader, train_mapper=train_mapper)


if __name__ == "__main__":
    main()
