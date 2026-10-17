"""Stage-2 v0: ConvNeXt2 + asymmetric loss (ASL) only. No Triplet Memory, entropy gate, prior or MoE.

This is the ASL system of the three-system comparison (ASL / entropy-gated memory / + Markov prior).
Entropy and Markov-prior systems train on train_2_v5.py; nothing is imported from it. The data pipeline,
ASL loss, early stopping, EMA / SWA and evaluation code below are verbatim copies of train_2_v5.py (checked
by scripts/check_v0_v5_parity.py), so the only difference between the two scripts is the Triplet Memory.
Reference recipe = job 49375 (768 px, EMA 0.999, SWA of the best 3 epochs, drop-path 0.1, 4 of 8 cosine epochs).
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import hashlib
import io
import json
import logging
import math
import random
from datetime import datetime
from pathlib import Path

import cv2  # type: ignore
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import webdataset as wds
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryCalibrationError,
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelF1Score,
)
from tqdm import tqdm

from convnext import ConvNeXt2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SIZE = 512  # default train/val resize; override per run with --img-size
AUG_BASE_SEED = 42

# Per-worker WebDataset .shuffle() buffer (raw, pre-decode samples; ~6.65MB each). 400 * 8 workers ~= 21GB.
SAMPLE_SHUFFLE_BUFSIZE = 400

# Original ISBI submission Stage-1 (internal val mAP 0.385).
DEFAULT_STAGE1_CKPT = "checkpoint3/Model_20260119_062652/model_best.pth"


def stable_seed_from_key(key: str, epoch: int, base_seed: int | None = None) -> int:
    """
    Stable seed that does not depend on DataLoader worker order,
    global numpy state, or sample loading order.

    base_seed defaults to the module-level AUG_BASE_SEED, read at call time (not
    bind time) so that main() can override it from --aug-base-seed before any
    augmentation call happens.
    """
    if base_seed is None:
        base_seed = AUG_BASE_SEED
    s = f"{base_seed}_{key}_{epoch}"
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (2**32)


# Resampling filters for the train crop resize (--train-interp, default linear) and the val resize (--val-interp,
# default lanczos). Linear does not antialias the ~4x downscale of the originals; area does, and scores best at
# eval time (docs/plan/2026-10-16-asl-046-ceiling-investigation.md).
INTERP = {"lanczos": cv2.INTER_LANCZOS4, "linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA}


def resize_square(img: np.ndarray, size: int, interp: str) -> np.ndarray:
    """cv2.resize to (size, size) with the named filter. INTER_AREA only antialiases when shrinking (it acts like
    nearest-neighbour when enlarging), so "area" falls back to INTER_LINEAR for an image smaller than size."""
    flag = INTERP[interp]
    if interp == "area" and min(img.shape[:2]) < size:
        flag = cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=flag)


def deterministic_random_resized_crop(
    img: np.ndarray,
    rng: np.random.RandomState,
    size: int = SIZE,
    scale=(0.9, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
    interp: str = "linear",
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
            return resize_square(crop, size, interp)

    # Fallback: deterministic center crop
    short = min(h, w)
    top = (h - short) // 2
    left = (w - short) // 2
    crop = img[top:top + short, left:left + short]
    return resize_square(crop, size, interp)


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
    rrc_scale_min: float = 0.9,
    interp: str = "linear",
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
        scale=(rrc_scale_min, 1.0),
        interp=interp,
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


def deterministic_val_resize(img: np.ndarray, size: int = SIZE, interp: str = "lanczos") -> np.ndarray:
    """
    Validation transform. The default mirrors your previous val_transform:
        A.Resize(size, size, interpolation=cv2.INTER_LANCZOS4)
    """
    return resize_square(img, size, interp)


def decode_and_transform(
    sample,
    is_train: bool = True,
    epoch: int = 0,
    size: int = SIZE,
    rrc_scale_min: float = 0.9,
    interp: str | None = None,
):
    """interp: resize filter of this split; None = linear for train, lanczos for val (every run before 2026-10-16)."""
    key = str(sample.get("__key__", "unknown_key"))
    try:
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
            img_np = deterministic_aug(
                img=img_np,
                key=key,
                epoch=epoch,
                size=size,
                log_dict=None,
                rrc_scale_min=rrc_scale_min,
                interp=interp or "linear",
            )
        else:
            img_np = deterministic_val_resize(img_np, size=size, interp=interp or "lanczos")

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

    def __init__(self, is_train: bool, size: int = SIZE, rrc_scale_min: float = 0.9, interp: str | None = None):
        self.is_train = is_train
        self.size = size
        self.rrc_scale_min = rrc_scale_min
        self.interp = interp
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __call__(self, sample):
        return decode_and_transform(
            sample,
            is_train=self.is_train,
            epoch=self.epoch,
            size=self.size,
            rrc_scale_min=self.rrc_scale_min,
            interp=self.interp,
        )


class SeededSampleShuffle:
    """Sample-buffer shuffle seeded from (--seed, DataLoader worker id, epoch).

    webdataset 1.0.2 `.shuffle(n)` seeds its RNG from os.getpid() + time.time(), so the sample order of a
    "strict_repro" run was not reproducible from --seed. Workers are recreated every epoch (persistent_workers
    is off), so the mapper's epoch is read at iteration time.
    """

    def __init__(self, bufsize: int, seed: int, mapper):
        self.bufsize = bufsize
        self.seed = seed
        self.mapper = mapper

    def __call__(self, data):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        s = stable_seed_from_key(f"sample_shuffle_w{worker_id}", self.mapper.epoch, base_seed=self.seed)
        return wds.filters._shuffle(data, self.bufsize, seed=s)


def create_model(num_classes: int = 30, stage1_ckpt: str = DEFAULT_STAGE1_CKPT, drop_path_rate: float = 0.0):
    model = ConvNeXt2(depths=[3, 3, 27, 3, 2], dims=[128, 256, 512, 1024, 1024],
                      num_classes=num_classes, drop_path_rate=drop_path_rate)

    ckpt_path = stage1_ckpt
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    result = model.load_state_dict(checkpoint, strict=False)
    logger.info(f"Stage-1 init: {ckpt_path} "
                f"(missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)})")
    if result.missing_keys:
        raise RuntimeError(f"Stage-1 load left parameters uninitialised (no new modules exist): {result.missing_keys}")

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")
    return model


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8,
                 disable_torch_grad_focal_loss=True, reduction="mean", class_weights=None):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.reduction = reduction
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def forward(self, x, y):
        y = y.float()

        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - y) * torch.log(xs_neg.clamp(min=self.eps))
        if self.class_weights is not None:
            class_weights = self.class_weights.to(x.device)
            #print("class_weights: ", class_weights)
            los_pos = los_pos * class_weights[None, :]

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
    #x = normalize(x, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    x = x.float()  # force float32 before normalize

    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)

    x = (x - mean) / std
    return x


class Trainer:
    def __init__(self, cfg: dict, class_weights, g_torch: torch.Generator):
        self.cfg = cfg
        print(self.cfg)
        print("Size of Image: ", cfg["img_size"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")

        self.g_torch = g_torch
        self.scaler = GradScaler("cuda", enabled=cfg["use_amp"])
        self.class_weights = class_weights

        self.model = create_model(cfg["num_classes"], cfg["stage1_ckpt"], cfg["drop_path_rate"]).to(self.device)
        self.optim = AdamW(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # Per-optimizer-step warmup + cosine; --sched-epochs sets the anneal horizon independently of --epochs.
        steps_per_epoch = math.ceil(cfg["train_size"] / cfg["batch_size"])
        sched_epochs = cfg.get("sched_epochs") or cfg["epochs"]
        total_opt_steps = max(1, int(sched_epochs * steps_per_epoch) // cfg["accum_steps"])
        warmup_steps = max(1, int(cfg.get("warmup_frac", 0.03) * total_opt_steps))
        self._total_opt_steps = total_opt_steps

        def _lr_lambda(cur_step: int) -> float:
            if cur_step < warmup_steps:
                return cur_step / warmup_steps
            prog = (cur_step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

        self.sched = LambdaLR(self.optim, _lr_lambda)
        logger.info(f"LR schedule: warmup {warmup_steps} / total {total_opt_steps} optimizer steps "
                    f"(sched_epochs={sched_epochs}, steps_per_epoch={steps_per_epoch}, "
                    f"accum={cfg['accum_steps']})")
        self.crit = AsymmetricLoss(gamma_neg=self.cfg["gamma_neg"], gamma_pos=self.cfg["gamma_pos"],
                                   clip=0.05, reduction="mean", class_weights=self.class_weights)

        # Weight EMA (evaluated + saved instead of the raw weights).
        self.ema_decay = float(self.cfg.get("ema_decay", 0.999))
        self.ema = {k: v.detach().clone().float() for k, v in self.model.state_dict().items()}
        # --swa-last-k: per-epoch (monitor, epoch, cpu_ema_state) for post-fit SWA.
        self.swa_last_k = int(self.cfg.get("swa_last_k", 0) or 0)
        self._swa_pool = []
        logger.info(f"v0 (ASL only): ema_decay={self.ema_decay} swa_last_k={self.swa_last_k}")

        C = cfg["num_classes"]
        self.map = MultilabelAveragePrecision(num_labels=C, average="macro").to(self.device)
        self.auc = MultilabelAUROC(num_labels=C, average="macro").to(self.device)
        self.f1 = MultilabelF1Score(num_labels=C, average="macro", threshold=0.5).to(self.device)
        self.ece = BinaryCalibrationError(n_bins=15, norm="l1").to(self.device)

        self.history = {
            "script": "train_2_v0.py",
            "stage1_ckpt": cfg["stage1_ckpt"],
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

    @torch.no_grad()
    def _ema_update(self):
        d = self.ema_decay
        for k, v in self.model.state_dict().items():
            e = self.ema[k]
            if v.dtype.is_floating_point:
                e.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                e.copy_(v)

    def _ema_state_dict(self):
        # cast EMA params back to each parameter's original dtype
        ref = self.model.state_dict()
        return {k: self.ema[k].to(ref[k].dtype) for k in ref}

    def _save_best(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        # Save the EMA weights - this is the model evaluate.py should load.
        torch.save(self._ema_state_dict(), os.path.join(out_dir, "model_best.pth"))
        self._save_json(self.history, os.path.join(out_dir, "history.json"))
        logger.info(f"Saved BEST (EMA) checkpoint to: {out_dir}")

    def _snapshot_ema_cpu(self):
        """A CPU float32 copy of the current EMA weights (for --swa-last-k)."""
        return {k: v.detach().to("cpu", torch.float32).clone() for k, v in self.ema.items()}

    def _save_swa(self, out_dir: str, val_loader):
        """Average the EMA snapshots of the K best-monitored epochs -> model_swa.pth.

        Stochastic Weight Averaging over the 2-3 near-best epochs that early
        stopping otherwise discards. Consistent with the method (EMA is already
        core); rank-preserving averaging of nearby minima tends to land in a
        flatter basin.
        """
        k = self.swa_last_k
        if k <= 0 or not self._swa_pool:
            return
        reverse = (self.es.mode == "max")
        picked = sorted(self._swa_pool, key=lambda t: t[0], reverse=reverse)[:k]
        epochs = sorted(e for _, e, _ in picked)
        logger.info(f"SWA: averaging EMA weights of epochs {epochs} "
                    f"(monitor={self.monitor}, mode={self.es.mode})")

        ref = self.model.state_dict()
        states = [s for _, _, s in picked]
        avg = {}
        for key in ref:
            if ref[key].is_floating_point():
                acc = torch.zeros_like(states[0][key], dtype=torch.float32)
                for s in states:
                    acc += s[key]
                avg[key] = (acc / len(states)).to(ref[key].dtype)
            else:
                avg[key] = states[-1][key].clone()   # counters: take the latest

        os.makedirs(out_dir, exist_ok=True)
        torch.save(avg, os.path.join(out_dir, "model_swa.pth"))

        # Evaluate the SWA weights; restore the live weights afterwards.
        _live = {kk: v.detach().clone() for kk, v in self.model.state_dict().items()}
        try:
            self.model.load_state_dict(avg, strict=True)
            va_loss, m = self._evaluate_impl(val_loader, epoch=0)
        finally:
            self.model.load_state_dict(_live, strict=True)

        self.history["swa_epochs"] = epochs
        self.history["swa_metrics"] = {"val_loss": va_loss, **m}
        self._save_json(self.history, os.path.join(out_dir, "history.json"))
        logger.info(
            f"SWA weights -> {out_dir}/model_swa.pth | "
            f"val_loss={va_loss:.4f} mAP={m['mAP']:.4f} mAUC={m['mAUC']:.4f} "
            f"mF1={m['mF1']:.4f} mECE={m['mECE']:.4f}"
        )

    def train_one_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        running = 0.0
        accum_steps = self.cfg["accum_steps"]
        self.optim.zero_grad(set_to_none=True)
        pending_grad = False   # a backward has run since the last optimizer step

        limit = int(self.cfg.get("limit_train_batches", 0) or 0)
        step = 0

        pbar = tqdm(loader, desc=f"Train {epoch}/{self.cfg['epochs']}", leave=False)
        for _, (x, y) in enumerate(pbar):
            if limit and step >= limit:
                break
            step += 1
            x = preprocess_batch(x.to(self.device, non_blocking=True))
            y = y.to(self.device, non_blocking=True)
            if not torch.is_floating_point(y):
                y = y.float()

            with autocast("cuda", enabled=self.cfg["use_amp"]):
                logits = self.model(x)[0]
                loss = self.crit(logits, y)

            raw_loss = loss.detach()
            loss = loss / accum_steps

            self.scaler.scale(loss).backward()
            pending_grad = True

            if step % accum_steps == 0:
                self.scaler.step(self.optim)
                self.scaler.update()
                self.sched.step()
                self.optim.zero_grad(set_to_none=True)
                self._ema_update()
                pending_grad = False

            running += float(raw_loss.item())

        # Flush a partial accumulation window, only if a backward is actually pending.
        if pending_grad:
            self.scaler.step(self.optim)
            self.scaler.update()
            self.sched.step()
            self.optim.zero_grad(set_to_none=True)
            self._ema_update()

        step = max(1, step)
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self.sched.get_last_lr()[0]:.2e}")
        logger.info(f"[Epoch {epoch}] ASL steps = {step} train_loss(avg)={running / step:.4f}")
        return running / max(1, step)

    @torch.no_grad()
    def evaluate(self, loader, epoch: int):
        # Evaluate the EMA weights (what _save_best persists). Swap them in for
        # the eval pass, restore the live training weights afterwards.
        _live = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self._ema_state_dict(), strict=True)
        try:
            return self._evaluate_impl(loader, epoch)
        finally:
            self.model.load_state_dict(_live, strict=True)

    @torch.no_grad()
    def _evaluate_impl(self, loader, epoch: int):
        self.model.eval()
        self.map.reset(); self.auc.reset(); self.f1.reset(); self.ece.reset()

        running = 0.0
        limit = int(self.cfg.get("limit_val_batches", 0) or 0)
        pbar = tqdm(loader, desc=f"Eval  {epoch}/{self.cfg['epochs']}", leave=False)
        for step, (x, y) in enumerate(pbar, start=1):
            if limit and step > limit:
                break
            x = preprocess_batch(x.to(self.device, non_blocking=True))
            y = y.to(self.device, non_blocking=True)
            if not torch.is_floating_point(y):
                y = y.float()

            # Evaluate in fp32: fp16 logits under autocast add noise to the
            # sigmoid probabilities that feed mAP / ECE.
            with autocast("cuda", enabled=False):
                logits = self.model(x.float())[0]
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

    def _save_last(self, epoch: int, best_epoch):
        """Epoch-boundary snapshot for --resume (written atomically, deleted when the run finishes)."""
        path = os.path.join(self.cfg["out_dir"], "last.pth")
        state = {
            "epoch": epoch, "best_epoch": best_epoch,
            "model": self.model.state_dict(), "ema": self.ema,
            "optim": self.optim.state_dict(), "sched": self.sched.state_dict(),
            "scaler": self.scaler.state_dict(),
            "es": {"best": self.es.best, "bad_epochs": self.es.bad_epochs},
            "swa_pool": self._swa_pool, "history": self.history,
            "g_torch": self.g_torch.get_state(),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }
        torch.save(state, path + ".tmp")
        os.replace(path + ".tmp", path)

    def _load_last(self):
        path = os.path.join(self.cfg["out_dir"], "last.pth")
        if not os.path.exists(path):
            logger.info(f"--resume: no {path}, starting from epoch 1")
            return 1, None
        st = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(st["model"])
        self.ema = {k: v.to(self.device) for k, v in st["ema"].items()}
        self.optim.load_state_dict(st["optim"])
        self.sched.load_state_dict(st["sched"])
        self.scaler.load_state_dict(st["scaler"])
        self.es.best, self.es.bad_epochs = st["es"]["best"], st["es"]["bad_epochs"]
        self._swa_pool = st["swa_pool"]
        self.history = st["history"]
        self.g_torch.set_state(st["g_torch"])
        random.setstate(st["rng"]["python"]); np.random.set_state(st["rng"]["numpy"])
        torch.set_rng_state(st["rng"]["torch"])
        if st["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(st["rng"]["cuda"])
        logger.info(f"--resume: restored {path} (finished epoch {st['epoch']})")
        return st["epoch"] + 1, st["best_epoch"]

    def fit(self, train_loader, val_loader, train_mapper=None):
        os.makedirs(self.cfg["out_dir"], exist_ok=True)

        start_epoch, best_epoch = (self._load_last() if self.cfg.get("resume") else (1, None))
        for epoch in range(start_epoch, self.cfg["epochs"] + 1):
            if self.es.should_stop:
                break
            if train_mapper is not None:
                train_mapper.set_epoch(epoch)

            tr_loss = self.train_one_epoch(train_loader, epoch)
            va_loss, m = self.evaluate(val_loader, epoch)
            # NOTE: self.sched is stepped per optimizer update inside
            # train_one_epoch, not once per epoch.

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

            if self.swa_last_k > 0:
                self._swa_pool.append((monitor_value, epoch, self._snapshot_ema_cpu()))

            if improved:
                best_epoch = epoch
                self.history["best_epoch"] = best_epoch
                self.history["best_metric"] = float(self.es.best)
                self._save_best(self.cfg["out_dir"])

            self._save_last(epoch, best_epoch)

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

        if self.swa_last_k > 0:
            self._save_swa(self.cfg["out_dir"], val_loader)

        last = os.path.join(self.cfg["out_dir"], "last.pth")
        if os.path.exists(last):
            os.remove(last)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_webdataset(train_dir: str, val_dir: str, seed: int = 42, strict_repro: bool = True,
                    img_size: int = SIZE, shuffle_buf: int = SAMPLE_SHUFFLE_BUFSIZE,
                    rrc_scale_min: float = 0.9, train_interp: str = "linear", val_interp: str = "lanczos"):
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

    train_mapper = DecodeAndTransform(is_train=True, size=img_size, rrc_scale_min=rrc_scale_min, interp=train_interp)
    val_mapper = DecodeAndTransform(is_train=False, size=img_size, interp=val_interp)
    logger.info(f"img_size={img_size} | shuffle_buf={shuffle_buf} | rrc_scale_min={rrc_scale_min} "
                f"| resize interp train={train_interp} val={val_interp}")

    if strict_repro:
        # Deterministic shard order, no shardshuffle, but a seeded sample-buffer shuffle: raw shard
        # order gives correlated minibatches (same study / site / scanner). Reproducible via seed_worker.
        train_ds = (
            wds.WebDataset(uri_train, shardshuffle=False)
            .compose(SeededSampleShuffle(shuffle_buf, seed, train_mapper))
            .map(train_mapper)
            .select(lambda x: x is not None)
        )
    else:
        train_ds = (
            wds.WebDataset(uri_train, shardshuffle=True)
            .shuffle(shuffle_buf)
            .map(train_mapper)
            .select(lambda x: x is not None)
        )

    val_ds = (
        wds.WebDataset(uri_val, shardshuffle=False)
        .map(val_mapper)
        .select(lambda x: x is not None)
    )

    return train_ds, val_ds, train_mapper


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accum-steps", type=int, default=6,
                    help="gradient accumulation steps; effective batch = batch_size * accum_steps.")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for seed_everything(): torch/numpy/random init, weight init, "
                         "dataloader shuffle order. Does NOT affect per-sample augmentation, "
                         "which is keyed separately off --aug-base-seed.")
    ap.add_argument("--aug-base-seed", type=int, default=42,
                    help="seeds per-sample augmentation (stable_seed_from_key).")
    ap.add_argument("--out-dir", default=None,
                    help="override the default ./checkpoint_ASL/v0_lr<lr>/seed_<S>/Model_<ts> path")
    ap.add_argument("--stage1-ckpt", default=DEFAULT_STAGE1_CKPT,
                    help="Stage-1 checkpoint to initialize from; relative to train/ or absolute.")
    ap.add_argument("--num-classes", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4, help="peak LR after warmup.")
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--drop-path-rate", type=float, default=0.0, help="ConvNeXt2 stochastic depth rate.")
    ap.add_argument("--epochs", type=int, default=8,
                    help="total epoch budget (early stopping may end sooner).")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--warmup-frac", type=float, default=0.05,
                    help="fraction of total optimizer steps spent in linear LR warmup")
    ap.add_argument("--sched-epochs", type=float, default=None,
                    help="cosine LR anneal horizon in epochs. Default None = use --epochs.")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="weight-EMA decay; the EMA weights are what is evaluated and saved.")
    ap.add_argument("--limit-train-batches", type=int, default=0,
                    help="debug/smoke: stop each training epoch after N batches (0 = no limit).")
    ap.add_argument("--limit-val-batches", type=int, default=0,
                    help="debug/smoke: stop each eval pass after N batches (0 = no limit).")
    ap.add_argument("--img-size", type=int, default=SIZE, help="square train/val resize.")
    ap.add_argument("--rrc-scale-min", type=float, default=0.9,
                    help="lower bound of the train RandomResizedCrop area scale.")
    ap.add_argument("--train-interp", choices=list(INTERP), default="linear",
                    help="resize filter of the train crop (default linear, every run before 2026-10-16; "
                         "area = antialiased).")
    ap.add_argument("--val-interp", choices=list(INTERP), default="lanczos",
                    help="resize filter of the val transform used for model selection and SWA scoring "
                         "(default lanczos, = evaluate/evaluate_tta.py --interp lanczos).")
    ap.add_argument("--shuffle-buf", type=int, default=SAMPLE_SHUFFLE_BUFSIZE,
                    help="per-worker WebDataset sample-shuffle buffer (~21GB across 8 workers at 400).")
    ap.add_argument("--swa-last-k", type=int, default=0,
                    help="if >0, after fit() average the EMA weights of the K best-monitored epochs "
                         "into model_swa.pth and evaluate it. 0 = off.")
    ap.add_argument("--train-size", type=int, default=103300,
                    help="approx number of training samples; only used to size the LR schedule horizon.")
    ap.add_argument("--gamma-neg", type=float, default=2.0)
    ap.add_argument("--gamma-pos", type=float, default=0.0)
    ap.add_argument("--monitor", default="mAP",
                    choices=["mAP", "mAUC", "mF1", "mECE", "val_loss"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-amp", dest="use_amp", action="store_false", default=True,
                    help="disable AMP (mixed precision); enabled by default")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--val-num-workers", type=int, default=2)
    ap.add_argument("--train-shards-dir", default="/data/psytp7/wds_shards_train_raw")
    ap.add_argument("--val-shards-dir", default="/data/psytp7/wds_shards_val_raw")
    ap.add_argument("--no-strict-repro", dest="strict_repro", action="store_false", default=True,
                    help="disable strict reproducibility (webdataset shardshuffle/sample-buffer "
                         "shuffle instead of deterministic shard order); enabled by default")
    ap.add_argument("--class-weight-order", default="shard", choices=["shard", "csv"],
                    help="'shard' (DEFAULT) aligns the ASL positive-term class_weights to logit order via "
                         "label_info.pt class_names. 'csv' reproduces the old index-scramble bug.")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out_dir>/last.pth (written after every epoch) if it exists.")

    return ap.parse_args()


def main():
    args = parse_args()

    # stable_seed_from_key() reads this module global as its default base_seed, so setting it here
    # before any augmentation call is sufficient.
    global AUG_BASE_SEED
    AUG_BASE_SEED = args.aug_base_seed

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg = {
        "num_classes": args.num_classes,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "warmup_frac": args.warmup_frac,
        "img_size": args.img_size,
        "rrc_scale_min": args.rrc_scale_min,
        "train_interp": args.train_interp,
        "val_interp": args.val_interp,
        "shuffle_buf": args.shuffle_buf,
        "swa_last_k": args.swa_last_k,
        "sched_epochs": args.sched_epochs,
        "drop_path_rate": args.drop_path_rate,
        "class_weight_order": args.class_weight_order,
        "train_size": args.train_size,
        "gamma_neg": args.gamma_neg,
        "gamma_pos": args.gamma_pos,
        "monitor": args.monitor,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "use_amp": args.use_amp,
        "num_workers": args.num_workers,
        "val_num_workers": args.val_num_workers,
        "train_shards_dir": args.train_shards_dir,
        "val_shards_dir": args.val_shards_dir,
        "seed": args.seed,
        "strict_repro": args.strict_repro,
        "stage1_ckpt": args.stage1_ckpt,
        "ema_decay": args.ema_decay,
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        "resume": args.resume,
    }

    # Set after the literal: an f-string reading cfg[...] inside the dict that defines cfg raises NameError.
    cfg["out_dir"] = args.out_dir or (
        f"./checkpoint_ASL/v0_lr{cfg['lr']:g}/seed_{cfg['seed']}/Model_{ts}"
    )

    SEED = cfg["seed"]

    seed_everything(SEED)
    logger.info(f"accum_steps={cfg['accum_steps']} seed={cfg['seed']} aug_base_seed={AUG_BASE_SEED} "
                f"epochs={cfg['epochs']} patience={cfg['patience']} "
                f"(effective batch = {cfg['batch_size'] * cfg['accum_steps']}) "
                f"| out_dir={cfg['out_dir']}")

    # Class-frequency stats for the ASL positive-term class weights. The model's 30 logits and the shard
    # label vectors are in "shard order" == class_names of label_info.pt, NOT the CSV column order.
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "CXRLT_2026_training_filtered.csv"))
    csv_label_cols = list(df.columns[3:])

    if cfg["class_weight_order"] == "shard":
        _li = torch.load(os.path.join(cfg["train_shards_dir"], "label_info.pt"), weights_only=False)
        order = list(_li["class_names"])
        if sorted(order) != sorted(csv_label_cols):
            raise ValueError("label_info.pt class_names vs CSV label columns differ")
    else:
        order = csv_label_cols
        logger.warning("--class-weight-order csv: class_weights stay in CSV column order "
                       "(reproduces the index-scramble bug)")

    labels_np = df[order].values.astype(np.float32)

    N = labels_np.shape[0]
    lcounts = labels_np.sum(axis=0)
    class_freq = lcounts / N

    print("Label counts:", lcounts)
    print("Class freq:", class_freq)

    class_weights = np.log(1.0 / (class_freq + 1e-8))
    class_weights = class_weights / class_weights.mean()
    class_weights = np.clip(class_weights, 0.5, 2.0)

    logger.info(f"class_weights ({cfg['class_weight_order']} order): "
                + ", ".join(f"{n}={w:.3f}" for n, w in zip(order, class_weights)))

    g_torch = torch.Generator()
    g_torch.manual_seed(SEED)

    train_ds, val_ds, train_mapper = make_webdataset(
        cfg["train_shards_dir"],
        cfg["val_shards_dir"],
        seed=SEED,
        strict_repro=cfg["strict_repro"],
        img_size=cfg["img_size"],
        shuffle_buf=cfg["shuffle_buf"],
        rrc_scale_min=cfg["rrc_scale_min"],
        train_interp=cfg["train_interp"],
        val_interp=cfg["val_interp"],
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

    trainer = Trainer(cfg, class_weights, g_torch)
    trainer.fit(train_loader, val_loader, train_mapper=train_mapper)


if __name__ == "__main__":
    main()
