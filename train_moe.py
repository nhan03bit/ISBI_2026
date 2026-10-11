"""Reconstruction of the original MoE Stage 2 (ISBI 2026 challenge submission, b12).

The submission's MoE trainer was not preserved. This rebuilds it from the paper's
Sections 2.3, 2.4 and 3.2 on top of the Stage-1 pipeline in train.py (same data,
augmentation, Asymmetric Loss, AdamW, cosine schedule, metrics), changing only what
the paper says changed:

  * ConvNeXt3 (convnext3.py): top-2 MoE over the MLP of every block in stages 4-5
    (indices 3, 4) and over the decoder FFN after cross-attention; 4 experts.
  * Expert 0 is the Stage-1 layer; experts 1..E-1 are copies of it plus small noise.
  * Stages 1-3 (and every downsample layer) frozen; only stage 4-5 blocks (experts and
    routers included) and the decoder are trained.
  * loss = ASL + 5e-3 * load-balancing aux loss. lr 4e-5, weight decay 0.01, 384 px,
    2 epochs, initialised from the Stage-1 checkpoint.

Not stated in the paper, so chosen here: the expert noise scale (--expert-noise, a
fraction of each tensor's std), batch size 32 and the cosine schedule (train.py's).
Because the original code is gone this is a reconstruction, not a reproduction of the
0.405 row of b12 Table 1.
"""
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np

import train as stage1
from train import (
    SIZE, Trainer, logger, make_webdataset, preprocess_batch, seed_everything,
    setup_run_log_file,
)
from convnext3 import ConvNeXt3

DEFAULT_STAGE1_CKPT = "/home/psytp7/ISBI_2026/train/checkpoint3/Model_20260119_062652/model_best.pth"


@torch.no_grad()
def _copy_with_noise(dst: nn.Linear, src: nn.Linear, noise: float, gen: torch.Generator) -> None:
    for d, s in ((dst.weight, src.weight), (dst.bias, src.bias)):
        d.copy_(s)
        if noise > 0:
            d.add_(torch.randn(d.shape, generator=gen) * (noise * s.float().std().item()))


@torch.no_grad()
def init_experts_from_base(model: ConvNeXt3, noise: float, seed: int) -> int:
    """Experts 1..E-1 <- expert 0 (the Stage-1 weights) + N(0, (noise*std)^2). Returns #experts set."""
    gen = torch.Generator().manual_seed(seed)
    n = 0
    for stage in (model.stages[3], model.stages[4]):
        for blk in stage:
            for ex in blk.moe_mlp.extra_experts:
                _copy_with_noise(ex.fc1, blk.pwconv1, noise, gen)
                _copy_with_noise(ex.fc2, blk.pwconv2, noise, gen)
                n += 1
    for layer in model.head.decoder_layers:
        for ex in layer.moe_ffn_ca.extra_experts:
            _copy_with_noise(ex.fc1, layer.ffn_ca[0], noise, gen)
            _copy_with_noise(ex.fc2, layer.ffn_ca[3], noise, gen)
            n += 1
    return n


def freeze_early_stages(model: ConvNeXt3) -> None:
    """Train only stage 4-5 blocks and the decoder (b12 Sec. 2.4); everything else frozen."""
    for p in model.parameters():
        p.requires_grad = False
    for module in (model.stages[3], model.stages[4], model.head):
        for p in module.parameters():
            p.requires_grad = True


def build_moe_model(cfg: dict) -> ConvNeXt3:
    model = ConvNeXt3(
        depths=[3, 3, 27, 3, 2], dims=[128, 256, 512, 1024, 1024],
        num_classes=cfg["num_classes"], moe_n_experts=cfg["n_experts"],
    )
    sd = torch.load(cfg["stage1_ckpt"], map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if "extra_experts" not in k and ".router." not in k]
    if bad or unexpected:
        raise RuntimeError(f"Stage-1 checkpoint does not fit ConvNeXt3: missing={bad[:5]} unexpected={unexpected[:5]}")
    n = init_experts_from_base(model, cfg["expert_noise"], cfg["seed"])
    freeze_early_stages(model)
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"MoE: {cfg['n_experts']} experts top-2, {n} extra experts initialised from Stage-1 "
        f"(noise={cfg['expert_noise']}), router tensors left at init, stage1_ckpt={cfg['stage1_ckpt']}"
    )
    logger.info(f"MoE: trainable {tr:,} / {sum(p.numel() for p in model.parameters()):,} params")
    return model


class MoETrainer(Trainer):
    def __init__(self, cfg: dict):
        # Trainer.__init__ resolves create_model as a train.py module global.
        stage1.create_model = lambda *a, **k: build_moe_model(cfg)
        super().__init__(cfg)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optim = AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        self.sched = CosineAnnealingLR(self.optim, T_max=cfg["epochs"])
        self.aux_weight = cfg["aux_weight"]
        self.history["init"] = None

    def train_one_epoch(self, loader, epoch: int) -> float:
        self.model.train()
        running = 0.0
        aux_running = 0.0
        limit = int(self.cfg.get("limit_train_batches", 0) or 0)
        pbar = tqdm(loader, desc=f"Train {epoch}/{self.cfg['epochs']}", leave=False)
        for step, (x, y) in enumerate(pbar, start=1):
            if limit and step > limit:
                break
            x = preprocess_batch(x.to(self.device, non_blocking=True))
            y = y.to(self.device, non_blocking=True)
            if not torch.is_floating_point(y):
                y = y.float()

            self.optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.amp_enabled):
                logits = self.model(x)
            aux = self.model.last_aux_loss.float()
            cls = self.crit(logits.float(), y)
            loss = cls + self.aux_weight * aux

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

            running += float(cls.item())
            aux_running += float(aux.item())
            pbar.set_postfix(loss=f"{cls.item():.4f}", aux=f"{aux.item():.3f}")
            if step % self.cfg["log_every"] == 0:
                logger.info(f"[Epoch {epoch} Step {step}] train_loss(avg)={running/step:.4f} "
                            f"last_loss={cls.item():.4f} aux(avg)={aux_running/step:.4f}")
        return running / max(1, step)

    def fit(self, train_loader, val_loader, train_mapper=None):
        # Epoch-0 evaluation of the initialised MoE (experts = Stage-1 + noise): the Stage-1
        # operating point at this resolution, measured with the same code as the epochs.
        if self.cfg["eval_init"] and not os.path.exists(os.path.join(self.cfg["out_dir"], "last.pth")):
            _, m0 = self.evaluate(val_loader, 0)
            self.history["init"] = m0
            logger.info("Epoch 00/%d (init, before training) | mAP=%.4f mAUC=%.4f mF1=%.4f mECE=%.4f"
                        % (self.cfg["epochs"], m0["mAP"], m0["mAUC"], m0["mF1"], m0["mECE"]))
        super().fit(train_loader, val_loader, train_mapper=train_mapper)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=2, help="b12: MoE stage trained 2 additional epochs")
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=SIZE, help="b12: 384")
    ap.add_argument("--n-experts", type=int, default=4, help="b12: 4 experts, top-2")
    ap.add_argument("--aux-weight", type=float, default=5e-3, help="b12: lambda = 5e-3")
    ap.add_argument("--expert-noise", type=float, default=0.01,
                    help="noise std as a fraction of each source tensor's std (not stated in b12)")
    ap.add_argument("--stage1-ckpt", default=DEFAULT_STAGE1_CKPT)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--no-eval-init", dest="eval_init", action="store_false", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    ap.add_argument("--limit-train-batches", type=int, default=0)
    ap.add_argument("--limit-val-batches", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"./checkpoint/moe_recon/seed_{args.seed}/Model_{ts}"
    logger.info(f"Run log: {setup_run_log_file('./logs', ts)}")
    cfg = {
        "num_classes": 30, "lr": args.lr, "weight_decay": args.weight_decay, "epochs": args.epochs,
        "patience": args.epochs, "min_delta": 0.0, "monitor": "mAP",
        "batch_size": args.batch_size, "num_workers": args.num_workers, "val_num_workers": 2,
        "log_every": 500, "out_dir": out_dir, "img_size": args.img_size,
        "limit_train_batches": args.limit_train_batches, "limit_val_batches": args.limit_val_batches,
        "init_weights": None,
        "train_shards_dir": "/data/psytp7/wds_shards_train_raw",
        "val_shards_dir": "/data/psytp7/wds_shards_val_raw",
        "seed": args.seed, "strict_repro": True, "resume": args.resume,
        "n_experts": args.n_experts, "aux_weight": args.aux_weight,
        "expert_noise": args.expert_noise, "stage1_ckpt": args.stage1_ckpt,
        "eval_init": args.eval_init,
    }
    logger.info(f"args: {vars(args)}")

    seed_everything(args.seed)
    g_torch = torch.Generator()
    g_torch.manual_seed(args.seed)

    train_ds, val_ds, train_mapper = make_webdataset(
        cfg["train_shards_dir"], cfg["val_shards_dir"], seed=args.seed,
        strict_repro=cfg["strict_repro"], img_size=cfg["img_size"],
    )

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=True,
        generator=g_torch, worker_init_fn=seed_worker if cfg["num_workers"] > 0 else None,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], num_workers=cfg["val_num_workers"], pin_memory=True,
        generator=g_torch, worker_init_fn=seed_worker, persistent_workers=False,
    )

    trainer = MoETrainer(cfg)
    trainer.fit(train_loader, val_loader, train_mapper=train_mapper)


if __name__ == "__main__":
    main()
