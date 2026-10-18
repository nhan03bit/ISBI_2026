"""CXR-LT 2026 Task-1 leaderboard predictions from Stage-2 checkpoints (2026-10-17).

Reads the template CSV that comes with the Task-1 images (ImageID + the 30 class columns, in the order the
leaderboard expects) and the Task-1 PNG folder, and writes the same CSV with every class column replaced by the
predicted probability. Rows keep the template's order.

The image path is the internal eval's (evaluate_tta.decode_and_resize): cv2 IMREAD_COLOR decode of the PNG bytes,
BGR -> RGB, square resize with --interp, /255, ImageNet normalisation. Models load through
evaluate_tta_v5.load_model (ConvNeXt2, v5 prior, MoE); a v5 prior model gets an all-zero prior, the "no history"
condition of the test set. An ensemble is the mean of the models' flip-TTA probabilities, as in evaluate_tta.run.
Model outputs are in label_info.pt class_names order and are mapped to the template's columns by name.

Never reads the PadChest CSV: it holds these images' labels. Label values already in the template, if any, are
ignored and overwritten.

    python -I evaluate/predict_task1.py --images-dir DIR --template labels.csv --out pred.csv \
        train/checkpoint_Triplet_3/unified768/c/seed_42/Model_run/model_swa.pth@768 [more checkpoints]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2  # type: ignore
import numpy as np
import pandas as pd
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "evaluate"))
import evaluate_tta as base  # noqa: E402  (resize_square, INTERP, parse_ckpt_spec)
import evaluate_tta_v5 as v5e  # noqa: E402  (load_model, model_output)
import convnext  # noqa: E402
from prior_conditioning import PRIOR_DIM, is_prior_model  # noqa: E402

# The repo-root convnext.py also defines ConvNeXt2 but returns bare logits; the evals use train/convnext.py.
assert Path(convnext.__file__).resolve() == (ROOT / "train/convnext.py").resolve(), convnext.__file__

LABEL_INFO = "/data/psytp7/wds_shards_val_raw/label_info.pt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def image_path(images_dir: Path, image_id: str) -> Path | None:
    """The template gives ImageID with or without the .png extension."""
    for p in (images_dir / f"{image_id}.png", images_dir / image_id):
        if p.is_file():
            return p
    return None


class Task1Images(Dataset):
    def __init__(self, paths: list[Path], sizes: list[int], interp: str, channel_mean: bool = False):
        self.paths, self.sizes, self.interp, self.channel_mean = paths, sizes, interp, channel_mean

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = cv2.imdecode(np.fromfile(self.paths[i], np.uint8), cv2.IMREAD_COLOR)
        if img is None:  # every template row needs a prediction: fail, never drop
            raise RuntimeError(f"cv2 could not decode {self.paths[i]}")
        if self.channel_mean:
            # Training images are grayscale replicated to 3 identical channels; some Task-1 PNGs carry independent
            # noise per channel. The equal-weight mean restores identical channels (a no-op when they already are).
            img = np.repeat(np.rint(img.mean(2, dtype=np.float32))[..., None], 3, 2).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        xs = {str(s): torch.from_numpy(np.ascontiguousarray(base.resize_square(img, s, self.interp)))
              .permute(2, 0, 1).float() / 255.0 for s in self.sizes}
        return i, xs


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="path or path@768 (scales after @, comma-separated)")
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True, help="ImageID + 30 class columns, leaderboard order")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--interp", choices=list(base.INTERP), default="area")
    ap.add_argument("--scales", default="768", help="default scales for a checkpoint given without @")
    ap.add_argument("--no-tta-flip", dest="tta_flip", action="store_false", default=True)
    ap.add_argument("--channel-mean", action="store_true",
                    help="average the 3 decoded channels before resizing (Task-1 PNGs with per-channel noise)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--label-info", default=LABEL_INFO)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N template rows only")
    return ap.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; pick a new --out")
    default_scales = [int(s) for s in args.scales.split(",")]
    specs = [base.parse_ckpt_spec(c, default_scales) for c in args.checkpoints]
    sizes = sorted({s for _, sc in specs for s in sc})
    # Per-model dump names, three directory levels (e.g. unified768__c__seed_42): two levels collide for
    # unified768 vs unified768_seeded. Checked before any GPU work.
    names = []
    for path, scales in specs:
        rp = Path(path).resolve()
        names.append("__".join(rp.parts[-5:-2] + (rp.stem,)) + f"@{'-'.join(map(str, scales))}"
                     + ("_flip" if args.tta_flip else "") + ("" if args.interp == "lanczos" else f"_{args.interp}")
                     + ("_chmean" if args.channel_mean else "") + ".npy")
    if len(set(names)) != len(names):
        raise ValueError(f"two checkpoints map to one dump name: {names}")

    model_names = list(torch.load(args.label_info, weights_only=False)["class_names"])
    tpl = pd.read_csv(args.template, dtype={"ImageID": str})
    cols = list(tpl.columns)
    if "ImageID" not in cols:
        raise ValueError(f"{args.template} has no ImageID column: {cols[:5]}")
    class_cols = [c for c in cols if c != "ImageID"]
    if sorted(class_cols) != sorted(model_names):
        raise ValueError("template class columns != label_info class_names\n"
                         f"  only in template: {sorted(set(class_cols) - set(model_names))}\n"
                         f"  only in model:    {sorted(set(model_names) - set(class_cols))}")
    if tpl["ImageID"].duplicated().any():
        raise ValueError(f"duplicate ImageIDs in the template: {tpl['ImageID'][tpl['ImageID'].duplicated()].head().tolist()}")
    if args.limit:
        tpl = tpl.head(args.limit).copy()
    perm = [model_names.index(c) for c in class_cols]  # probs_template = probs_model[:, perm]

    paths = [image_path(args.images_dir, i) for i in tpl["ImageID"]]
    missing = [i for i, p in zip(tpl["ImageID"], paths) if p is None]
    if missing:
        raise FileNotFoundError(f"{len(missing)} template images not in {args.images_dir}: {missing[:10]}")
    print(f"template {args.template}: {len(tpl)} rows | class order {'== model' if perm == list(range(30)) else 'permuted'}"
          f" | images found {len(paths)} | sizes {sizes} | interp {args.interp} | flip {args.tta_flip}"
          f" | channel-mean {args.channel_mean}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [v5e.load_model(p, device) for p, _ in specs]
    loader = DataLoader(Task1Images(paths, sizes, args.interp, args.channel_mean), batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=True, shuffle=False)
    per_model = np.full((len(models), len(paths), 30), np.nan, np.float32)
    for b, (idx, xs) in enumerate(loader):
        xs = {k: normalize(v.to(device, non_blocking=True), IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
              for k, v in xs.items()}
        n = len(idx)
        for m, (model, (_, scales)) in enumerate(zip(models, specs)):
            prior = torch.zeros((n, PRIOR_DIM), device=device) if is_prior_model(model) else None
            o = v5e.model_output(model, xs, args.tta_flip, scales, False, prior)
            per_model[m, idx.numpy()] = o.cpu().numpy()
        if b % 50 == 0:
            print(f"batch {b}/{len(loader)}", flush=True)
    if np.isnan(per_model).any():
        raise RuntimeError("some rows got no prediction")

    probs = per_model.mean(0)[:, perm]
    out = tpl[["ImageID"]].copy()
    for j, c in enumerate(class_cols):
        out[c] = probs[:, j]
    out = out[cols]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.8g")

    # Per-model probabilities (template row order, model class order) for later ensembling without a rerun.
    dump = args.out.with_suffix(".per_model")
    dump.mkdir(exist_ok=True)
    for name, arr in zip(names, per_model):
        np.save(dump / name, arr)
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        head = ""
    manifest = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "out": str(args.out.resolve()), "out_sha256": sha256(args.out),
        "template": str(args.template.resolve()), "template_sha256": sha256(args.template), "rows": len(out),
        "images_dir": str(args.images_dir.resolve()),
        "checkpoints": [{"path": str(Path(p).resolve()), "scales": sc, "bytes": Path(p).stat().st_size,
                         "kind": type(m).__name__, "zero_prior": is_prior_model(m), "dump": n}
                        for (p, sc), m, n in zip(specs, models, names)],
        "interp": args.interp, "tta_flip": args.tta_flip, "channel_mean": args.channel_mean, "ensemble": "mean of per-model probabilities",
        "template_class_order": class_cols, "model_class_order": model_names, "git_head": head,
        "limit": args.limit,
    }
    with open(args.out.with_suffix(".manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.out} ({len(out)} rows) + {dump}/ + manifest", flush=True)


if __name__ == "__main__":
    main()
