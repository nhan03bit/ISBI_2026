"""Data-pipeline parity: the first N train / val samples that train_2_v0.py and train_2_v5.py produce
(same seed, image size, shuffle buffer, epoch) must be bit-identical. CPU only; reads shards, trains nothing.

    python -I scripts/check_v0_v5_data.py --n 32 --img-size 768

The default --shuffle-buf 1 is a pass-through. With a real buffer (--shuffle-buf 400) the sample-buffer shuffle is
SeededSampleShuffle (seed, worker id, epoch), so v0 and v5 must still agree, a same-seed repeat must reproduce the
order and another seed must not. (Before 2026-10-14 webdataset seeded it from os.getpid() + time.time().)
"""
import argparse
import hashlib
import importlib.util
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
sys.path.insert(0, str(TRAIN))


def load(name):
    spec = importlib.util.spec_from_file_location(name, TRAIN / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def digest(mod, split, n, a):
    import torch
    mod.AUG_BASE_SEED = a.aug_base_seed
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    train_ds, val_ds, train_mapper = mod.make_webdataset(
        a.train_dir, a.val_dir, seed=a.seed, strict_repro=True, img_size=a.img_size,
        shuffle_buf=a.shuffle_buf, rrc_scale_min=0.9)
    train_mapper.set_epoch(a.epoch)
    ds = train_ds if split == "train" else val_ds
    out = []
    for i, sample in enumerate(ds):
        if i >= n:
            break
        x, y = sample[0], sample[1]
        out.append((hashlib.sha1(x.numpy().tobytes()).hexdigest(), hashlib.sha1(y.numpy().tobytes()).hexdigest()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=768)
    ap.add_argument("--shuffle-buf", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug-base-seed", type=int, default=42)
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--train-dir", default="/data/psytp7/wds_shards_train_raw")
    ap.add_argument("--val-dir", default="/data/psytp7/wds_shards_val_raw")
    a = ap.parse_args()

    v0, v5 = load("train_2_v0"), load("train_2_v5")
    ok = True
    for split in ("train", "val"):
        d0, d5 = digest(v0, split, a.n, a), digest(v5, split, a.n, a)
        same = d0 == d5 and len(d0) == a.n
        ok &= same
        print(f"{split}: v0 {len(d0)} samples, v5 {len(d5)} samples, identical={same}")
    if a.shuffle_buf > 1:
        # seeded sample-buffer shuffle: a repeat with the same seed must reproduce the order, another seed must not
        rep = digest(v0, "train", a.n, a) == digest(v0, "train", a.n, a)
        a2 = argparse.Namespace(**{**vars(a), "seed": a.seed + 1})
        other = digest(v0, "train", a.n, a) != digest(v0, "train", a.n, a2)
        print(f"shuffle: same-seed repeat identical={rep}, other seed differs={other}")
        ok &= rep and other
    if not ok:
        sys.exit("DATA PARITY FAILED")
    print("DATA PARITY OK")


if __name__ == "__main__":
    main()
