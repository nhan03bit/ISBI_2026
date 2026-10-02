"""evaluate_tta.py for Stage-2 v5 checkpoints (patient-prior label-query conditioning).

Loads v3 (ConvNeXt2), v4 (ConvNeXt2Markov) and v5 (ConvNeXt2Prior) checkpoints through
train/prior_conditioning.load_model_for_state_dict, so all arms of the A-vs-B comparison
(docs/plan/2026-09-30-markov-ab-training-plan.md) are dumped by one tool with one row order.
Same CLI, TTA, ensembling and --dump-probs as evaluate_tta.py, plus:

  --prior-file priors.pt       per-image prior vectors (analysis/patient_prior.py); v5 models
                               need it, v3/v4 models ignore it
  --prior-mode with|zero       feed each image's prior, or an all-zero prior (= "no history"),
                               to v5 models; their dump files get a _prior-with / _prior-zero tag
  keys.txt                     the shard filename of every dumped row, so dumps no longer need
                               the reconstructed analysis/out/val_key_order.npy (if keys.txt
                               already exists in --dump-probs DIR it must match exactly)

    python evaluate_tta_v5.py --prior-file PRIORS --prior-mode with --dump-probs DIR ckptA@768 ...
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchmetrics.classification import (BinaryCalibrationError, MultilabelAUROC,
                                         MultilabelAveragePrecision, MultilabelF1Score)
from torchvision.transforms.functional import normalize

import evaluate_tta as base
from prior_conditioning import PRIOR_DIM, is_prior_model, load_model_for_state_dict

_PRIOR_LOOKUP = None  # {fname: r (31,)}, set in main before the loader forks its workers
_base_decode_and_resize = base.decode_and_resize  # the original; base's name is rebound in run()


def load_model(ckpt_path, device):
    sd = torch.load(ckpt_path, map_location="cpu")
    model, res = load_model_for_state_dict(sd)
    kind = type(model).__name__
    base.logger.info(f"loaded {ckpt_path} as {kind} "
                     f"(missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)})")
    return model.to(device).eval()


def decode_and_resize(sample):
    fname = sample["fname"].decode()
    r = np.zeros(PRIOR_DIM, np.float32)
    if _PRIOR_LOOKUP is not None:
        if fname not in _PRIOR_LOOKUP:
            raise KeyError(f"{fname} has no entry in the --prior-file")
        r = _PRIOR_LOOKUP[fname]
    out = _base_decode_and_resize(sample)
    if out is None:
        return None
    xs, y = out
    return xs, y, torch.from_numpy(np.array(r, np.float32)), fname


@torch.no_grad()
def model_output(model, xs, tta_flip, scales, logit_avg, prior):
    terms = []
    for s in scales:
        x = xs[str(s)]
        for v in [x] + ([torch.flip(x, dims=[3])] if tta_flip else []):
            out = model(v, prior=prior) if prior is not None else model(v)
            lg = (out[0] if isinstance(out, tuple) else out).float()
            terms.append(lg if logit_avg else torch.sigmoid(lg))
    return torch.stack(terms).mean(0)


@torch.no_grad()
def run(specs, val_dir, batch_size, num_workers, tta_flip, logit_avg, dump_dir, prior_mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(p, device) for p, _ in specs]
    if any(is_prior_model(m) for m in models) and _PRIOR_LOOKUP is None:
        raise ValueError("a v5 prior checkpoint needs --prior-file")

    C = 30
    m_map = MultilabelAveragePrecision(num_labels=C, average="macro").to(device)
    m_auc = MultilabelAUROC(num_labels=C, average="macro").to(device)
    m_f1 = MultilabelF1Score(num_labels=C, average="macro", threshold=0.5).to(device)
    m_ece = BinaryCalibrationError(n_bins=15, norm="l1").to(device)

    per_model = [[] for _ in specs]
    labels, keys = [], []
    base.decode_and_resize = decode_and_resize  # make_val_loader maps base.decode_and_resize
    loader = base.make_val_loader(val_dir, batch_size, num_workers)
    for xs, y, r, fnames in loader:
        xs = {k: normalize(v.float().to(device, non_blocking=True), IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
              for k, v in xs.items()}
        y_int = (y.to(device) > 0.5).int()
        r = r.to(device)
        prior = r if prior_mode == "with" else torch.zeros_like(r)

        acc = None
        for i, (model, (_, scales)) in enumerate(zip(models, specs)):
            o = model_output(model, xs, tta_flip, scales, logit_avg, prior if is_prior_model(model) else None)
            per_model[i].append((torch.sigmoid(o) if logit_avg else o).cpu())
            acc = o if acc is None else acc + o
        acc = acc / len(models)
        probs = torch.sigmoid(acc) if logit_avg else acc
        labels.append(y_int.cpu())
        keys.extend(fnames)

        m_map.update(probs, y_int)
        m_auc.update(probs, y_int)
        m_f1.update(probs, y_int)
        m_ece.update(probs.reshape(-1), y_int.reshape(-1))

    if dump_dir is not None:
        dump_dir = Path(dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        lab = torch.cat(labels).numpy().astype(np.uint8)
        key_file = dump_dir / "keys.txt"
        if key_file.exists():
            old = key_file.read_text().split("\n")
            if old != keys:
                raise RuntimeError(f"row order differs from the existing {key_file}")
            if not (np.load(dump_dir / "labels.npy") == lab).all():
                raise RuntimeError("labels differ from the existing labels.npy")
        else:
            key_file.write_text("\n".join(keys))
            np.save(dump_dir / "labels.npy", lab)
        manifest = []
        for (path, scales), chunks, model in zip(specs, per_model, models):
            rp = Path(path).resolve()
            name = "__".join(rp.parts[-4:-2] + (rp.stem,)).replace("/", "_")
            tag = f"_prior-{prior_mode}" if is_prior_model(model) else ""
            name = f"{name}@{'-'.join(map(str, scales))}{'_flip' if tta_flip else ''}{tag}.npy"
            np.save(dump_dir / name, torch.cat(chunks).numpy().astype(np.float32))
            manifest.append({"file": name, "checkpoint": str(rp), "scales": scales, "tta_flip": tta_flip,
                             "prior_mode": prior_mode if is_prior_model(model) else None})
        with open(dump_dir / f"manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(manifest, f, indent=2)
        base.logger.info(f"dumped per-model probs for {len(specs)} model(s) + keys.txt to {dump_dir}")

    return {
        "val_mAP": float(m_map.compute().item()),
        "val_mAUC": float(m_auc.compute().item()),
        "val_mF1": float(m_f1.compute().item()),
        "val_mECE": float(m_ece.compute().item()),
        "n_rows": len(keys),
    }


def main():
    global _PRIOR_LOOKUP
    argv = sys.argv[1:]
    # --prior-file / --prior-mode are v5-only; strip them before evaluate_tta's parser runs
    prior_file, prior_mode = None, "with"
    for flag in ("--prior-file", "--prior-mode"):
        if flag in argv:
            i = argv.index(flag)
            val = argv[i + 1]
            del argv[i:i + 2]
            if flag == "--prior-file":
                prior_file = val
            else:
                prior_mode = val
    if prior_mode not in ("with", "zero"):
        raise ValueError("--prior-mode must be 'with' or 'zero'")
    sys.argv = [sys.argv[0]] + argv
    args = base.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    if prior_file:
        pri = torch.load(prior_file, weights_only=False)
        _PRIOR_LOOKUP = {n: pri["val"]["r"][i] for i, n in enumerate(pri["val"]["fnames"])}
        base.logger.info(f"prior-file {prior_file}: {len(_PRIOR_LOOKUP)} val images | mode={prior_mode}")

    default_scales = [int(s) for s in args.scales.split(",") if s.strip()]
    specs = [base.parse_ckpt_spec(c, default_scales) for c in args.checkpoints]
    base._LOAD_SIZES = tuple(sorted({s for _, sc in specs for s in sc}))

    res = run(specs, args.val_shards_dir, args.batch_size, args.num_workers,
              args.tta_flip, args.logit_avg, args.dump_probs, prior_mode)
    res.update({
        "checkpoints": [str(Path(p).resolve()) for p, _ in specs],
        "scales_per_checkpoint": [sc for _, sc in specs],
        "n_models": len(specs),
        "tta_flip": args.tta_flip,
        "scales": list(base._LOAD_SIZES),
        "logit_avg": args.logit_avg,
        "prior_file": prior_file,
        "prior_mode": prior_mode,
        "evaluated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    })
    print(json.dumps(res, indent=2))
    out = Path(args.out_json) if args.out_json else Path("eval_tta_v5_result.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    base.logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
