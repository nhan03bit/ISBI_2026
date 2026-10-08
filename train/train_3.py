"""Phase A: matched decoder-only ASL fine-tuning. Run --help for explicit inputs."""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

from stage3_data import IndexedImages, build_index, exposure, schedule, sha256


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'train-shards', 'val-shards', 'train-fold', 'val-fold', 'metadata', 'out'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--arm', choices=['natural', 'minority'], required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--updates', type=int, default=2152)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--accum-steps', type=int, default=12)
    p.add_argument('--targeted-fraction', type=float, default=.125)
    p.add_argument('--img-size', type=int, default=768)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--val-limit-batches', type=int, default=0, help='Smoke tests only; 0 evaluates all images')
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--ema-decay', type=float, default=.999)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--allow-missing', action='store_true', help='Record missing shard images explicitly')
    p.add_argument('--stop-after', type=int, help='Stop cleanly without changing the schedule horizon')
    return p.parse_args()


def freeze_backbone(model):
    for name, parameter in model.named_parameters():
        if not name.startswith('head.'):
            parameter.requires_grad_(False)
    model.eval()
    model.head.train()


def main():
    started = time.monotonic()
    args = arguments()
    if min(args.updates, args.batch_size, args.accum_steps, args.save_every) < 1:
        raise ValueError('Budgets and batch sizes must be positive')
    if not 0 <= args.ema_decay < 1 or args.lr <= 0:
        raise ValueError('Invalid EMA decay or learning rate')
    if args.workers < 0 or args.val_limit_batches < 0 or args.img_size < 1:
        raise ValueError('Invalid worker count, validation limit or image size')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import train_2_v3 as v3
    from sklearn.metrics import average_precision_score

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    for key in ('resume', 'preflight_only', 'stop_after'):
        config.pop(key)
    names = list(torch.load(Path(args.train_shards) / 'label_info.pt', weights_only=False)['class_names'])
    if len(names) != 30 or len(set(names)) != 30:
        raise ValueError('Expected 30 unique checkpoint labels')
    val_info = Path(args.val_shards) / 'label_info.pt'
    if val_info.exists() and list(torch.load(val_info, weights_only=False)['class_names']) != names:
        raise ValueError('Train/validation shard label orders differ')
    train = build_index(args.train_shards, args.train_fold, args.metadata, names)
    val = build_index(args.val_shards, args.val_fold, args.metadata, names)
    if set(r['filename'] for r in train['records']) & set(r['filename'] for r in val['records']):
        raise ValueError('Train/validation image overlap')
    manifest = dict(config=config, checkpoint_hash=sha256(args.checkpoint), names=names,
                    environment=dict(torch=str(torch.__version__), numpy=np.__version__),
                    data={k: {a: b for a, b in idx.items() if a != 'records'} for k, idx in [('train', train), ('val', val)]},
                    source_hashes={p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name('stage3_data.py'),
                                                             Path(v3.__file__), Path(__file__).with_name('convnext.py')]})
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Output exists or resume manifest differs; use a new output directory')
    else:
        if args.resume:
            raise ValueError('No manifest to resume')
        manifest_path.write_text(json.dumps(manifest, indent=2))
    if (train['missing'] or val['missing']) and not args.allow_missing:
        raise ValueError('Missing shard images recorded in manifest; review before --allow-missing')
    batches = schedule(train, args.updates * args.accum_steps, args.batch_size,
                       args.targeted_fraction if args.arm == 'minority' else 0, args.seed)
    (out / 'planned_exposure.json').write_text(json.dumps(exposure(train, batches), indent=2))
    train_ds = IndexedImages(train, True, args.img_size, args.seed)
    val_ds = IndexedImages(val, False, args.img_size, args.seed)
    if args.preflight_only:
        # Full decode/label audit is intentionally opt-in and does not construct the model.
        for dataset in (train_ds, val_ds):
            for i in range(len(dataset)):
                dataset[i]
        print('Full shard/label/decode preflight passed', flush=True)
        return
    v3.seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = v3.ConvNeXt2(depths=[3, 3, 27, 3, 2], dims=[128, 256, 512, 1024, 1024], num_classes=30)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    freeze_backbone(model)
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    freq = np.asarray(train['counts']) / train['total']
    weights = np.log(1 / (freq + 1e-8))
    weights = np.clip(weights / weights.mean(), .5, 2)
    criterion = v3.AsymmetricLoss(gamma_neg=2, gamma_pos=0, class_weights=weights)
    tail = np.flatnonzero((freq > 0) & (freq < .01))
    head = np.argsort(-freq)[:5]
    medium = np.array([c for c in range(30) if c not in set(tail) | set(head)])
    train_patients = {r['patient'] for r in train['records']}
    val_patients = np.asarray([r['patient'] for r in val['records']])
    unseen = np.asarray([p not in train_patients for p in val_patients])
    # Only decoder state changes; storing a second entire backbone wastes GPU memory.
    ema = {k: v.detach().clone() for k, v in model.head.state_dict().items()}
    start = 0
    history = []
    if args.resume:
        state = torch.load(out / 'resume.pt', map_location='cpu', weights_only=False)
        model.head.load_state_dict(state['head'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        ema = {k: v.to(device) for k, v in state['ema'].items()}
        start, history = state['update'], state['history']
        random.setstate(state['random'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda'])
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_ds, batch_sampler=batches[start * args.accum_steps:], num_workers=args.workers,
                        generator=generator, pin_memory=device.type == 'cuda')
    validation = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers,
                            generator=torch.Generator().manual_seed(args.seed))

    def evaluate(step):
        raw = {k: v.detach().clone() for k, v in model.head.state_dict().items()}
        model.head.load_state_dict(ema)
        model.eval()
        scores, labels = [], []
        with torch.no_grad():
            for batch_id, (x, y) in enumerate(validation):
                if args.val_limit_batches and batch_id >= args.val_limit_batches:
                    break
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    logits, _ = model(v3.preprocess_batch(x.to(device)))
                scores.append(logits.float().sigmoid().cpu().numpy())
                labels.append(y.numpy())
        scores, labels = np.concatenate(scores), np.concatenate(labels)
        eval_patients = val_patients[:len(labels)]
        ap = [float(average_precision_score(labels[:, c], scores[:, c])) if labels[:, c].sum() else None for c in range(30)]
        def mean(ids):
            values = [ap[c] for c in ids if ap[c] is not None]
            return sum(values) / len(values) if values else None
        result = dict(update=step, mAP=mean(range(30)), tail_AP=mean(tail), head_AP=mean(head),
                      medium_AP=mean(medium), per_class_AP=ap, positive_counts=labels.sum(0).tolist(),
                      smoke_subset=bool(args.val_limit_batches),
                      positive_patients=[len(set(eval_patients[labels[:, c] == 1])) for c in range(30)],
                      brier_per_class=((scores - labels) ** 2).mean(0).tolist(),
                      predicted_positive_fraction=(scores >= .5).mean(0).tolist())
        np.savez_compressed(out / f'predictions_{step:06d}.npz', scores=scores, labels=labels,
                            names=names, patients=eval_patients,
                            keys=[r['key'] for r in val['records'][:len(labels)]], unseen=unseen[:len(labels)],
                            tail=tail, head=head)
        torch.save(model.state_dict(), out / f'model_{step:06d}.pth')
        history.append(result)
        (out / 'metrics.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(result), flush=True)
        model.head.load_state_dict(raw)
        freeze_backbone(model)

    def save(step):
        state = dict(update=step, head=model.head.state_dict(), optimizer=optimizer.state_dict(),
                     scaler=scaler.state_dict(), ema=ema, history=history, random=random.getstate(),
                     numpy=np.random.get_state(), torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        torch.save(state, out / 'resume.tmp')
        os.replace(out / 'resume.tmp', out / 'resume.pt')
        (out / 'actual_exposure.json').write_text(json.dumps(exposure(train, batches[:step * args.accum_steps]), indent=2))
        (out / 'runtime.json').write_text(json.dumps(dict(completed_updates=step,
                session_seconds=time.monotonic() - started, device=str(device),
                peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None), indent=2))

    if not args.resume:
        evaluate(0)
        save(0)
    optimizer.zero_grad(set_to_none=True)
    for j, (x, y) in enumerate(loader):
        step = start + j // args.accum_steps
        warmup = max(1, int(.05 * args.updates))
        factor = (step + 1) / warmup if step < warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, args.updates - warmup)))
        for group in optimizer.param_groups:
            group['lr'] = args.lr * factor
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            logits, _ = model(v3.preprocess_batch(x.to(device)))
            loss = criterion(logits.float(), y.to(device)) / args.accum_steps
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite classification loss')
        scaler.scale(loss).backward()
        if (j + 1) % args.accum_steps:
            continue
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient; refusing silent skipped update')
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for k, value in model.head.state_dict().items():
                if value.is_floating_point():
                    ema[k].lerp_(value, 1 - args.ema_decay)
                else:
                    ema[k].copy_(value)
        done = step + 1
        print(f'update={done}/{args.updates} loss_last_batch={loss.item() * args.accum_steps:.6f}', flush=True)
        if done in {max(1, args.updates // 2), args.updates}:
            evaluate(done)
        stop = args.stop_after is not None and done >= args.stop_after
        if done % args.save_every == 0 or done == args.updates or stop:
            save(done)
        if stop:
            break


if __name__ == '__main__':
    main()
