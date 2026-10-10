"""Validate matched ablation manifests and report fixed-endpoint AP (no best-epoch selection)."""
import argparse
import csv
import json
from pathlib import Path
import statistics

ARMS = {'baseline': (0, 'all'), 'triplet': (.1, 'all'), 'entropy': (.1, 'mean')}


def report(root):
    rows = []
    seed_recipe = None
    for seed_dir in sorted(Path(root).glob('seed_*')):
        reference = None
        reference_counts = None
        initial_reference = None
        for arm, expected in ARMS.items():
            directory = seed_dir / arm
            manifest = json.loads((directory / 'manifest.json').read_text())
            cfg = manifest['config'].copy()
            if (cfg['triplet_lambda'], cfg['anchor_rule']) != expected:
                raise ValueError(f'Incorrect arm configuration: {directory}')
            if not cfg['fixed_budget'] or not cfg['strict_init'] or cfg['swa_last_k']:
                raise ValueError('Requires strict init, fixed budget and no SWA')
            for key in ('out_dir', 'triplet_lambda', 'anchor_rule'):
                cfg.pop(key)
            matched = dict(manifest, config=cfg)
            across_seeds = dict(matched, config={k: v for k, v in cfg.items() if k != 'seed'})
            if seed_recipe is not None and across_seeds != seed_recipe:
                raise ValueError(f'Unmatched recipe across seeds: {directory}')
            seed_recipe = across_seeds
            if reference is not None and matched != reference:
                raise ValueError(f'Unmatched manifest: {directory}')
            reference = matched
            history = json.loads((directory / 'history.json').read_text())
            counts = [entry['anchors'] for entry in history['triplet_diagnostics']]
            if reference_counts is not None and counts != reference_counts:
                raise ValueError('Arms processed different training image counts')
            reference_counts = counts
            if initial_reference is not None and history['initial_metrics'] != initial_reference:
                raise ValueError('Initialization evaluations differ; investigate before attribution')
            initial_reference = history['initial_metrics']
            final = history['epoch_metrics'][-1]
            if final['epoch'] != cfg['epochs'] or not (directory / 'model_final.pth').exists():
                raise ValueError(f'Incomplete fixed-budget run: {directory}')
            rows.append(dict(seed=cfg['seed'], arm=arm, epoch=final['epoch'],
                             smoke=bool(cfg['limit_train_batches'] or cfg['limit_val_batches']),
                             **{k: final[k] for k in ('mAP', 'head_AP', 'medium_AP', 'tail_AP')}))
    if not rows:
        raise ValueError('No seed_* runs found')
    with open(Path(root) / 'ablation.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['Fixed endpoint; EMA; no TTA/SWA/ensemble.', 'Arm | n seeds | overall AP | head AP | medium AP | tail AP']
    if any(r['smoke'] for r in rows):
        lines.insert(0, 'SMOKE RESULTS ONLY — not performance evidence.')
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        cells = []
        for key in ('mAP', 'head_AP', 'medium_AP', 'tail_AP'):
            values = [r[key] for r in selected]
            if any(v is None for v in values):
                cells.append('unsupported')
            else:
                cells.append(f'{statistics.mean(values):.6f}' +
                             (f' +/- {statistics.stdev(values):.6f}' if len(values) > 1 else ' (one seed)'))
        lines.append(' | '.join([arm, str(len(selected))] + cells))
    for first, second in [('baseline', 'triplet'), ('triplet', 'entropy')]:
        differences = [next(r['mAP'] for r in rows if r['seed'] == seed and r['arm'] == second) -
                       next(r['mAP'] for r in rows if r['seed'] == seed and r['arm'] == first)
                       for seed in sorted({r['seed'] for r in rows})]
        lines.append(f'{second} minus {first}: mean paired mAP delta {statistics.mean(differences):+.6f}; seeds={differences}')
    (Path(root) / 'RESULTS.txt').write_text('\n'.join(lines) + '\n')
    return rows, '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory')
    args = parser.parse_args()
    print(report(args.run_directory)[1])
