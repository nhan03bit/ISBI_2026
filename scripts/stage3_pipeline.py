"""Submit Phase A, or monitor an existing run. Only scheduled phases do computation."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]


def run(command, **kwargs):
    print('+ ' + shlex.join(map(str, command)), flush=True)
    try:
        return subprocess.run(list(map(str, command)), check=True, **kwargs)
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr, file=sys.stderr, flush=True)
        raise


def common(config):
    return ['--checkpoint', config['checkpoint'], '--train-shards', config['train'],
            '--val-shards', config['val'], '--metadata', config['metadata'],
            '--train-fold', 'data/train_fold_1.csv', '--val-fold', 'data/val_fold_1.csv',
            '--batch-size', '4', '--accum-steps', '12', '--img-size', '768', '--workers', '4',
            '--seed', '42'] + (['--allow-missing'] if config['allow_missing'] else [])


def phase(name, directory):
    config = json.loads((directory / 'pipeline.json').read_text())
    os.chdir(ROOT)
    train = [sys.executable, '-u', 'train/train_3.py'] + common(config)
    if name == 'preflight':
        run([sys.executable, '-m', 'unittest', 'discover', '-s', 'train', '-p', 'test_stage3.py'])
        run(train + ['--arm', 'minority', '--preflight-only', '--out', directory / 'preflight'])
    elif name == 'gpu':
        run(['nvidia-smi'])
        run([sys.executable, '-c', 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; '
             'print(torch.__version__, torch.cuda.get_device_name(0)); torch.zeros(1, device="cuda"); torch.cuda.synchronize()'])
        run(train + ['--arm', 'minority', '--updates', '5', '--val-limit-batches', '2', '--out', directory / 'smoke'])
        for arm in ('natural', 'minority'):
            run(train + ['--arm', arm, '--out', directory / arm])
    else:
        for subset, extra in [('full', []), ('unseen', ['--unseen-only'])]:
            with open(directory / f'comparison_{subset}.json', 'w') as output:
                run([sys.executable, 'analysis/stage3_compare.py', directory / 'natural/predictions_002152.npz',
                     directory / 'minority/predictions_002152.npz'] + extra, stdout=output)
        lines = ['Phase A results (one seed; development evidence only)',
                 'Arm          Initial mAP   Final mAP    Final tail AP']
        for arm in ('natural', 'minority'):
            metrics = json.loads((directory / arm / 'metrics.json').read_text())
            lines.append(f'{arm:12} {metrics[0]["mAP"]:.6f}      {metrics[-1]["mAP"]:.6f}     {metrics[-1]["tail_AP"]:.6f}')
        lines.append('See comparison_full.json and comparison_unseen.json for paired intervals.')
        (directory / 'RESULTS.txt').write_text('\n'.join(lines) + '\n')
        print('\n'.join(lines), flush=True)
    (directory / f'{name}.done').touch()


def snapshot(directory):
    print(f'Stage 3: {directory}\n')
    jobs_file = directory / 'jobs.json'
    jobs = json.loads(jobs_file.read_text()) if jobs_file.exists() else {}
    if jobs:
        result = subprocess.run(['squeue', '-j', ','.join(jobs.values()), '-o', '%.12i %.12T %.35R'],
                                capture_output=True, text=True)
        print(result.stdout.strip() or 'No queued/running jobs; checking saved status.')
        result = subprocess.run(['sacct', '-X', '-j', ','.join(jobs.values()), '--format=JobID,State,ExitCode', '--noheader'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout.strip())
    for arm in ('preflight', 'smoke', 'natural', 'minority'):
        path = directory / arm / 'progress.json'
        if not path.exists():
            print(f'{arm:10} waiting')
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        fraction = min(1, data['done'] / data['total']) if data['total'] else 0
        bar = '#' * int(20 * fraction) + '-' * (20 - int(20 * fraction))
        print(f'{arm:10} [{bar}] {fraction:5.1%} {data["phase"]} {data["done"]}/{data["total"]}')
        metrics = directory / arm / 'metrics.json'
        if metrics.exists():
            try:
                latest = json.loads(metrics.read_text())[-1]
                print(f'           Last evaluation: update {latest["update"]}, mAP={latest["mAP"]}, tail AP={latest["tail_AP"]}')
            except (OSError, ValueError, IndexError):
                pass
    if (directory / 'RESULTS.txt').exists():
        print('\n' + (directory / 'RESULTS.txt').read_text())
    print(f'\nDetailed logs: {directory}/logs/\nCtrl+C closes this monitor; submitted jobs continue.')


def submit(directory, allow_missing):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'logs').mkdir()
    config = dict(checkpoint=os.environ.get('CKPT', str(ROOT / 'train/checkpoint_Triplet_3/push/img768_dp01_seed42/Model_run/model_swa.pth')),
                  train=os.environ.get('TRAIN', '/data/psytp7/wds_shards_train_raw'),
                  val=os.environ.get('VAL', '/data/psytp7/wds_shards_val_raw'),
                  metadata=os.environ.get('META', str(ROOT / 'data/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv')),
                  allow_missing=allow_missing)
    for key in ('checkpoint', 'train', 'val', 'metadata'):
        config[key] = str(Path(config[key]).resolve())
        if not Path(config[key]).exists():
            raise FileNotFoundError(f'{key}: {config[key]}')
    (directory / 'pipeline.json').write_text(json.dumps(config, indent=2))
    jobs = {}
    for name in ('preflight', 'gpu', 'report'):
        command = ['sbatch', '--parsable', '--job-name', f's3_{name}', '--chdir', str(ROOT),
                   '--output', str(directory / 'logs' / f'{name}_%j.out'), '--kill-on-invalid-dep=yes',
                   '--cpus-per-task', '8' if name == 'gpu' else '4',
                   '--mem', '64G' if name == 'gpu' else '16G',
                   '--time', '72:00:00' if name == 'gpu' else '4:00:00',
                   '--partition', os.environ.get('GPU_PARTITION', 'amp48') if name == 'gpu' else os.environ.get('CPU_PARTITION', 'general')]
        if name == 'gpu':
            command += ['--gpus', '1', '--exclude', os.environ.get('EXCLUDE', 'colossus')]
        if jobs:
            command += ['--dependency', 'afterok:' + list(jobs.values())[-1]]
        command += ['--wrap', shlex.join([sys.executable, '-u', str(Path(__file__).resolve()),
                                         '--phase', name, '--run', str(directory)])]
        result = run(command, capture_output=True, text=True)
        jobs[name] = result.stdout.strip().split(';')[0]
        (directory / 'jobs.json').write_text(json.dumps(jobs, indent=2))
    print('Submitted: ' + str(jobs), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path)
    p.add_argument('--phase', choices=['preflight', 'gpu', 'report'])
    p.add_argument('--watch', action='store_true', help='Monitor an existing --run without submitting')
    p.add_argument('--no-watch', action='store_true')
    p.add_argument('--allow-missing', action='store_true', help='Only after reviewing missing-image exclusions')
    args = p.parse_args()
    if (args.phase or args.watch) and args.run is None:
        p.error('--phase/--watch requires --run')
    directory = (args.run or ROOT / 'runs' / ('stage3_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    if args.phase:
        phase(args.phase, directory)
        return
    if not args.watch:
        submit(directory, args.allow_missing)
    if args.no_watch:
        return
    try:
        while True:
            if sys.stdout.isatty():
                print('\033[2J\033[H', end='')
            snapshot(directory)
            if (directory / 'report.done').exists():
                break
            time.sleep(10)
    except KeyboardInterrupt:
        print('\nMonitor closed. Jobs continue. Reopen with --watch --run ' + str(directory))


if __name__ == '__main__':
    main()
