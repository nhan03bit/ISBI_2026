# Running Stage 3 Phase A

## One-command workflow

After logging in, run:

```bash
cd /home/psytp7/ISBI_2026 && git switch long-tailed && git pull --ff-only && bash scripts/run_stage3.sh
```

This submits three dependent Slurm jobs and opens a terminal monitor:

1. CPU: synthetic tests and full image/label preflight.
2. GPU: five-update smoke test, then natural and minority arms sequentially.
3. CPU: paired bootstrap comparison and `RESULTS.txt` summary.

No training is run on the login node. A failed job prevents dependent jobs from running.
Missing images stop preflight by default: inspect the saved manifest before explicitly
rerunning with `--allow-missing`. Every new invocation creates a dated run directory;
it does not silently resume or overwrite previous experiments.

The monitor refreshes every ten seconds. It shows Slurm queue/accounting status, separate
preflight/training/validation progress bars, and the latest available AP values. Bars reset
when the phase changes; indexing/checkpoint loading has no percentage estimate. AP updates
only at evaluation checkpoints, not every optimizer step. Ctrl+C closes only the monitor.

Reopen a monitor without submitting new jobs:

```bash
bash scripts/run_stage3.sh --watch --run runs/stage3_YYYYMMDD_HHMMSS
```

Outputs live in that run directory: `logs/` for detailed logs, `natural/` and `minority/`
for checkpoints/predictions/metrics, `comparison_full.json`, `comparison_unseen.json`,
and `RESULTS.txt` for the final comparison. The displayed comparison is one-seed development
evidence, not a claim of statistical robustness or independent-patient generalization.

Paths can be overridden with `CKPT`, `TRAIN`, `VAL`, `META`, and `VENV_DIR`; partitions with
`CPU_PARTITION` and `GPU_PARTITION`. The combined GPU job requests 72 hours because it runs
both arms; this is a limit, not a runtime prediction. No automatic later-stage experiments
or repeat-seed runs are submitted. For individual jobs/resume, use the commands below.

Implemented: natural and minority-sampled decoder-only ASL, deterministic draw schedules,
indexed uncompressed shards, exposure accounting, EMA, resume, prediction exports and paired
patient bootstrap. No reserved queues, triplet/ranking objectives or boosting yet.

## Local checks

```text
python -m unittest discover -s train -p test_stage3.py
python train/train_3.py --help
```

The integration test uses real augmentation and ASL with synthetic images and a tiny model.
It checks a frozen backbone, changed head weights, export, and exact interrupted/resumed
CPU results. It does not establish full ConvNeXt GPU compatibility or model improvement.

## University setup

### Existing cluster setup found in Stage 1/2 files

Repository references: `README.md` (SSH/Slurm), `scripts/train_stage1_v4.sh`,
`scripts/train_2_v3.sh`, and `scripts/submit_stage2_v3_seedsweep.sh`.
These are recorded settings, not a fresh check of the live cluster:

- Login: `ssh psytp7@jarvis.cs.nott.ac.uk`, from the university network/VPN.
- Repository: `/home/psytp7/ISBI_2026`; environment: `.venv/bin/activate`.
- Shards: `/data/psytp7/wds_shards_train_raw` and `wds_shards_val_raw`.
- Stage 2 default: partition `amp48`, one GPU, 8 CPUs, 64 GB host RAM, 36 hours.
- Existing sweeps exclude `colossus` because of recorded CUDA initialization failures.
- Run compute via Slurm, not on the login node. Keep `#SBATCH` lines contiguous.

`scripts/train_3.sh` follows this setup, uses logs relative to the submission directory,
and explicitly checks CUDA before starting. It defaults to the seed-42 checkpoint path
recorded in `analysis/out/probs_ab_flip/manifest_20261001_155521.json`; verify that file
still exists on the cluster. Override `PROJECT_DIR`, `VENV_DIR`, `CKPT`, `TRAIN`, `VAL`,
or `META` through environment variables if paths differ.

After transferring the new files and completing the data preflight described below:

```bash
ssh psytp7@jarvis.cs.nott.ac.uk
cd /home/psytp7/ISBI_2026
mkdir -p logs
sinfo
sbatch --time=1:00:00 scripts/train_3.sh minority runs/s3_smoke --updates 5 --val-limit-batches 2
# After the smoke test succeeds:
sbatch -J s3_A1_s42 scripts/train_3.sh natural runs/s3_A1_s42
sbatch -J s3_A2_s42 scripts/train_3.sh minority runs/s3_A2_s42
squeue -u "$USER"
# Replace JOB_ID with the submitted job number:
tail -f logs/isbi2026_stage3_JOB_ID.out
```

Overridden job names also change the log filename, e.g. `logs/s3_A1_s42_JOB_ID.out`.
Pass `--resume` with the same arm/output/options to restart a saved run. Resource requests
are starting settings inherited from Stage 2, not a measured Stage 3 runtime requirement.
The trainer is single-GPU; requesting additional GPUs does not distribute its training.

For the full decode preflight, first obtain a CPU allocation, then activate `.venv` and
run the preflight command below inside that allocation:

```bash
srun -p general -c 4 --mem=16G --time=2:00:00 --pty bash
cd /home/psytp7/ISBI_2026
source .venv/bin/activate
```

Transfer the local work using an archive or an approved Git workflow; these files have not
been pushed, so pulling origin alone will not retrieve them. Use the university's existing
CUDA-compatible PyTorch environment and repository dependencies. The scripts additionally
use scikit-learn for AP (already used elsewhere in the repository). Keep the same environment
for both arms; save `python -m pip freeze` and GPU details alongside results.

Run from the repository root. Set these Bash variables to real university paths:

```bash
CKPT=/path/to/verified/v3/seed42/model_swa.pth
TRAIN=/path/to/wds_shards_train_raw
VAL=/path/to/wds_shards_val_raw
META=/path/to/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
COMMON=(--checkpoint "$CKPT" --train-shards "$TRAIN" --val-shards "$VAL"
        --train-fold data/train_fold_1.csv --val-fold data/val_fold_1.csv
        --metadata "$META" --batch-size 4 --accum-steps 12 --img-size 768 --seed 42)
```

Use the original metadata CSV for patient identities, not a spreadsheet-rounded patient-ID
export. Patient strings are hashed for grouping. Prediction artifacts still belong in the
restricted research workspace; hashing does not make them anonymous public data.

First audit all shard labels and decode every image without building the model:

```bash
python train/train_3.py "${COMMON[@]}" --arm minority --preflight-only --out runs/s3_preflight
```

Missing images cause a failure and their filenames are recorded in `manifest.json`. Review
the exclusions. If intentional, use a **new output directory** with `--allow-missing` and
carry that option into both arms. Minority prevalence retains the full fold denominator.
Do not silently delete CSV rows to pass this check.

GPU smoke test (separate output; results are explicitly marked as a validation subset):

```bash
python train/train_3.py "${COMMON[@]}" --arm minority --updates 5 --val-limit-batches 2 --out runs/s3_smoke
```

Then matched performance runs, preferably as scheduled single-GPU university jobs:

```bash
python train/train_3.py "${COMMON[@]}" --arm natural --workers 4 --out runs/s3_A1_s42
python train/train_3.py "${COMMON[@]}" --arm minority --workers 4 --out runs/s3_A2_s42
```

Default budget: 2,152 updates x 48 image presentations = 103,296 presentations. This is an
image-budget equivalent of one pass, not one visit to every image in the minority arm.
Initial/midpoint/endpoint EMA predictions are evaluated without TTA. LR is 1e-5, AdamW
weight decay .01, warmup 5%, cosine decay, EMA .999. These are provisional settings.

Resume by repeating the exact command with `--resume`. Configuration, input inventory,
CSV hashes, checkpoint hash and source hashes must match. Worker count is also fixed.
For a planned interruption use `--stop-after 100` with the original update budget; remove
it on resume. Do not change `--updates` to simulate an interruption: it changes the LR schedule.

Draws are regenerated deterministically and sliced at the last completed saved update.
Prefetched batches do not advance resume state. Each sample's occurrence number controls
augmentation. Exact GPU bitwise reproduction is not promised across hardware/library changes.

## Inspect and compare

`manifest.json` records configuration, provenance and exclusions. `planned_exposure.json`
and `actual_exposure.json` report positive exposure, distinct positive patients and repeat
histograms. `metrics.json` has per-class/overall/head/tail AP, positive counts, positive
patients, Brier scores and predicted-positive fractions. `resume.pt` contains the trainable
head, optimizer, scaler, EMA, update count and RNG state. `model_*.pth` are full EMA model
state dictionaries for the existing image-only architecture.

```bash
python analysis/stage3_compare.py runs/s3_A1_s42/predictions_002152.npz runs/s3_A2_s42/predictions_002152.npz > runs/s3_delta.json
python analysis/stage3_compare.py runs/s3_A1_s42/predictions_002152.npz runs/s3_A2_s42/predictions_002152.npz --unseen-only > runs/s3_unseen_delta.json
```

Bootstrap resamples patients jointly across the two arms. Unsupported labels are listed
through the returned group membership; replicates losing all positives for a supported
label are excluded from that group's interval and the valid replicate count is reported.
The unseen subset here includes **all projections**; it is not the previously audited
frontal-only five-tail subset. Do not equate those populations. These intervals describe
validation sampling uncertainty, not variation across training seeds.

Check improvement against both continued-training A1 and initialization. The engineering
guards are not proof of equivalence. Repeat promising comparisons across matched seeds;
new Stage 3 seeds still share the same Stage 2 initialization unless its checkpoint changes.
The existing patient overlap prevents a strong generalization claim. A truly independent
evaluation must exclude patients seen by the initializing Stage 2 checkpoint as well.

## Scope of this first implementation

The existing v3 transforms, model and ASL are imported unchanged. Class weights are
recomputed in verified shard label order. The trainer fails on mismatched labels, decode
errors, nonfinite losses/gradients and incompatible checkpoints. It does not silently skip
samples. Indexed random I/O and worker throughput must be measured on university storage.
Full shard payload hashes are not computed: the manifest uses path/size/mtime inventories,
CSV/checkpoint/source hashes, and per-access label verification.

Stop after Phase A to inspect the evidence. A failure of resampling alone does not logically
disprove metric learning; it triggers a diagnosis rather than automatic construction of
the remaining memory components. No GPU job is submitted by these scripts.
