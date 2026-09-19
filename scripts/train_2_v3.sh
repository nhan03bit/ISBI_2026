#!/bin/bash
#SBATCH -J "isbi2026_train2v3"
#SBATCH -p amp48
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH -o /home/psytp7/logs/%x_%j.out

# NOTE: keep the #SBATCH directives above as one contiguous block with no
# comments interleaved - this cluster's submit filter rejects the job otherwise.
# %x is the job name, so submit_stage2_v3_sweep.sh gives each grid point its own
# -J and therefore its own %x log.
#
# Track A ("new tricks"): train_2_v3.py adds, over train_2_v2.py -
#   - metric embedding = mean-pooled decoded query embeddings (--metric-embed query)
#   - XBM memory bank 4096 (--memory-size)
#   - head-class-aware positive mining (sample_triplets_v13, --head-classes)
#   - tunable triplet weight (--triplet-lambda)
#   - weight EMA (--ema-decay); EMA weights are what gets evaluated + saved
#   - multi-epoch budget (--epochs 8 --patience 3 --lr 1e-4 --warmup-frac 0.05)
# Resources are usually overridden by submit_stage2_v3_sweep.sh (-p amp48,ada24).

set -u

echo "=== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) started $(date) ==="
echo "args: $*"
nvidia-smi

source /home/psytp7/ISBI_2026/.venv/bin/activate
cd /home/psytp7/ISBI_2026/train

python train_2_v3.py "$@"

echo "=== Finished $(date) ==="
