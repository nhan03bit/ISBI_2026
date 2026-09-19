#!/bin/bash
#SBATCH -J "isbi2026_stage1_v4"
#SBATCH -p amp20,amp48
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=6-00:00:00
#SBATCH -o /home/psytp7/logs/%x_%j.out

# NOTE: keep the #SBATCH directives above as one contiguous block with no
# comments interleaved - this cluster's submit filter rejects the job otherwise.
#
# Tier 2: stronger Stage-1. train.py now has a CLI (see parse_args). Pass the
# recipe as args:
#   sbatch scripts/train_stage1_v4.sh --backbone-init 22k --img-size 512 \
#       --drop-path-rate 0.1 --weight-decay 0.03 --batch-size 16 \
#       --epochs 30 --patience 10 --out-dir ./checkpoint/stage1_v4_22k512dp
# No args -> the historical 384/1k/full-30 recipe.
#
# Auto-resumes from <out-dir>/last.pth if the job is requeued.

set -euo pipefail
PROJECT_DIR="/home/psytp7/ISBI_2026"
cd "$PROJECT_DIR"

echo "=== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) started $(date) ==="
echo "args: $*"
nvidia-smi

source "$PROJECT_DIR/.venv/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train.py "$@"

echo "=== Finished $(date) ==="
