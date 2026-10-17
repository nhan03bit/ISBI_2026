#!/bin/bash
#SBATCH -J isbi2026_shiftprobe
#SBATCH -p amp16
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH -o /home/psytp7/logs/%x_%j.out

# NOTE: keep the #SBATCH directives above as one contiguous block with no comments interleaved (the cluster's
# submit filter rejects the job otherwise).
#
# Task-1 image-shift probe (analysis/task1_shift_probe.py): on internal val, the cost of independent per-channel
# noise like the Task-1 PNGs' and whether the channel mean (predict_task1.py --channel-mean) recovers it.
# Eval-only resources (A4000 16G is enough).
#   sbatch scripts/task1_shift_probe.sh --sigma 20 --limit 6000 \
#       train/checkpoint_Triplet_3/unified768/c/seed_42/Model_run/model_swa.pth

set -u
echo "=== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) started $(date) ==="
echo "args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv

source /home/psytp7/ISBI_2026/.venv/bin/activate
cd /home/psytp7/ISBI_2026
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)" \
    || { echo "no usable CUDA GPU on $(hostname); aborting" >&2; exit 3; }

python -I analysis/task1_shift_probe.py "$@"
echo "=== Finished $(date) ==="
