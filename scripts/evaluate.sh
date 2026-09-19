#!/bin/bash
#SBATCH -J "isbi2026_evaluate"
#SBATCH -p amp48
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH -o /home/psytp7/logs/%x_%j.out

# NOTE: keep the #SBATCH directives above as one contiguous block with no
# comments interleaved - this cluster's submit filter rejects the job otherwise.
#
# Usage: sbatch evaluate.sh <path/to/model_best.pth> [extra evaluate.py args...]
#   e.g. sbatch evaluate.sh checkpoint_Triplet_2/accum_6/seed_86/Model_.../model_best.pth
#        sbatch evaluate.sh checkpoint_Triplet_2/accum_8/seed_42/Model_.../model_best.pth --batch-size 16

set -u

CKPT="${1:-}"
if [ -z "$CKPT" ]; then
    echo "usage: sbatch evaluate.sh <path/to/model_best.pth> [extra evaluate.py args...]" >&2
    exit 1
fi
shift

echo "=== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) started $(date) ==="
echo "checkpoint: $CKPT"
echo "extra args: $*"
nvidia-smi

source /home/psytp7/ISBI_2026/.venv/bin/activate
cd /home/psytp7/ISBI_2026/evaluate

python evaluate.py --checkpoint "$CKPT" "$@"

echo "=== Finished $(date) ==="
