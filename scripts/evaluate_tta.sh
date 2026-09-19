#!/bin/bash
#SBATCH -J "isbi2026_eval_tta"
#SBATCH -p amp48
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH -o /home/psytp7/logs/%x_%j.out

# NOTE: keep the #SBATCH directives above as one contiguous block with no
# comments interleaved - this cluster's submit filter rejects the job otherwise.
#
# TTA (horizontal-flip) + optional multi-checkpoint ensemble evaluation on the
# internal val shards. Pass one checkpoint for pure TTA, or several to ensemble:
#   sbatch scripts/evaluate_tta.sh train/checkpoint_Triplet_3/.../model_best.pth
#   sbatch scripts/evaluate_tta.sh CKPT_A CKPT_B CKPT_C
#   sbatch scripts/evaluate_tta.sh --no-tta CKPT        # ablate the TTA

set -u

if [ "$#" -eq 0 ]; then
    echo "usage: sbatch scripts/evaluate_tta.sh [--no-tta] <ckpt> [<ckpt> ...]" >&2
    exit 1
fi

echo "=== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) started $(date) ==="
echo "args: $*"
nvidia-smi

source /home/psytp7/ISBI_2026/.venv/bin/activate
cd /home/psytp7/ISBI_2026/evaluate

python evaluate_tta.py "$@"

echo "=== Finished $(date) ==="
