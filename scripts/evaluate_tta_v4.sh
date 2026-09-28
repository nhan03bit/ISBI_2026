#!/bin/bash
#SBATCH -J "isbi2026_eval_tta_v4"
#SBATCH -p amp48
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH -o /home/psytp7/logs/%x_%j.out

# NOTE: keep the #SBATCH directives above as one contiguous block with no
# comments interleaved - this cluster's submit filter rejects the job otherwise.
#
# v4 variant: evaluate_tta_v4.py (loads Stage-2 v4 Markov checkpoints correctly;
# v3 checkpoints load as before, so the two can be mixed in one ensemble).
#
# TTA (horizontal-flip) + optional multi-checkpoint ensemble evaluation on the
# internal val shards. Pass one checkpoint for pure TTA, or several to ensemble:
#   sbatch scripts/evaluate_tta_v4.sh train/checkpoint_Triplet_3/.../model_best.pth
#   sbatch scripts/evaluate_tta_v4.sh CKPT_A CKPT_B CKPT_C
#   sbatch scripts/evaluate_tta_v4.sh --no-tta CKPT        # ablate the TTA

set -u

if [ "$#" -eq 0 ]; then
    echo "usage: sbatch scripts/evaluate_tta_v4.sh [--no-tta] <ckpt> [<ckpt> ...]" >&2
    exit 1
fi

echo "=== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) started $(date) ==="
echo "args: $*"
nvidia-smi

source /home/psytp7/ISBI_2026/.venv/bin/activate
cd /home/psytp7/ISBI_2026/evaluate

python evaluate_tta_v4.py "$@"

echo "=== Finished $(date) ==="
