#!/bin/bash
#SBATCH -J "isbi2026_eval_tta_v5"
#SBATCH -p amp48
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH -o /home/psytp7/logs/%x_%j.out

# NOTE: keep the #SBATCH directives above as one contiguous block with no
# comments interleaved - this cluster's submit filter rejects the job otherwise.
#
# v5 variant: evaluate_tta_v5.py (loads v3 / v4 Markov / v5 prior checkpoints; v5 needs
# --prior-file and --prior-mode with|zero; dumps also write keys.txt = row filenames).
#
# TTA (horizontal-flip) + optional multi-checkpoint ensemble evaluation on the
# internal val shards. Pass one checkpoint for pure TTA, or several to ensemble:
#   sbatch scripts/evaluate_tta_v5.sh train/checkpoint_Triplet_3/.../model_best.pth
#   sbatch scripts/evaluate_tta_v5.sh CKPT_A CKPT_B CKPT_C
#   sbatch scripts/evaluate_tta_v5.sh --no-tta CKPT        # ablate the TTA

set -u

if [ "$#" -eq 0 ]; then
    echo "usage: sbatch scripts/evaluate_tta_v5.sh [--no-tta] <ckpt> [<ckpt> ...]" >&2
    exit 1
fi

echo "=== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) started $(date) ==="
echo "args: $*"
nvidia-smi

source /home/psytp7/ISBI_2026/.venv/bin/activate
cd /home/psytp7/ISBI_2026/evaluate

python evaluate_tta_v5.py "$@"

echo "=== Finished $(date) ==="
