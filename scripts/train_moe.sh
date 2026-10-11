#!/bin/bash
#SBATCH -J "isbi2026_moerecon"
#SBATCH -p amp48
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH -o /home/psytp7/logs/%x_%j.out

# NOTE: keep the #SBATCH directives above as one contiguous block with no
# comments interleaved - this cluster's submit filter rejects the job otherwise.
#
# Reconstruction of the original MoE Stage 2 (b12): train_moe.py, run from the repo
# root because train.py reads ./logs and imports the root convnext*.py / moe.py.
# Resources are overridden by submit_moe_seedsweep.sh.

set -u

echo "=== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) started $(date) ==="
echo "args: $*"
nvidia-smi

source /home/psytp7/ISBI_2026/.venv/bin/activate
cd /home/psytp7/ISBI_2026
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_moe.py "$@"

echo "=== Finished $(date) ==="
