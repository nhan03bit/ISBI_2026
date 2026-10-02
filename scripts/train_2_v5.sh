#!/bin/bash
#SBATCH -J "isbi2026_train2v5"
#SBATCH -p amp48
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH -o /home/psytp7/logs/%x_%j.out

# NOTE: keep the #SBATCH directives above as one contiguous block with no
# comments interleaved - this cluster's submit filter rejects the job otherwise.
# %x is the job name, so each submit gives its own log.
#
# Stage-2 v5 = train_2_v4.py + optional patient-prior label-query conditioning
# (--prior-file, train/prior_conditioning.py). Without --prior-file it is
# identical to v4. Launcher: scripts/submit_stage2_v5_prior.sh.
# Evaluate v5 checkpoints with evaluate/evaluate_tta_v5.py (it needs --prior-file).

set -u

echo "=== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) started $(date) ==="
echo "args: $*"
nvidia-smi

source /home/psytp7/ISBI_2026/.venv/bin/activate
cd /home/psytp7/ISBI_2026/train

python train_2_v5.py "$@"

echo "=== Finished $(date) ==="
