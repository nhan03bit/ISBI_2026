#!/bin/bash
#SBATCH -J "isbi2026_train2v2"
#SBATCH -p amp48
#SBATCH -G 1
#SBATCH -c 12
#SBATCH --mem=100G
#SBATCH --time=45:00:00
#SBATCH -o /home/psytp7/logs/%x_%j.out

# NOTE: keep the #SBATCH directives above as one contiguous block with no
# comments interleaved - this cluster's submit filter rejects the job otherwise.
# %x is the job name, so `sbatch -J isbi2026_train2v2_accum7 ...` gives each
# run of the sweep its own log instead of three jobs sharing one name.

set -u

echo "=== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) started $(date) ==="
echo "args: $*"
nvidia-smi

source /home/psytp7/ISBI_2026/.venv/bin/activate
cd /home/psytp7/ISBI_2026/train

python train_2_v2.py "$@"

echo "=== Finished $(date) ==="
