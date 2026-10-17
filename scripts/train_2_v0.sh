#!/bin/bash
#SBATCH -J "isbi2026_train2v0"
#SBATCH -p amp48
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH -o /home/psytp7/logs/%x_%j.out

# NOTE: keep the #SBATCH directives above as one contiguous block with no
# comments interleaved - this cluster's submit filter rejects the job otherwise.
#
# train_2_v0.py = train_2_v2.py with the Triplet Memory removed entirely (no memory bank,
# mining, triplet loss or entropy gate). ASL only, v2 recipe otherwise (512 px, no EMA/SWA).

set -u

echo "=== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) started $(date) ==="
echo "args: $*"
nvidia-smi

source /home/psytp7/ISBI_2026/.venv/bin/activate
cd /home/psytp7/ISBI_2026/train
# colossus sometimes hands out a dead GPU and torch silently falls back to CPU: fail fast instead
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)" \
    || { echo "no usable CUDA GPU on $(hostname); aborting" >&2; exit 3; }

python train_2_v0.py "$@"

echo "=== Finished $(date) ==="
