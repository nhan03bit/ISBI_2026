#!/bin/bash
#SBATCH -J isbi2026_predict_task1
#SBATCH -p amp16
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH -o /home/psytp7/logs/%x_%j.out

# NOTE: keep the #SBATCH directives above as one contiguous block with no comments interleaved (the cluster's
# submit filter rejects the job otherwise).
#
# Task-1 leaderboard predictions (evaluate/predict_task1.py): template CSV + Task-1 PNG folder -> filled CSV,
# per-model probabilities and a manifest. Eval-only resources (A4000 16G is enough). Pass -x rogue if amp20 is
# added (rogue hung evals 2026-10-16).
#   sbatch scripts/predict_task1.sh --images-dir DIR --template labels.csv --out OUT.csv CKPT@768 [CKPT@768 ...]

set -u
echo "=== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) started $(date) ==="
echo "args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv

source /home/psytp7/ISBI_2026/.venv/bin/activate
cd /home/psytp7/ISBI_2026
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)" \
    || { echo "no usable CUDA GPU on $(hostname); aborting" >&2; exit 3; }

python -I evaluate/predict_task1.py "$@"
echo "=== Finished $(date) ==="
