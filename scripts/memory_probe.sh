#!/bin/bash
#SBATCH -J isbi2026_memprobe
#SBATCH -p amp16
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH -o /home/psytp7/logs/%x_%j.out

# NOTE: keep the #SBATCH directives above as one contiguous block with no comments interleaved (the cluster's
# submit filter rejects the job otherwise).
#
# Memory readout probe (analysis/memory_readout_probe.py): one no-TTA pass over the train and val shards, per-class
# and pooled kNN over the train memory excluding same-patient images, prototype scores. Eval-only resources (A4000
# 16G is enough: ~6 GB of embeddings stay on the GPU). Pass -x rogue if amp20 is added (rogue hung evals 2026-10-16).
#   sbatch -J isbi2026_memprobe_asl scripts/memory_probe.sh --tag asl_seeded \
#       train/checkpoint_ASL/v0_768_seeded/seed_42/Model_run/model_swa.pth
#   sbatch ... scripts/memory_probe.sh --tag smoke --limit-batches 20 --no-save-emb --out /data/psytp7/memprobe_smoke CKPT

set -u
echo "=== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) started $(date) ==="
echo "args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv

source /home/psytp7/ISBI_2026/.venv/bin/activate
cd /home/psytp7/ISBI_2026
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)" \
    || { echo "no usable CUDA GPU on $(hostname); aborting" >&2; exit 3; }

python -I analysis/memory_readout_probe.py "$@"
echo "=== Finished $(date) ==="
