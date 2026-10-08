#!/bin/bash
#SBATCH -J "isbi2026_stage3"
#SBATCH -p amp48
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH -x colossus
#SBATCH -o logs/%x_%j.out

# Keep Slurm directives contiguous (existing cluster submit-filter requirement).
# Submit from repository root after mkdir -p logs. Extra trainer args pass through.
# Usage: sbatch scripts/train_3.sh natural runs/s3_A1_s42
#        sbatch --time=1:00:00 scripts/train_3.sh minority runs/s3_smoke --updates 5 --val-limit-batches 2
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT_DIR"
source "${VENV_DIR:-$PROJECT_DIR/.venv}/bin/activate"
export PYTHONUNBUFFERED=1

if [ "$#" -lt 2 ]; then
    echo "usage: $0 natural|minority OUTPUT_DIR [train_3.py options]" >&2
    exit 2
fi
ARM="$1"
OUT="$2"
shift 2

echo "Job ${SLURM_JOB_ID:-interactive} on $(hostname) at $(date)"
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable; refusing CPU fallback"; print(torch.__version__, torch.cuda.get_device_name(0)); torch.zeros(1, device="cuda"); torch.cuda.synchronize()'

CKPT="${CKPT:-$PROJECT_DIR/train/checkpoint_Triplet_3/push/img768_dp01_seed42/Model_run/model_swa.pth}"
TRAIN="${TRAIN:-/data/psytp7/wds_shards_train_raw}"
VAL="${VAL:-/data/psytp7/wds_shards_val_raw}"
META="${META:-$PROJECT_DIR/data/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv}"

python train/train_3.py \
    --checkpoint "$CKPT" --train-shards "$TRAIN" --val-shards "$VAL" \
    --train-fold data/train_fold_1.csv --val-fold data/val_fold_1.csv \
    --metadata "$META" --arm "$ARM" --out "$OUT" \
    --batch-size 4 --accum-steps 12 --img-size 768 --workers 4 --seed 42 "$@"
