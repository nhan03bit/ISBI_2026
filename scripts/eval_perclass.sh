#!/bin/bash
#SBATCH -J "isbi2026_eval_perclass"
#SBATCH -p amp48,ada24
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH -o /home/psytp7/logs/%x_%j.out

# NOTE: keep the #SBATCH directives above as one contiguous block with no
# comments interleaved - this cluster's submit filter rejects the job otherwise.
#
# Per-class / head-medium-tail evaluation of one checkpoint. Writes
#   <ckpt_dir>/eval_result[.<tag>].json   (macro + group_mAP + per_class)
#   <ckpt_dir>/preds[.<tag>].npz          (probs, labels, class_names)
# next to the checkpoint, for analysis/per_class_report.py and
# analysis/cooccurrence.py.
#
# Usage:
#   sbatch eval_perclass.sh <path/to/model_best.pth> <img_size> [tag] [extra args]
#   sbatch eval_perclass.sh train/checkpoint3/Model_20260119_062652/model_best.pth 512 res512
#   sbatch eval_perclass.sh train/checkpoint3/Model_20260119_062652/model_best.pth 384 res384

set -u

CKPT="${1:-}"
IMG="${2:-512}"
TAG="${3:-res${IMG}}"
if [ -z "$CKPT" ]; then
    echo "usage: sbatch eval_perclass.sh <model_best.pth> <img_size> [tag] [extra args]" >&2
    exit 1
fi
shift $(( $# >= 3 ? 3 : $# ))

echo "=== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) started $(date) ==="
echo "checkpoint : $CKPT"
echo "img_size   : $IMG    tag: $TAG"
nvidia-smi

source /home/psytp7/ISBI_2026/.venv/bin/activate
cd /home/psytp7/ISBI_2026/evaluate

python evaluate.py \
    --checkpoint "$CKPT" \
    --img-size "$IMG" \
    --tag "$TAG" \
    --save-preds \
    --num-workers 8 \
    "$@"

echo "=== Finished $(date) ==="
