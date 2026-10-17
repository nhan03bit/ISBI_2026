#!/bin/bash
#SBATCH -J isbi2026_smoke_nca
#SBATCH -p amp16
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH -o /home/psytp7/logs/%x_%j.out

# NOTE: keep the #SBATCH directives above as one contiguous block with no comments interleaved (the cluster's
# submit filter rejects the job otherwise).
#
# 2026-10-16 checks for train_2_v5.py --metric-loss nca (train/class_memory.py), on one GPU at 512 px:
#   1. parity: the pre-change copy (train/_parity_v5_orig.py) and the new script with default flags must give
#      bit-identical train losses and EMA weights (the triplet path is untouched);
#   2. nca: the new term runs, stays finite and is live (anchors/step, gradient ratio, scale), with the warm-up
#      lowered so it switches on within the short run.
set -u
echo "=== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) started $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv
source /home/psytp7/ISBI_2026/.venv/bin/activate
cd /home/psytp7/ISBI_2026/train
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 3)" || { echo "no CUDA" >&2; exit 3; }

OUT=checkpoint_smoke/nca_$SLURM_JOB_ID
COMMON=(--seed 42 --accum-steps 12 --batch-size 4 --monitor mAP --memory-size 2048 --metric-embed query
        --head-classes 5 --class-weight-order shard --swa-last-k 0 --lr 1e-4 --epochs 1 --patience 1
        --sched-epochs 8 --drop-path-rate 0.1 --img-size 512 --log-every 12 --num-workers 2 --val-num-workers 1
        --limit-val-batches 4 --stage1-ckpt checkpoint3/Model_20260119_062652/model_best.pth
        --triplet-lambda 0.1 --anchor-rule mean)

python _parity_v5_orig.py "${COMMON[@]}" --limit-train-batches 48 --out-dir "$OUT/orig" || exit 1
python train_2_v5.py "${COMMON[@]}" --limit-train-batches 48 --out-dir "$OUT/new_default" || exit 1
python - "$OUT" <<'EOF'
import json, sys, torch
out = sys.argv[1]
h = [json.load(open(f"{out}/{d}/history.json")) for d in ("orig", "new_default")]
print("train_loss orig", h[0]["train_loss"], "new", h[1]["train_loss"], "equal:", h[0]["train_loss"] == h[1]["train_loss"])
a, b = (torch.load(f"{out}/{d}/model_best.pth", map_location="cpu") for d in ("orig", "new_default"))
same = all(torch.equal(a[k], b[k]) for k in a) and a.keys() == b.keys()
print("EMA weights bit-identical:", same, "| val mAP", h[0]["mAP"], h[1]["mAP"])
sys.exit(0 if same and h[0]["train_loss"] == h[1]["train_loss"] else 4)
EOF
echo "parity exit $?"

python train_2_v5.py "${COMMON[@]}" --limit-train-batches 120 --metric-loss nca --nca-warm 64 \
    --out-dir "$OUT/nca" || exit 1
grep -h "nca" "$OUT/nca/history.json" >/dev/null && python -c "
import json; h = json.load(open('$OUT/nca/history.json')); print('nca_stats', h.get('nca_stats')); print('train_loss', h['train_loss'], 'val mAP', h['mAP'])"
nvidia-smi --query-gpu=memory.used --format=csv
echo "=== Finished $(date) ==="
