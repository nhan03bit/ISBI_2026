#!/bin/bash
# Submit three matched Stage-2 arms. No minority sampling or history/Markov module.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

if [ "${1:-}" = "--worker" ]; then
    ARM="$2"; OUTPUT="$3"; SEED="$4"; CHECKPOINT="$5"
    source "${VENV_DIR:-$ROOT/.venv}/bin/activate"
    export PYTHONUNBUFFERED=1
    nvidia-smi
    python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; torch.zeros(1, device="cuda"); torch.cuda.synchronize()'
    EXTRA=()
    case "$ARM" in
        baseline) EXTRA=(--triplet-lambda 0 --anchor-rule all) ;;
        triplet) EXTRA=(--triplet-lambda 0.1 --anchor-rule all) ;;
        entropy) EXTRA=(--triplet-lambda 0.1 --anchor-rule mean) ;;
        *) echo "Unknown arm: $ARM" >&2; exit 2 ;;
    esac
    SMOKE_ARGS=()
    if [ "${SMOKE:-0}" = 1 ]; then
        SMOKE_ARGS=(--epochs 1 --limit-train-batches 24 --limit-val-batches 2)
    fi
    exec python train/train_2_v3.py \
        --stage1-ckpt "$CHECKPOINT" --strict-init --fixed-budget \
        --seed "$SEED" --aug-base-seed 42 --out-dir "$OUTPUT" \
        --img-size 768 --batch-size 4 --accum-steps 12 \
        --epochs 4 --sched-epochs 8 --warmup-frac 0.05 --lr 1e-4 \
        --weight-decay 0.01 --drop-path-rate 0.1 --freeze-backbone-stages 0 --llrd 1 \
        --memory-size 2048 --embedding-dim 768 --metric-embed query --head-classes 5 \
        --margin 0.5 --ema-decay 0.999 --swa-last-k 0 --class-weight-order shard \
        --gamma-neg 2 --gamma-pos 0 --num-workers 8 --val-num-workers 2 --shuffle-buf 400 \
        --train-shards-dir "${TRAIN:-/data/psytp7/wds_shards_train_raw}" \
        --val-shards-dir "${VAL:-/data/psytp7/wds_shards_val_raw}" \
        "${EXTRA[@]}" "${SMOKE_ARGS[@]}"
fi

CHECKPOINT="${CKPT:-$ROOT/train/checkpoint3/Model_20260119_062652/model_best.pth}"
test -f "$CHECKPOINT" || { echo "Stage-1 checkpoint missing: $CHECKPOINT" >&2; exit 1; }
CHECKPOINT="$(realpath "$CHECKPOINT")"
RUN="$ROOT/runs/entropy_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN/logs"
echo "Run directory: $RUN"
for SEED in ${SEEDS:-42}; do
    for ARM in baseline triplet entropy; do
        printf -v WRAP '%q ' bash "$ROOT/scripts/run_entropy_ablation.sh" --worker "$ARM" "$RUN/seed_$SEED/$ARM" "$SEED" "$CHECKPOINT"
        JOB=$(sbatch --parsable --job-name="entropy_${ARM}_s${SEED}" \
            --partition="${GPU_PARTITION:-amp48}" --gpus=1 --cpus-per-task=8 --mem=64G \
            --time="${TIME:-36:00:00}" --exclude="${EXCLUDE:-colossus}" \
            --output="$RUN/logs/${ARM}_s${SEED}_%j.out" --chdir="$ROOT" --wrap="$WRAP")
        printf '%s\t%s\t%s\n' "$SEED" "$ARM" "$JOB" | tee -a "$RUN/jobs.tsv"
    done
done
echo "Monitor: squeue -u \$USER; tail -f $RUN/logs/*.out"
echo "After all jobs complete: .venv/bin/python analysis/entropy_ablation_report.py $RUN"
