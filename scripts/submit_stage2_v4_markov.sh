#!/bin/bash
# Stage-2 v4: Markov label-refine layer on the ML-Decoder logits (train/train_2_v4.py,
# train/markov_layer.py). Background: docs/result/2026-09-26-markov-applicability.md - a
# FIXED random-walk smoothing over label co-occurrence hurt the best ensemble offline
# (-0.0006..-0.0053 mAP); this tests the TRAINABLE version, which starts as the identity
# (gate = bias = 0) and can only contribute what training finds useful.
#
# Recipe = the push-wave img768 recipe (drop-path 0.1, 4 of 8 cosine epochs, SWA of 3 EMA
# epochs), seed 42, so every arm is matched to push/img768_dp01_seed42
# (job 49000: best 0.4585 @ep3, SWA 0.4600; flip 0.4650; frontal & unseen-patient 0.4706).
#
# Runs (RUNS selects a subset, default all):
#   mk1       K = 1 step, learnable transition matrix
#   mk2       K = 2 steps, learnable transition matrix
#   mk1fix    K = 1 step, transition frozen at the co-occurrence init (gate/bias only)
#
#   SMOKE=1 ./submit_stage2_v4_markov.sh     # 40-batch check first
#   ./submit_stage2_v4_markov.sh
# Evaluate with scripts/evaluate_tta_v4.sh (evaluate_tta.sh would drop the layer).
PARTITIONS="${PARTITIONS:-amp48,ada24}"
EXCLUDE="${EXCLUDE:-colossus}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
TIME="${TIME:-36:00:00}"
SMOKE="${SMOKE:-0}"
RUNS="${RUNS:-mk1 mk2 mk1fix}"
SEED="${SEED:-42}"
IMG="${IMG:-768}"
BS="${BS:-4}"
ACCUM="${ACCUM:-12}"
LR_MULT="${LR_MULT:-10}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoint3/Model_20260119_062652/model_best.pth}"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMMON=(--accum-steps "$ACCUM" --batch-size "$BS" --monitor mAP
        --memory-size 2048 --metric-embed query --head-classes 5
        --triplet-lambda 0.1 --class-weight-order shard --swa-last-k 3
        --lr 1e-4 --epochs 4 --patience 2 --sched-epochs 8 --drop-path-rate 0.1
        --img-size "$IMG" --markov-lr-mult "$LR_MULT"
        --stage1-ckpt "$STAGE1_CKPT")
SMOKE_ARGS=()
if [ "$SMOKE" = 1 ]; then
    SMOKE_ARGS=(--epochs 1 --patience 1 --limit-train-batches 40 --limit-val-batches 20 --log-every 10)
fi

submit() {  # name out-subdir train-args...
    local name=$1 sub=$2; shift 2
    local tag=""; [ "$SMOKE" = 1 ] && tag="_smoke"
    local jid
    jid=$(sbatch --parsable \
        -J "isbi2026_v4markov_${name}_s${SEED}${tag}" \
        -p "$PARTITIONS" ${EXCLUDE:+-x "$EXCLUDE"} -c "$CPUS" --mem="$MEM" --time="$TIME" \
        "$SCRIPT_DIR/train_2_v4.sh" \
            --seed "$SEED" \
            --out-dir "checkpoint_Triplet_3/markov/${sub}${tag}/Model_run" \
            "${COMMON[@]}" "$@" "${SMOKE_ARGS[@]}")
    echo "$name seed=$SEED -> job $jid | /home/psytp7/logs/isbi2026_v4markov_${name}_s${SEED}${tag}_${jid}.out"
}

echo "runs: $RUNS | img=$IMG seed=$SEED bs=$BS accum=$ACCUM lr_mult=$LR_MULT | smoke=$SMOKE | partitions=$PARTITIONS"
echo

for R in $RUNS; do
  case "$R" in
    mk1)    submit mk1    "img${IMG}_mk1_seed${SEED}"    --markov-steps 1 ;;
    mk2)    submit mk2    "img${IMG}_mk2_seed${SEED}"    --markov-steps 2 ;;
    mk1fix) submit mk1fix "img${IMG}_mk1fix_seed${SEED}" --markov-steps 1 --markov-fixed-transition ;;
    *) echo "unknown run: $R" >&2; exit 1 ;;
  esac
done

echo
echo "watch:  python3 scripts/mon.py 'isbi2026_v4markov_*'   |  grep 'markov:' <log>  (gate magnitudes per epoch)"
