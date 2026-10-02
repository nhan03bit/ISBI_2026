#!/bin/bash
# Stage-2 v5: patient-prior label-query conditioning ("B-early", train/train_2_v5.py,
# train/prior_conditioning.py). Plan: docs/plan/2026-09-30-markov-ab-training-plan.md.
# Each image's prior-study state goes through the patient Markov chain
# (analysis/patient_prior.py -> analysis/out/patient_markov/priors.pt) and the resulting
# 31-value vector offsets the ML-Decoder label queries. Zero-init, so every run starts
# exactly as v3; images without a prior always get a zero offset. Train priors use
# train-fold history only.
#
# Recipe = the push-wave img768 recipe (drop-path 0.1, 4 of 8 cosine epochs, SWA of 3 EMA
# epochs), matched to v3 push/img768_dp01_seed{42,1024} (jobs 49000 / 48999) and to the
# v4 Markov arms (seed 42).
#
# Runs (RUNS selects a subset, default qprior):
#   qprior    label-query conditioning, prior dropout 0.3, conditioner lr x10
# SEEDS selects seeds (default "42 1024").
#
#   SMOKE=1 SEEDS=42 ./submit_stage2_v5_prior.sh     # 40-batch check first
#   ./submit_stage2_v5_prior.sh
# Evaluate with scripts/evaluate_tta_v5.sh (needs --prior-file).
PARTITIONS="${PARTITIONS:-amp48,ada24}"
EXCLUDE="${EXCLUDE:-colossus}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
TIME="${TIME:-36:00:00}"
SMOKE="${SMOKE:-0}"
RUNS="${RUNS:-qprior}"
SEEDS="${SEEDS:-42 1024}"
IMG="${IMG:-768}"
BS="${BS:-4}"
ACCUM="${ACCUM:-12}"
PRIOR_FILE="${PRIOR_FILE:-/home/psytp7/ISBI_2026/analysis/out/patient_markov/priors.pt}"
PRIOR_DROPOUT="${PRIOR_DROPOUT:-0.3}"
PRIOR_LR_MULT="${PRIOR_LR_MULT:-10}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoint3/Model_20260119_062652/model_best.pth}"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$PRIOR_FILE" ] || { echo "missing $PRIOR_FILE - run: python analysis/patient_prior.py" >&2; exit 1; }

COMMON=(--accum-steps "$ACCUM" --batch-size "$BS" --monitor mAP
        --memory-size 2048 --metric-embed query --head-classes 5
        --triplet-lambda 0.1 --class-weight-order shard --swa-last-k 3
        --lr 1e-4 --epochs 4 --patience 2 --sched-epochs 8 --drop-path-rate 0.1
        --img-size "$IMG" --stage1-ckpt "$STAGE1_CKPT")
SMOKE_ARGS=()
if [ "$SMOKE" = 1 ]; then
    SMOKE_ARGS=(--epochs 1 --patience 1 --limit-train-batches 40 --limit-val-batches 20 --log-every 10)
fi

submit() {  # name seed out-subdir train-args...
    local name=$1 seed=$2 sub=$3; shift 3
    local tag=""; [ "$SMOKE" = 1 ] && tag="_smoke"
    local jid
    jid=$(sbatch --parsable \
        -J "isbi2026_v5prior_${name}_s${seed}${tag}" \
        -p "$PARTITIONS" ${EXCLUDE:+-x "$EXCLUDE"} -c "$CPUS" --mem="$MEM" --time="$TIME" \
        "$SCRIPT_DIR/train_2_v5.sh" \
            --seed "$seed" \
            --out-dir "checkpoint_Triplet_3/prior/${sub}${tag}/Model_run" \
            "${COMMON[@]}" "$@" "${SMOKE_ARGS[@]}")
    echo "$name seed=$seed -> job $jid | /home/psytp7/logs/isbi2026_v5prior_${name}_s${seed}${tag}_${jid}.out"
}

echo "runs: $RUNS | seeds: $SEEDS | img=$IMG bs=$BS accum=$ACCUM | prior dropout=$PRIOR_DROPOUT lr_mult=$PRIOR_LR_MULT | smoke=$SMOKE | partitions=$PARTITIONS"
echo

for S in $SEEDS; do
  for R in $RUNS; do
    case "$R" in
      qprior) submit qprior "$S" "img${IMG}_qprior_seed${S}" \
                  --prior-file "$PRIOR_FILE" --prior-dropout "$PRIOR_DROPOUT" --prior-lr-mult "$PRIOR_LR_MULT" ;;
      *) echo "unknown run: $R" >&2; exit 1 ;;
    esac
  done
done

echo
echo "watch:  python3 scripts/mon.py 'isbi2026_v5prior_*'   |  grep 'prior:' <log>  (conditioner norms per epoch)"
