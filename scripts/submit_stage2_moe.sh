#!/bin/bash
# Matched-recipe Mixture-of-Experts baseline and ASL-only control (train/train_2_v5.py).
# Plan: /home/psytp7/.claude/plans/purring-wandering-quiche.md (2026-10-08).
#
# Every run is the twin of an existing Triplet Memory run: same Stage-1 checkpoint, resolution,
# seed, batch/accum, LR schedule, epochs, EMA, SWA, drop-path and class weights. The only change
# is the Stage-2 module/loss. None of these runs has any Triplet Memory: --no-triplet removes the
# triplet loss, the cross-batch memory bank and the metric embedding.
#   moe        --arch moe: MoE in stages 3-4 + decoder FFN after CA (train/convnext_moe.py),
#              ASL + 5e-3 x load-balancing aux
#   moeprior   moe + patient-history Markov prior into the label queries (as v5 qprior)
#   asl        plain ConvNeXt2, ASL only (no MoE, no triplet)
#
# Twins (Triplet Memory job):
#   moe:768:42 moe:768:1024 asl:768:42 asl:768:1024 moeprior:768:42 moeprior:768:1024
#       -> push/img768_dp01_seed{42,1024} (49000/48999), prior/img768_qprior_seed{42,1024}
#   moe:896:86 moe:1024:86  -> push/img896_dp01_seed86 (48998), push/img1024_dp01_seed86 (48997)
#   moe:640:86              -> res_sweep/img640_seed86 (47094; 8 epochs, patience 3, no dp)
#
#   SMOKE=1 ./submit_stage2_moe.sh moe:1024:86 moeprior:768:42    # 40-batch check first
#   ./submit_stage2_moe.sh                                        # all 9 runs
# out_dir: train/checkpoint_MoE/<run>_img<px>_seed<S>/Model_run
# Evaluate with scripts/evaluate_tta_v5.sh (MoE checkpoints load strictly; moeprior needs
# --prior-file and --prior-mode with|zero).
PARTITIONS="${PARTITIONS:-amp48,ada24}"
EXCLUDE="${EXCLUDE:-colossus}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
TIME="${TIME:-36:00:00}"
SMOKE="${SMOKE:-0}"
BS="${BS:-4}"
ACCUM="${ACCUM:-12}"
PRIOR_FILE="${PRIOR_FILE:-/home/psytp7/ISBI_2026/analysis/out/patient_markov/priors.pt}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoint3/Model_20260119_062652/model_best.pth}"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNS=("$@")
[ "$#" -eq 0 ] && RUNS=(moe:768:42 moe:768:1024 moe:896:86 moe:1024:86 moe:640:86
                       moeprior:768:42 moeprior:768:1024 asl:768:42 asl:768:1024)

BASE=(--accum-steps "$ACCUM" --batch-size "$BS" --monitor mAP --no-triplet
      --class-weight-order shard --swa-last-k 3 --lr 1e-4 --stage1-ckpt "$STAGE1_CKPT")
PUSH=(--epochs 4 --patience 2 --sched-epochs 8 --drop-path-rate 0.1)   # push / v5 prior recipe
RES640=(--epochs 8 --patience 3)                                       # res_sweep recipe
SMOKE_ARGS=()
[ "$SMOKE" = 1 ] && SMOKE_ARGS=(--epochs 1 --patience 1 --limit-train-batches 40 --limit-val-batches 20 --log-every 10)

for spec in "${RUNS[@]}"; do
    IFS=: read -r run px seed <<< "$spec"
    case "$run" in
        moe)      arm=(--arch moe) ;;
        moeprior) [ -f "$PRIOR_FILE" ] || { echo "missing $PRIOR_FILE" >&2; exit 1; }
                  arm=(--arch moe --prior-file "$PRIOR_FILE" --prior-dropout 0.3 --prior-lr-mult 10) ;;
        asl)      arm=(--arch convnext2) ;;
        *) echo "unknown run: $run (moe|moeprior|asl)" >&2; exit 1 ;;
    esac
    if [ "$px" = 640 ]; then recipe=("${RES640[@]}"); else recipe=("${PUSH[@]}"); fi
    tag=""; [ "$SMOKE" = 1 ] && tag="_smoke"
    name="isbi2026_${run}_img${px}_s${seed}${tag}"
    jid=$(sbatch --parsable -J "$name" \
        -p "$PARTITIONS" ${EXCLUDE:+-x "$EXCLUDE"} -c "$CPUS" --mem="$MEM" --time="$TIME" \
        "$SCRIPT_DIR/train_2_v5.sh" \
            --seed "$seed" --img-size "$px" \
            --out-dir "checkpoint_MoE/${run}_img${px}_seed${seed}${tag}/Model_run" \
            "${BASE[@]}" "${recipe[@]}" "${arm[@]}" "${SMOKE_ARGS[@]}")
    echo "$spec -> job $jid | /home/psytp7/logs/${name}_${jid}.out"
done
