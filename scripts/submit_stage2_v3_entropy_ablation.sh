#!/bin/bash
# Entropy ablation for train_2_v3.py at 512 px (supervisor's missing validation, 2026-10-08).
# Plan: docs/plan/2026-10-08-entropy-ablation-isbi-submit-plan.md
#
# baseline -> triplet/memory bank WITHOUT entropy -> triplet/memory bank WITH entropy, with the
# resolution, schedule, Stage-1 checkpoint and seeds of the v3 seed sweep held fixed. The args are
# exactly those of scripts/submit_stage2_v3_seedsweep.sh (jobs 46472-47099 = the entropy arm C,
# checkpoint_Triplet_3/seedsweep/seed_<S>); each arm changes one thing:
#   asl       --triplet-lambda 0       ASL only: the bank and mining still run, zero gradient (A)
#   nogate    --anchor-rule all        triplet + bank, every sample is an anchor (B)
#   randgate  --anchor-rule random     triplet + bank, random anchors, same count as the gate (B')
#   gate      --anchor-rule mean       the entropy-gated method itself; reruns of existing seeds as a
#                                      reproducibility check (C)
# Supersedes submit_stage2_v3_nolam_seedsweep.sh (never submitted beyond its smoke job 51099).
#
# Each training job is followed (afterok) by scripts/eval_perclass.sh on its model_best.pth at
# 512 px -> eval_result.<arm>_s<S>.json + preds.<arm>_s<S>.npz next to the checkpoint.
#
#   SMOKE=1 ./submit_stage2_v3_entropy_ablation.sh asl:42 nogate:42 randgate:42 gate:42
#   ./submit_stage2_v3_entropy_ablation.sh                  # all 20 runs (order below)
#   NO_EVAL=1 ./submit_stage2_v3_entropy_ablation.sh ...    # training only
#   DEPEND=111:222 ./submit_stage2_v3_entropy_ablation.sh   # start only after jobs 111 and 222 succeed
#
# out_dir: train/checkpoint_Triplet_3/ablation512/<arm>/seed_<S>/Model_run
PARTITIONS="${PARTITIONS:-amp48,ada24}"
EXCLUDE="${EXCLUDE:-colossus}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-24:00:00}"
SMOKE="${SMOKE:-0}"
NO_EVAL="${NO_EVAL:-0}"
DEPEND="${DEPEND:-}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoint3/Model_20260119_062652/model_best.pth}"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

RUNS=("$@")
if [ "$#" -eq 0 ]; then
    # seeds 86/123/1024 of every control arm first (an early 3-seed read), then the other three,
    # then the two reproducibility reruns of the entropy arm
    for S in 86 123 1024 42 456 789; do
        for A in asl nogate randgate; do RUNS+=("$A:$S"); done
    done
    RUNS+=(gate:86 gate:1024)
fi

COMMON=(--accum-steps 6 --batch-size 8 --monitor mAP --epochs 8 --patience 3
        --lr 1e-4 --memory-size 2048 --metric-embed query --head-classes 5
        --class-weight-order shard --stage1-ckpt "$STAGE1_CKPT")
SMOKE_ARGS=()
[ "$SMOKE" = 1 ] && SMOKE_ARGS=(--epochs 1 --patience 1 --limit-train-batches 40 --limit-val-batches 20 --log-every 20)

echo "runs: ${RUNS[*]} | smoke=$SMOKE | partitions=$PARTITIONS exclude=$EXCLUDE"
for spec in "${RUNS[@]}"; do
    IFS=: read -r arm seed <<< "$spec"
    case "$arm" in
        asl)      armargs=(--triplet-lambda 0) ;;
        nogate)   armargs=(--triplet-lambda 0.1 --anchor-rule all) ;;
        randgate) armargs=(--triplet-lambda 0.1 --anchor-rule random) ;;
        gate)     armargs=(--triplet-lambda 0.1 --anchor-rule mean) ;;
        *) echo "unknown arm: $arm (asl|nogate|randgate|gate)" >&2; exit 1 ;;
    esac
    tag=""; [ "$SMOKE" = 1 ] && tag="_smoke"
    rel="checkpoint_Triplet_3/ablation512${tag}/${arm}/seed_${seed}/Model_run"
    name="isbi2026_abl_${arm}_s${seed}${tag}"
    jid=$(sbatch --parsable -J "$name" ${DEPEND:+--dependency=afterok:$DEPEND} \
        -p "$PARTITIONS" ${EXCLUDE:+-x "$EXCLUDE"} -c "$CPUS" --mem="$MEM" --time="$TIME" \
        "$SCRIPT_DIR/train_2_v3.sh" \
            --seed "$seed" --out-dir "$rel" \
            "${COMMON[@]}" "${armargs[@]}" "${SMOKE_ARGS[@]}")
    echo "$spec -> job $jid | /home/psytp7/logs/${name}_${jid}.out"
    if [ "$SMOKE" != 1 ] && [ "$NO_EVAL" != 1 ]; then
        ejid=$(sbatch --parsable -J "isbi2026_evabl_${arm}_s${seed}" --dependency=afterok:"$jid" \
            ${EXCLUDE:+-x "$EXCLUDE"} \
            "$SCRIPT_DIR/eval_perclass.sh" "$REPO/train/$rel/model_best.pth" 512 "${arm}_s${seed}")
        echo "    eval -> job $ejid (afterok:$jid)"
    fi
done
