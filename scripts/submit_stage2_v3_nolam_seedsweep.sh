#!/bin/bash
# No-entropy CONTROL seed sweep for train_2_v3.py.
#
# Identical to submit_stage2_v3_seedsweep.sh (the `fix_only` entropy-gated arm) except
# --triplet-lambda 0: the triplet term (and therefore the entropy-selected anchors) has
# zero weight, so Stage 2 is ASL-only fine-tuning from the same Stage-1 checkpoint with
# the same lr / schedule / EMA / class weights / seeds. Pair each seed with
# checkpoint_Triplet_3/nolam_seedsweep/seed_<S> to measure the entropy-triplet effect.
# NOTE: this removes the whole triplet term, not just the entropy gate; a gate-only
# ablation (random anchors) needs an anchor-rule flag that v3 does not have.
#
#   ./submit_stage2_v3_nolam_seedsweep.sh                # seeds 42 86 123 456 789 1024
#   SMOKE=1 ./submit_stage2_v3_nolam_seedsweep.sh 42
#
# out_dir: train/checkpoint_Triplet_3/nolam_seedsweep/seed_<S>/Model_run
PARTITIONS="${PARTITIONS:-amp48,ada24}"
# colossus silently fails torch CUDA init -> CPU fallback; excluded by default.
EXCLUDE="${EXCLUDE:-colossus}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-16:00:00}"
SMOKE="${SMOKE:-0}"
AUGVARY="${AUGVARY:-0}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoint3/Model_20260119_062652/model_best.pth}"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEEDS=("$@")
[ "$#" -eq 0 ] && SEEDS=(42 86 123 456 789 1024)

COMMON=(--accum-steps 6 --batch-size 8 --monitor mAP --epochs 8 --patience 3
        --lr 1e-4 --memory-size 2048 --metric-embed query --head-classes 5
        --triplet-lambda 0 --class-weight-order shard
        --stage1-ckpt "$STAGE1_CKPT")
if [ "$SMOKE" = 1 ]; then
    COMMON+=(--epochs 1 --patience 1 --limit-train-batches 40 --limit-val-batches 20 --log-every 20)
fi

echo "seeds: ${SEEDS[*]} | augvary=$AUGVARY smoke=$SMOKE | partitions=$PARTITIONS exclude=$EXCLUDE"
echo

for S in "${SEEDS[@]}"; do
    aug=()
    [ "$AUGVARY" = 1 ] && aug=(--aug-base-seed "$S")
    jid=$(sbatch --parsable \
        -J "isbi2026_v3nolam_s${S}" \
        -p "$PARTITIONS" ${EXCLUDE:+-x "$EXCLUDE"} -c "$CPUS" --mem="$MEM" --time="$TIME" \
        "$SCRIPT_DIR/train_2_v3.sh" \
            --seed "$S" \
            --out-dir "checkpoint_Triplet_3/nolam_seedsweep/seed_${S}/Model_run" \
            "${COMMON[@]}" "${aug[@]}")
    echo "seed $S -> job $jid | /home/psytp7/logs/isbi2026_v3nolam_s${S}_${jid}.out"
done

echo
echo "watch:  scripts/mon.py 'isbi2026_v3nolam_*'    (or scripts/mon.py for the full v3 board)"
echo "curves: scripts/mon.py 'isbi2026_v3nolam_*' --epochs"
