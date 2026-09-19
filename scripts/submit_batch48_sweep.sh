#!/bin/bash
# Stage-2 sweep at FIXED effective batch = 48, from the original ISBI submission
# Stage-1 checkpoint (checkpoint3/Model_20260119_062652, val mAP 0.385 - now the
# default in train_2_v2.py after the stage1_full30 regression was found).
#
# Grid: accum_steps in {6, 8} x seeds {86, 123, 456, 789, 1024} = 10 jobs.
#   accum 6 -> --batch-size 8  (= 48, the paper recipe)
#   accum 8 -> --batch-size 6  (= 48)
# so the only thing that varies within an accum pair is accumulation granularity,
# not the effective batch. 3 epochs, patience 2 (monitor mAP): enough to see
# whether epoch 2-3 ever beats epoch 1 from the stronger checkpoint.
#
#   ./submit_batch48_sweep.sh                       # default grid
#   SEEDS="7 13 21" ./submit_batch48_sweep.sh       # override seeds
#   ACCUM_LEGS="6:8" EPOCHS=1 ./submit_batch48_sweep.sh   # paper-exact single leg
#
# Wall time: ~2 h/epoch observed (triplet mining + XBM cost once the bank fills),
# so ~6 h for 3 epochs. train_2_v2.sh hard-codes #SBATCH --time=4:00:00, which is
# too short - overridden to 10 h here. All partitions are MaxTime=UNLIMITED.
# QOS caps: amp48 gpu=4, ada24 gpu=4 -> 8 of 10 run, 2 queue behind them.
PARTITIONS="${PARTITIONS:-amp48,ada24}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-10:00:00}"
EPOCHS="${EPOCHS:-3}"
PATIENCE="${PATIENCE:-2}"
SEEDS="${SEEDS:-86 123 456 789 1024}"
ACCUM_LEGS="${ACCUM_LEGS:-6:8 8:6}"   # each entry is accum:batch_size (product = 48)
STAGE1_CKPT="${STAGE1_CKPT:-checkpoint3/Model_20260119_062652/model_best.pth}"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "stage1_ckpt = $STAGE1_CKPT"
echo "epochs=$EPOCHS patience=$PATIENCE time=$TIME | legs: $ACCUM_LEGS | seeds: $SEEDS"
echo

for leg in $ACCUM_LEGS; do
    ACCUM="${leg%%:*}"; BS="${leg##*:}"
    for S in $SEEDS; do
        jid=$(sbatch --parsable \
            -J "isbi2026_train2v2_eb48_accum${ACCUM}_seed${S}" \
            -p "$PARTITIONS" -c "$CPUS" --mem="$MEM" --time="$TIME" \
            "$SCRIPT_DIR/train_2_v2.sh" \
                --accum-steps "$ACCUM" --batch-size "$BS" --seed "$S" \
                --epochs "$EPOCHS" --patience "$PATIENCE" \
                --stage1-ckpt "$STAGE1_CKPT")
        echo "accum=$ACCUM bs=$BS (eff $((ACCUM*BS))) seed=$S -> job $jid | /home/psytp7/logs/isbi2026_train2v2_eb48_accum${ACCUM}_seed${S}_${jid}.out"
    done
done
