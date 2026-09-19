#!/bin/bash
# Track B Wave-1: PURE train_2_v2.py hyperparameter sweep - NO code changes to
# the method, only CLI knobs. Isolates "how far does careful tuning of the
# published recipe get us" from Track A's algorithmic changes.
#
#   ./submit_stage2_pure_sweep.sh                 # default Wave-1 grid (8 jobs)
#   LRS="3e-5 1e-4" SEEDS="86" ./submit_stage2_pure_sweep.sh
#   EPOCHS=1 ./submit_stage2_pure_sweep.sh        # smoke test
#
# Fixed: effective batch 48 (accum 6 x batch 8), monitor mAP, converged Stage-1
# checkpoint (val mAP 0.385, paper Table 1). Swept: --lr x --seed.
# 8 epochs / patience 3 so a multi-epoch cosine actually has room to help
# (3-epoch runs already beat the 1-epoch tuned recipe: 0.412 vs 0.404).
#
# Wave 2 (run separately once these land): at the best --lr, sweep
#   --memory-size {256,1024,4096} x --margin {0.3,0.5}  (seed 86)
# then a 5-seed confirmation at the single best config.
#
# Cluster: QOS gpu=4 per partition; amp48,ada24 -> all 8 start. ~2 h/epoch,
# --time 30h covers 8 epochs + eval.
PARTITIONS="${PARTITIONS:-amp48,ada24}"
EXCLUDE="${EXCLUDE:-}"   # e.g. EXCLUDE=colossus to avoid a flaky node
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-30:00:00}"
EPOCHS="${EPOCHS:-8}"
PATIENCE="${PATIENCE:-3}"
LRS="${LRS:-3e-5 5e-5 8e-5 1.2e-4}"
SEEDS="${SEEDS:-86 123}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoint3/Model_20260119_062652/model_best.pth}"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "epochs=$EPOCHS patience=$PATIENCE | lrs: $LRS | seeds: $SEEDS"
echo "stage1_ckpt=$STAGE1_CKPT | partitions=$PARTITIONS"
echo

for LR in $LRS; do
    for S in $SEEDS; do
        jid=$(sbatch --parsable \
            -J "isbi2026_v2pure_lr${LR}_s${S}" \
            -p "$PARTITIONS" ${EXCLUDE:+-x "$EXCLUDE"} -c "$CPUS" --mem="$MEM" --time="$TIME" \
            "$SCRIPT_DIR/train_2_v2.sh" \
                --accum-steps 6 --batch-size 8 --seed "$S" \
                --lr "$LR" --epochs "$EPOCHS" --patience "$PATIENCE" \
                --monitor mAP --stage1-ckpt "$STAGE1_CKPT")
        echo "lr=$LR seed=$S -> job $jid | /home/psytp7/logs/isbi2026_v2pure_lr${LR}_s${S}_${jid}.out"
    done
done

echo
echo "watch:  squeue --me"
echo "curves: grep -H 'Epoch .*mAP=' /home/psytp7/logs/isbi2026_v2pure_*.out"
