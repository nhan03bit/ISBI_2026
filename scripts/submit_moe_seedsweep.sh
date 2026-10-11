#!/bin/bash
# Seed sweep of the reconstructed original MoE Stage 2 (train_moe.py, b12 recipe:
# 4 experts top-2, aux 5e-3, lr 4e-5, 384 px, 2 epochs, stages 1-3 frozen) from the
# Stage-1 checkpoint behind Table 1. Gives the MoE row a seed distribution to set
# against the 5-seed entropy row and the v3 seedsweeps.
#
#   ./submit_moe_seedsweep.sh                  # seeds 42 86 123 456 789 1024
#   ./submit_moe_seedsweep.sh 42 86            # explicit seeds
#   SMOKE=1 ./submit_moe_seedsweep.sh 42       # 40 train / 20 val batches
#   IMG=512 TAG=moe512 ./submit_moe_seedsweep.sh   # resolution-matched arm
#
# out_dir: checkpoint/<TAG>/seed_<S>/Model_run  (TAG default moe_recon)
PARTITIONS="${PARTITIONS:-amp48,ada24}"
EXCLUDE="${EXCLUDE:-colossus}"     # colossus silently falls back to CPU
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-16:00:00}"
SMOKE="${SMOKE:-0}"
IMG="${IMG:-384}"
TAG="${TAG:-moe_recon}"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS=("$@")
[ "$#" -eq 0 ] && SEEDS=(42 86 123 456 789 1024)

COMMON=(--img-size "$IMG")
if [ "$SMOKE" = 1 ]; then
    COMMON+=(--epochs 1 --limit-train-batches 40 --limit-val-batches 20)
    TAG="${TAG}_smoke"
fi

echo "seeds: ${SEEDS[*]} | img=$IMG tag=$TAG smoke=$SMOKE | partitions=$PARTITIONS exclude=$EXCLUDE"
for S in "${SEEDS[@]}"; do
    jid=$(sbatch --parsable \
        -J "isbi2026_${TAG}_s${S}" \
        -p "$PARTITIONS" ${EXCLUDE:+-x "$EXCLUDE"} -c "$CPUS" --mem="$MEM" --time="$TIME" \
        "$SCRIPT_DIR/train_moe.sh" \
            --seed "$S" --out-dir "checkpoint/${TAG}/seed_${S}/Model_run" "${COMMON[@]}")
    echo "seed $S -> job $jid | /home/psytp7/logs/isbi2026_${TAG}_s${S}_${jid}.out"
done
echo
echo "watch: scripts/mon.py 'isbi2026_${TAG}_*'"
