#!/bin/bash
# "Measure impact first" probe for the two Method corrections, isolated:
#   class-weight order:  csv (published bug)  vs  shard (fixed)
#   predictive entropy:  softmax (paper Eq.1) vs  binary (multi-label)
# 2 x 2 = 4 single-epoch jobs, seed 86, converged Stage-1. Compare macro + tail
# mAP across the four to decide whether a full 5-seed x 2-lineage re-run is worth
# it (submit_stage2_fixed.sh).
#
#   ./submit_classweight_probe.sh [--dry-run]     SEED=86 by default

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

PARTITIONS="${PARTITIONS:-amp48,ada24}"
SEED="${SEED:-86}"
CKPT="checkpoint3/Model_20260119_062652/model_best.pth"

RECIPE=(--lr 3e-5 --embedding-dim 768 --weight-decay 1e-2 --memory-size 96
        --epochs 1 --patience 1 --min-delta 0.0 --gamma-neg 2.0 --gamma-pos 0.0
        --monitor mAP --batch-size 8 --accum-steps 6 --num-workers 8
        --val-num-workers 2 --log-every 500 --margin 0.5 --disc 0
        --anchor-rule mean --pos-min-shared 1 --mining easy --lambda-tri 0.1)

for cw in csv shard; do
    for ent in softmax binary; do
        out="checkpoint_Triplet_2/probe/cw_${cw}__ent_${ent}/seed_${SEED}/Model_run"
        name="isbi2026_s2probe_cw${cw}_ent${ent}_seed${SEED}"
        if [ "$DRY" = 1 ]; then echo "would submit $name | out=$out"; continue; fi
        jid=$(sbatch --parsable -J "$name" -p "$PARTITIONS" -c 8 --mem=64G --time=12:00:00 \
            "$SCRIPT_DIR/train_2_v2.sh" \
                --stage1-ckpt "$CKPT" --num-classes 30 --seed "$SEED" \
                --class-weight-order "$cw" --entropy-mode "$ent" \
                --out-dir "$out" "${RECIPE[@]}")
        echo "$name -> job $jid"
    done
done

echo
echo "cw_csv + ent_softmax  == the published Table-1 config (should reproduce ~0.414 macro)"
echo "cw_shard + ent_binary == the fixed config"
