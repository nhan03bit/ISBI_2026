#!/bin/bash
# B0 re-run: the paper's Stage-2 recipe with the two corrections applied --
#   --entropy-mode binary        (per-label multi-label predictive entropy, not softmax)
#   --class-weight-order shard    (ASL class weights aligned to the logit order;
#                                  fixes the published index-scramble bug)
# 5 seeds x {converged, under-trained} Stage-1 = 10 jobs. Matches the paper's
# Experimental Setup otherwise (lr 3e-5, 1 epoch, patience 1, gamma_neg 2.0,
# effective batch 48 = bs 8 x accum 6, memory 96, margin 0.5).
#
#   ./submit_stage2_fixed.sh            # all 10
#   ./submit_stage2_fixed.sh --dry-run
#   SEEDS="86 123" LINEAGES="converged" ./submit_stage2_fixed.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

PARTITIONS="${PARTITIONS:-amp48,ada24}"
CPUS="${CPUS:-8}"; MEM="${MEM:-64G}"; TIME="${TIME:-12:00:00}"
SEEDS="${SEEDS:-86 123 456 789 1024}"
LINEAGES="${LINEAGES:-converged undertrained}"

CKPT_converged="checkpoint3/Model_20260119_062652/model_best.pth"
CKPT_undertrained="checkpoint3/stage1_full30/model_best.pth"

RECIPE=(--lr 3e-5 --embedding-dim 768 --weight-decay 1e-2 --memory-size 96
        --epochs 1 --patience 1 --min-delta 0.0 --gamma-neg 2.0 --gamma-pos 0.0
        --monitor mAP --batch-size 8 --accum-steps 6 --num-workers 8
        --val-num-workers 2 --log-every 500 --margin 0.5 --disc 0
        --entropy-mode binary --anchor-rule mean --pos-min-shared 1 --mining easy
        --lambda-tri 0.1 --class-weight-order shard)

for lin in $LINEAGES; do
    eval "ckpt=\$CKPT_${lin}"
    for S in $SEEDS; do
        out="checkpoint_Triplet_2/fixed_${lin}/seed_${S}/Model_run"
        name="isbi2026_s2fixed_${lin}_seed${S}"
        if [ "$DRY" = 1 ]; then
            echo "would submit $name | ckpt=$ckpt | out=$out"
            continue
        fi
        jid=$(sbatch --parsable -J "$name" -p "$PARTITIONS" -c "$CPUS" --mem="$MEM" --time="$TIME" \
            "$SCRIPT_DIR/train_2_v2.sh" \
                --stage1-ckpt "$ckpt" --num-classes 30 --seed "$S" \
                --out-dir "$out" "${RECIPE[@]}")
        echo "$name -> job $jid | /home/psytp7/logs/${name}_${jid}.out | out_dir train/$out"
    done
done

echo
echo "then per-class eval each:  sbatch scripts/eval_perclass.sh train/checkpoint_Triplet_2/fixed_<lin>/seed_<S>/Model_run/model_best.pth 512 res512"
