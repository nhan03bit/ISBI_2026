#!/bin/bash
# Tier 1a: resolution sweep for train_2_v3.py.
#
# Entropy.pdf slide 9 is the strongest doc signal for a further gain: going
# 384 -> 512 px TRIPLED the Stage-2 mAP improvement (+0.007 -> +0.020). Every v3
# run so far is at 512. This sweeps 640 and 768 on the corrected `fix_only`
# recipe. Shards are native-resolution PadChest originals -> no data rebuild.
#
#   ./submit_stage2_v3_res_sweep.sh                 # img 640,768 x seed 86  (2 jobs)
#   SEEDS="86 123 1024" ./submit_stage2_v3_res_sweep.sh
#   SIZES="576 640" ./submit_stage2_v3_res_sweep.sh
#   SMOKE=1 ./submit_stage2_v3_res_sweep.sh
#
# Effective batch stays ~48: batch 4 x accum 12. --swa-last-k 3 bundled in
# (Tier 1c). Yardstick = evaluate/evaluate.py on each model_best.pth afterwards.
PARTITIONS="${PARTITIONS:-amp48,ada24}"
EXCLUDE="${EXCLUDE:-colossus}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
TIME="${TIME:-24:00:00}"
SMOKE="${SMOKE:-0}"
SIZES="${SIZES:-640 768}"
SEEDS="${SEEDS:-86}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoint3/Model_20260119_062652/model_best.pth}"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMMON=(--accum-steps 12 --batch-size 4 --monitor mAP --epochs 8 --patience 3
        --lr 1e-4 --memory-size 2048 --metric-embed query --head-classes 5
        --triplet-lambda 0.1 --class-weight-order shard --swa-last-k 3
        --stage1-ckpt "$STAGE1_CKPT")
if [ "$SMOKE" = 1 ]; then
    COMMON+=(--epochs 2 --patience 2 --limit-train-batches 40 --limit-val-batches 20 --log-every 10)
fi

echo "sizes: $SIZES | seeds: $SEEDS | smoke=$SMOKE | partitions=$PARTITIONS exclude=$EXCLUDE mem=$MEM"
echo

for S in $SIZES; do
  for SD in $SEEDS; do
    jid=$(sbatch --parsable \
        -J "isbi2026_v3res_img${S}_s${SD}" \
        -p "$PARTITIONS" ${EXCLUDE:+-x "$EXCLUDE"} -c "$CPUS" --mem="$MEM" --time="$TIME" \
        "$SCRIPT_DIR/train_2_v3.sh" \
            --seed "$SD" --img-size "$S" \
            --out-dir "checkpoint_Triplet_3/res_sweep/img${S}_seed${SD}/Model_run" \
            "${COMMON[@]}")
    echo "img=$S seed=$SD -> job $jid | /home/psytp7/logs/isbi2026_v3res_img${S}_s${SD}_${jid}.out"
  done
done

echo
echo "watch:  scripts/mon.py 'isbi2026_v3res_*'"
echo "eval :  for d in train/checkpoint_Triplet_3/res_sweep/*/Model_run; do sbatch -x colossus scripts/eval_perclass.sh \$d/model_best.pth 512 \$(basename \$(dirname \$d)); done"
echo "        (also eval model_swa.pth, and try evaluate_tta.py --scales 512,640,768 across the res models)"
