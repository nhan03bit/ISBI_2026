#!/bin/bash
# Wave 6 (regularization) + img1024 seeds, after the 2026-09-25 push wave.
# Report: docs/result/2026-09-26-push-wave.md
#
# Evidence from the push wave: 768/896/1024 px all land at 0.455-0.462 (resolution
# has plateaued), and every run overfits after epoch 3 (train loss keeps falling,
# val loss rises). So the next single-model lever is regularization. The sweep runs
# at 768 px (2-3x faster than 1024, same mAP) on the push recipe; each arm is
# compared against its matched-seed baseline, push/img768_dp01_seed42 (0.4585,
# SWA 0.4600). Winning arms get carried to 1024 px.
#
# Runs (RUNS selects a subset, default all):
#   dp02        drop-path 0.2 (push used 0.1)
#   asl_g4      ASL gamma_neg 4 / gamma_pos 1 (default 2 / 0)
#   rrc07       RandomResizedCrop scale 0.7-1.0 (default 0.9-1.0)
#   wd05        weight decay 5e-2 (default 1e-2)
#   img1024     the 48997 recipe unchanged, seeds 42 and 1024 (ensemble members)
#
#   SMOKE=1 ./submit_stage2_v3_reg.sh      # 40-batch check of every new flag first
#   ./submit_stage2_v3_reg.sh
#   RUNS="img1024" ./submit_stage2_v3_reg.sh
PARTITIONS="${PARTITIONS:-amp48,ada24}"
PARTITIONS_1024="${PARTITIONS_1024:-amp48}"   # bs4 @1024 only verified on the 48GB A6000s
EXCLUDE="${EXCLUDE:-colossus}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
TIME="${TIME:-36:00:00}"
SMOKE="${SMOKE:-0}"
RUNS="${RUNS:-dp02 asl_g4 rrc07 wd05 img1024}"
SEED="${SEED:-42}"
BS="${BS:-4}"
ACCUM="${ACCUM:-12}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoint3/Model_20260119_062652/model_best.pth}"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMMON=(--accum-steps "$ACCUM" --batch-size "$BS" --monitor mAP
        --memory-size 2048 --metric-embed query --head-classes 5
        --triplet-lambda 0.1 --class-weight-order shard --swa-last-k 3
        --lr 1e-4 --epochs 4 --patience 2 --sched-epochs 8
        --stage1-ckpt "$STAGE1_CKPT")
SMOKE_ARGS=()
if [ "$SMOKE" = 1 ]; then
    SMOKE_ARGS=(--epochs 1 --patience 1 --limit-train-batches 40 --limit-val-batches 20 --log-every 10)
fi

submit() {  # name seed partitions out-subdir train-args...
    local name=$1 sd=$2 parts=$3 sub=$4; shift 4
    local tag=""; [ "$SMOKE" = 1 ] && tag="_smoke"
    local jid
    jid=$(sbatch --parsable \
        -J "isbi2026_v3reg_${name}_s${sd}${tag}" \
        -p "$parts" ${EXCLUDE:+-x "$EXCLUDE"} -c "$CPUS" --mem="$MEM" --time="$TIME" \
        "$SCRIPT_DIR/train_2_v3.sh" \
            --seed "$sd" \
            --out-dir "checkpoint_Triplet_3/reg/${sub}${tag}/Model_run" \
            "${COMMON[@]}" "$@" "${SMOKE_ARGS[@]}")
    echo "$name seed=$sd -> job $jid | /home/psytp7/logs/isbi2026_v3reg_${name}_s${sd}${tag}_${jid}.out"
}

echo "runs: $RUNS | seed=$SEED bs=$BS accum=$ACCUM | smoke=$SMOKE | partitions=$PARTITIONS exclude=$EXCLUDE"
echo

for R in $RUNS; do
  case "$R" in
    dp02)   submit dp02   "$SEED" "$PARTITIONS" "img768_dp02_seed$SEED"   --img-size 768 --drop-path-rate 0.2 ;;
    asl_g4) submit asl_g4 "$SEED" "$PARTITIONS" "img768_asl_g4_seed$SEED" --img-size 768 --drop-path-rate 0.1 \
                --gamma-neg 4 --gamma-pos 1 ;;
    rrc07)  submit rrc07  "$SEED" "$PARTITIONS" "img768_rrc07_seed$SEED"  --img-size 768 --drop-path-rate 0.1 \
                --rrc-scale-min 0.7 ;;
    wd05)   submit wd05   "$SEED" "$PARTITIONS" "img768_wd05_seed$SEED"   --img-size 768 --drop-path-rate 0.1 \
                --weight-decay 5e-2 ;;
    img1024)
      for SD in 42 1024; do
        submit img1024_dp01 "$SD" "$PARTITIONS_1024" "img1024_dp01_seed$SD" --img-size 1024 --drop-path-rate 0.1
      done ;;
    *) echo "unknown run: $R" >&2; exit 1 ;;
  esac
done

echo
echo "watch:  python3 scripts/mon.py 'isbi2026_v3reg_*'"
