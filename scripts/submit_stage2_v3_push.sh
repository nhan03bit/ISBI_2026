#!/bin/bash
# Push wave: internal val mAP 0.456 (img768_seed86) -> 0.49 target.
# Plan: /home/psytp7/.claude/plans/based-on-previous-stage-functional-hoare.md
#
# Evidence from res_sweep/seedsweep/sweep2: resolution is the strongest lever
# (512/640/768 -> 0.4434/0.4494/0.4560), every run peaks at epoch 2-3 then
# overfits, drop-path 0.1 was the only sweep2 knob that helped (+0.005 @512),
# and a shorter cosine horizon (sched3) hurt. So: keep the 8-epoch cosine
# horizon (--sched-epochs 8) but stop at epoch 4, add drop-path 0.1, go to
# 896/1024 px, and add seeds for the ensemble.
#
# Runs (RUNS selects a subset, default all):
#   cont1024      continue fine-tuning img768_seed86 EMA weights at 1024 px, low LR
#   img1024_dp01  Stage-1 init, 1024 px, drop-path 0.1, seed 86
#   img896_dp01   Stage-1 init,  896 px, drop-path 0.1, seed 86
#   img768_dp01   Stage-1 init,  768 px, drop-path 0.1, seeds 1024 42
#
#   SMOKE=1 RUNS="img1024_dp01" ./submit_stage2_v3_push.sh   # OOM check first
#   ./submit_stage2_v3_push.sh
#   BS=2 ACCUM=24 RUNS="img1024_dp01 cont1024" ./submit_stage2_v3_push.sh  # if 1024 OOMs at bs 4
PARTITIONS="${PARTITIONS:-amp48,ada24}"
EXCLUDE="${EXCLUDE:-colossus}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
TIME="${TIME:-36:00:00}"
SMOKE="${SMOKE:-0}"
RUNS="${RUNS:-cont1024 img1024_dp01 img896_dp01 img768_dp01}"
BS="${BS:-4}"
ACCUM="${ACCUM:-12}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoint3/Model_20260119_062652/model_best.pth}"
CONT_CKPT="${CONT_CKPT:-checkpoint_Triplet_3/res_sweep/img768_seed86/Model_run/model_best.pth}"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMMON=(--accum-steps "$ACCUM" --batch-size "$BS" --monitor mAP
        --memory-size 2048 --metric-embed query --head-classes 5
        --triplet-lambda 0.1 --class-weight-order shard --swa-last-k 3)
SCRATCH=(--lr 1e-4 --epochs 4 --patience 2 --sched-epochs 8 --drop-path-rate 0.1
         --stage1-ckpt "$STAGE1_CKPT")
SMOKE_ARGS=()
if [ "$SMOKE" = 1 ]; then
    SMOKE_ARGS=(--epochs 1 --patience 1 --limit-train-batches 40 --limit-val-batches 20 --log-every 10)
fi

submit() {  # name seed out-subdir train-args...
    local name=$1 sd=$2 sub=$3; shift 3
    local tag=""; [ "$SMOKE" = 1 ] && tag="_smoke"
    local jid
    jid=$(sbatch --parsable \
        -J "isbi2026_v3push_${name}_s${sd}${tag}" \
        -p "$PARTITIONS" ${EXCLUDE:+-x "$EXCLUDE"} -c "$CPUS" --mem="$MEM" --time="$TIME" \
        "$SCRIPT_DIR/train_2_v3.sh" \
            --seed "$sd" \
            --out-dir "checkpoint_Triplet_3/push/${sub}${tag}/Model_run" \
            "${COMMON[@]}" "$@" "${SMOKE_ARGS[@]}")
    echo "$name seed=$sd -> job $jid | /home/psytp7/logs/isbi2026_v3push_${name}_s${sd}${tag}_${jid}.out"
}

echo "runs: $RUNS | bs=$BS accum=$ACCUM | smoke=$SMOKE | partitions=$PARTITIONS exclude=$EXCLUDE"
echo

for R in $RUNS; do
  case "$R" in
    cont1024)
      # FixRes-style continuation: same arch, no drop-path (matches the source
      # weights), low LR, short horizon. create_model loads with strict=False -
      # check the log line "Stage-1 init: ... (missing=0 unexpected=0)".
      submit cont1024 86 cont1024_from768_s86 --img-size 1024 \
          --lr 2e-5 --epochs 2 --patience 1 --sched-epochs 2 --warmup-frac 0.1 \
          --stage1-ckpt "$CONT_CKPT" ;;
    img1024_dp01) submit img1024_dp01 86 img1024_dp01_seed86 --img-size 1024 "${SCRATCH[@]}" ;;
    img896_dp01)  submit img896_dp01  86 img896_dp01_seed86  --img-size 896  "${SCRATCH[@]}" ;;
    img768_dp01)
      for SD in 1024 42; do
        submit img768_dp01 "$SD" "img768_dp01_seed${SD}" --img-size 768 "${SCRATCH[@]}"
      done ;;
    *) echo "unknown run: $R" >&2; exit 1 ;;
  esac
done

echo
echo "watch:  grep 'Epoch .*mAP=' /home/psytp7/logs/isbi2026_v3push_*.out"
echo "eval :  sbatch -x colossus scripts/evaluate_tta.sh --dump-probs ../analysis/out/probs \\"
echo "          --out-json ../analysis/out/tta_push_ens.json <ckpt>@<its img size> ..."
echo "        python analysis/ens_select.py analysis/out/probs --out analysis/out/ens_select.json"
