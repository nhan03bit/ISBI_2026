#!/bin/bash
# Resolution sweep of the ASL-only vs entropy-gate contrast (2026-10-12).
# Plan / pre-registration: docs/plan/2026-10-12-res-sweep-plan.md
#
# Same "push" recipe as submit_stage2_v3_push.sh (4 epochs of an 8-epoch cosine, drop-path 0.1,
# SWA over the last 3 epochs, effective batch 48). The two arms differ in one flag:
#   asl   --triplet-lambda 0            ASL only (bank and mining run with zero gradient)
#   gate  --anchor-rule mean            entropy-gated triplet + bank (the method)
# Run spec = arm:res:seed. Each training job is followed (afterok) by eval_perclass.sh on
# model_best.pth at the same resolution -> eval_result.<arm>_<res>_s<seed>.json + preds npz.
#
#   SMOKE=1 ./submit_stage2_v3_res_ablation.sh asl:1024:42 gate:1024:42     # OOM / code-path check
#   ./submit_stage2_v3_res_ablation.sh                                      # all 24 runs
#   PARTITIONS=amp48 BS=2 ACCUM=24 ./submit_stage2_v3_res_ablation.sh asl:1024:42   # if 1024 OOMs
#
# out_dir: train/checkpoint_Triplet_3/resablation/<arm>/img<res>_seed<seed>/Model_run
PARTITIONS="${PARTITIONS:-amp48,ada24}"
EXCLUDE="${EXCLUDE:-colossus}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
TIME="${TIME:-36:00:00}"
SMOKE="${SMOKE:-0}"
NO_EVAL="${NO_EVAL:-0}"
BS="${BS:-4}"
ACCUM="${ACCUM:-12}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoint3/Model_20260119_062652/model_best.pth}"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

RUNS=("$@")
if [ "$#" -eq 0 ]; then
    for S in 42 123 456; do
        for R in 1024 768 896 640; do
            for A in asl gate; do RUNS+=("$A:$R:$S"); done
        done
    done
fi

COMMON=(--accum-steps "$ACCUM" --batch-size "$BS" --monitor mAP --memory-size 2048
        --metric-embed query --head-classes 5 --class-weight-order shard --swa-last-k 3
        --lr 1e-4 --epochs 4 --patience 2 --sched-epochs 8 --drop-path-rate 0.1
        --stage1-ckpt "$STAGE1_CKPT")
SMOKE_ARGS=()
[ "$SMOKE" = 1 ] && SMOKE_ARGS=(--epochs 1 --patience 1 --limit-train-batches 40 --limit-val-batches 20 --log-every 10)

echo "runs: ${RUNS[*]} | bs=$BS accum=$ACCUM | smoke=$SMOKE | partitions=$PARTITIONS exclude=$EXCLUDE"
for spec in "${RUNS[@]}"; do
    IFS=: read -r arm res seed <<< "$spec"
    case "$arm" in
        asl)  armargs=(--triplet-lambda 0) ;;
        gate) armargs=(--triplet-lambda 0.1 --anchor-rule mean) ;;
        *) echo "unknown arm: $arm (asl|gate)" >&2; exit 1 ;;
    esac
    tag=""; [ "$SMOKE" = 1 ] && tag="_smoke"
    rel="checkpoint_Triplet_3/resablation${tag}/${arm}/img${res}_seed${seed}/Model_run"
    name="isbi2026_resabl_${arm}_${res}_s${seed}${tag}"
    jid=$(sbatch --parsable -J "$name" -p "$PARTITIONS" ${EXCLUDE:+-x "$EXCLUDE"} -c "$CPUS" \
        --mem="$MEM" --time="$TIME" "$SCRIPT_DIR/train_2_v3.sh" \
        --seed "$seed" --img-size "$res" --out-dir "$rel" "${COMMON[@]}" "${armargs[@]}" "${SMOKE_ARGS[@]}")
    echo "$spec -> job $jid | /home/psytp7/logs/${name}_${jid}.out"
    if [ "$SMOKE" != 1 ] && [ "$NO_EVAL" != 1 ]; then
        ejid=$(sbatch --parsable -J "isbi2026_evresabl_${arm}_${res}_s${seed}" --dependency=afterok:"$jid" \
            --time=4:00:00 -x "${EXCLUDE:+$EXCLUDE,}tamar" \
            "$SCRIPT_DIR/eval_perclass.sh" "$REPO/train/$rel/model_best.pth" "$res" "${arm}_${res}_s${seed}")
        echo "    eval -> job $ejid (afterok:$jid)"
    fi
done
