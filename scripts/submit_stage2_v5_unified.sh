#!/bin/bash
# Unified benchmark (2026-10-12): triplet bank / entropy gate / Markov prior in ONE setting.
# Plan / pre-registration: docs/plan/2026-10-12-unified-benchmark-plan.md
#
# Everything on train/train_2_v5.py, the push recipe of submit_stage2_v5_prior.sh at 768 px
# (drop-path 0.1, 4 of 8 cosine epochs, SWA of the last 3 EMA epochs, effective batch 48), seeds
# 42 / 123 / 456 / 789 / 1024. Arms (run spec = arm:seed):
#   a  ASL only                           --no-triplet
#   b  triplet + bank, every image anchor --triplet-lambda 0.1 --anchor-rule all
#   c  entropy gate                       --triplet-lambda 0.1 --anchor-rule mean
#   d  entropy gate + Markov prior        c + PRIOR flags
#   e  ASL + Markov prior                 --no-triplet + PRIOR flags
#   f  stronger triplet (exploratory)     --triplet-lambda 1.0 --anchor-rule mean --mining hard
#   m  sparse mixture of experts          --arch moe --no-triplet   (state-of-the-art reference, 4 experts top-2)
# a (s42, s1024), d (s42, s1024) and m (s42, s1024) already exist (checkpoint_MoE/{asl,moe}_img768_*, prior/img768_qprior_*;
# their dumps are read from analysis/out/probs_moe_flip by analysis/unified_benchmark.py).
# Each training job is followed (afterok) by flip-TTA probability dumps of model_swa.pth at 768 px into
# analysis/out/probs_unified_flip (arms d and e twice: --prior-mode with and zero).
#
#   SMOKE=1 ./submit_stage2_v5_unified.sh b:42 d:42 e:42 f:42     # 40-batch check first
#   ./submit_stage2_v5_unified.sh                                 # all 26 new runs, seeds 42/123/456 first
#   NO_EVAL=1 ...  /  DUMP_ONLY="a:42 a:1024 d:42 d:1024" ./submit_stage2_v5_unified.sh   # re-dump existing runs
# out_dir: train/checkpoint_Triplet_3/unified768/<arm>/seed_<S>/Model_run
PARTITIONS="${PARTITIONS:-amp48,ada24}"
EXCLUDE="${EXCLUDE:-colossus}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
TIME="${TIME:-36:00:00}"
SMOKE="${SMOKE:-0}"
NO_EVAL="${NO_EVAL:-0}"
DUMP_ONLY="${DUMP_ONLY:-}"
IMG="${IMG:-768}"
BS="${BS:-4}"
ACCUM="${ACCUM:-12}"
ROOT=/home/psytp7/ISBI_2026
PRIOR_FILE="${PRIOR_FILE:-$ROOT/analysis/out/patient_markov/priors.pt}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoint3/Model_20260119_062652/model_best.pth}"
DUMP="$ROOT/analysis/out/probs_unified_flip"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$PRIOR_FILE" ] || { echo "missing $PRIOR_FILE" >&2; exit 1; }

COMMON=(--accum-steps "$ACCUM" --batch-size "$BS" --monitor mAP --memory-size 2048
        --metric-embed query --head-classes 5 --class-weight-order shard --swa-last-k 3
        --lr 1e-4 --epochs 4 --patience 2 --sched-epochs 8 --drop-path-rate 0.1
        --img-size "$IMG" --stage1-ckpt "$STAGE1_CKPT")
PRIOR=(--prior-file "$PRIOR_FILE" --prior-dropout 0.3 --prior-lr-mult 10)
SMOKE_ARGS=()
[ "$SMOKE" = 1 ] && SMOKE_ARGS=(--epochs 1 --patience 1 --limit-train-batches 40 --limit-val-batches 20 --log-every 10)

RUNS=("$@")
if [ "$#" -eq 0 ] && [ -z "$DUMP_ONLY" ]; then
    for S in 42 123 456 789 1024; do
        for A in a b c d e f m; do
            case "$A:$S" in a:42|a:1024|d:42|d:1024|m:42|m:1024) continue ;; esac   # exist already
            RUNS+=("$A:$S")
        done
    done
fi

mkdir -p "$DUMP"
for f in keys.txt labels.npy; do
    [ -f "$DUMP/$f" ] || cp "$ROOT/analysis/out/probs_v5_orig_recheck_flip/$f" "$DUMP/$f"
done

ckpt_dir() {  # arm seed -> run dir (existing runs live elsewhere)
    case "$1:$2" in
        a:42|a:1024) echo "$ROOT/train/checkpoint_MoE/asl_img${IMG}_seed$2/Model_run" ;;
        m:42|m:1024) echo "$ROOT/train/checkpoint_MoE/moe_img${IMG}_seed$2/Model_run" ;;
        d:42|d:1024) echo "$ROOT/train/checkpoint_Triplet_3/prior/img${IMG}_qprior_seed$2/Model_run" ;;
        *) echo "$ROOT/train/checkpoint_Triplet_3/unified${IMG}/$1/seed_$2/Model_run" ;;
    esac
}

dump_eval() {  # arm seed dependency-jobid-or-empty
    local arm=$1 seed=$2 dep=$3 rd; rd=$(ckpt_dir "$arm" "$seed")
    local base=(sbatch --parsable -p "$PARTITIONS" ${EXCLUDE:+-x "$EXCLUDE,tamar"} --time=4:00:00 ${dep:+--dependency=afterok:$dep})
    case "$arm" in
        d|e)
            for mode in with zero; do
                ej=$("${base[@]}" -J "isbi2026_evuni_${arm}_s${seed}_${mode}" "$SCRIPT_DIR/evaluate_tta_v5.sh" --batch-size "$BS" \
                     --dump-probs "$DUMP" --out-json "$DUMP/eval_${arm}_s${seed}_prior-${mode}.json" \
                     --prior-file "$PRIOR_FILE" --prior-mode "$mode" "$rd/model_swa.pth@$IMG")
                echo "    eval($mode) -> job $ej"
            done ;;
        *)
            ej=$("${base[@]}" -J "isbi2026_evuni_${arm}_s${seed}" "$SCRIPT_DIR/evaluate_tta_v5.sh" --batch-size "$BS" \
                 --dump-probs "$DUMP" --out-json "$DUMP/eval_${arm}_s${seed}.json" "$rd/model_swa.pth@$IMG")
            echo "    eval -> job $ej" ;;
    esac
}

if [ -n "$DUMP_ONLY" ]; then
    for spec in $DUMP_ONLY; do IFS=: read -r arm seed <<< "$spec"; echo "dump $spec"; dump_eval "$arm" "$seed" ""; done
    exit 0
fi

echo "runs: ${RUNS[*]} | img=$IMG bs=$BS accum=$ACCUM | smoke=$SMOKE | partitions=$PARTITIONS exclude=$EXCLUDE"
for spec in "${RUNS[@]}"; do
    IFS=: read -r arm seed <<< "$spec"
    case "$arm" in
        a) armargs=(--no-triplet) ;;
        b) armargs=(--triplet-lambda 0.1 --anchor-rule all) ;;
        c) armargs=(--triplet-lambda 0.1 --anchor-rule mean) ;;
        d) armargs=(--triplet-lambda 0.1 --anchor-rule mean "${PRIOR[@]}") ;;
        e) armargs=(--no-triplet "${PRIOR[@]}") ;;
        f) armargs=(--triplet-lambda 1.0 --anchor-rule mean --mining hard) ;;
        m) armargs=(--arch moe --no-triplet) ;;
        *) echo "unknown arm: $arm (a|b|c|d|e|f|m)" >&2; exit 1 ;;
    esac
    tag=""; [ "$SMOKE" = 1 ] && tag="_smoke"
    rel="checkpoint_Triplet_3/unified${IMG}${tag}/${arm}/seed_${seed}/Model_run"
    name="isbi2026_uni_${arm}_s${seed}${tag}"
    jid=$(sbatch --parsable -J "$name" -p "$PARTITIONS" ${EXCLUDE:+-x "$EXCLUDE"} -c "$CPUS" --mem="$MEM" --time="$TIME" \
          "$SCRIPT_DIR/train_2_v5.sh" --seed "$seed" --out-dir "$rel" "${COMMON[@]}" "${armargs[@]}" "${SMOKE_ARGS[@]}")
    echo "$spec -> job $jid | /home/psytp7/logs/${name}_${jid}.out"
    if [ "$SMOKE" != 1 ] && [ "$NO_EVAL" != 1 ]; then dump_eval "$arm" "$seed" "$jid"; fi
done
