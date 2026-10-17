#!/bin/bash
# Triplet Memory + Entropy (arm c) and Triplet Memory + Entropy + Markov prior (arm d) on train/train_2_v5.py, at 768 px,
# on the job-49375 recipe (drop-path 0.1, 4 of 8 cosine epochs, SWA of the last 3 EMA epochs, effective batch 48), seeds
# 42 / 123 / 456 / 789 / 1024. The ASL-only system trains on train/train_2_v0.py (scripts/submit_stage2_v0.sh), never here.
# Plan: docs/plan/2026-10-13-asl-v0-vs-entropy-v5-plan.md. Arms (run spec = arm:seed):
#   c  entropy gate                       --triplet-lambda 0.1 --anchor-rule mean
#   d  entropy gate + Markov prior        c + PRIOR flags
# p = sensitivity probe: arm c with --triplet-lambda 1.0 (not part of Table 1). OUT_TAG=seeded retrains every arm in
# unified768_seeded/ (the seeded sample-buffer shuffle, 2026-10-14) and writes dumps to probs_unified_flip_seeded.
# d:42 and d:1024 already exist (prior/img768_qprior_seed{42,1024}; seed 42 = job 49375); all c and d seeds are finished.
# Each training job is followed (afterok) by flip-TTA probability dumps of model_swa.pth at 768 px into
# analysis/out/probs_unified_flip (arm d twice: --prior-mode with and zero = history withheld).
#
#   SMOKE=1 ./submit_stage2_v5_unified.sh c:42 d:42               # 40-batch check first
#   ./submit_stage2_v5_unified.sh                                 # every c/d seed that has no run yet
#   NO_EVAL=1 ...  /  DUMP_ONLY="c:42 d:42" ./submit_stage2_v5_unified.sh   # re-dump existing runs
# out_dir: train/checkpoint_Triplet_3/unified768/<arm>/seed_<S>/Model_run
PARTITIONS="${PARTITIONS:-amp48,ada24}"
EXCLUDE="${EXCLUDE-colossus}"   # EXCLUDE= (empty) allows every node; train_2_v5.sh aborts on a dead GPU
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
TIME="${TIME:-36:00:00}"
SMOKE="${SMOKE:-0}"
NO_EVAL="${NO_EVAL:-0}"
DUMP_ONLY="${DUMP_ONLY:-}"
IMG="${IMG:-768}"
OUT_TAG="${OUT_TAG:-}"          # e.g. seeded -> unified768_seeded/<arm> and probs_unified_flip_seeded; every arm is retrained there
EVAL_INTERPS="${EVAL_INTERPS:-lanczos}"   # eval resize filters for arms other than d, e.g. "lanczos area"
BS="${BS:-4}"
ACCUM="${ACCUM:-12}"
ROOT=/home/psytp7/ISBI_2026
PRIOR_FILE="${PRIOR_FILE:-$ROOT/analysis/out/patient_markov/priors.pt}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoint3/Model_20260119_062652/model_best.pth}"
DUMP="$ROOT/analysis/out/probs_unified_flip${OUT_TAG:+_$OUT_TAG}"

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

ckpt_dir() {  # arm seed -> run dir (existing runs live elsewhere)
    case "${OUT_TAG:+tag}:$1:$2" in
        :d:42|:d:1024) echo "$ROOT/train/checkpoint_Triplet_3/prior/img${IMG}_qprior_seed$2/Model_run" ;;
        *) echo "$ROOT/train/checkpoint_Triplet_3/unified${IMG}${OUT_TAG:+_$OUT_TAG}/$1/seed_$2/Model_run" ;;
    esac
}

RUNS=("$@")
if [ "$#" -eq 0 ] && [ -z "$DUMP_ONLY" ]; then
    for S in 42 123 456 789 1024; do
        for A in c d; do
            [ -f "$(ckpt_dir "$A" "$S")/model_swa.pth" ] || RUNS+=("$A:$S")   # skip finished runs
        done
    done
fi

mkdir -p "$DUMP"
for f in keys.txt labels.npy; do
    [ -f "$DUMP/$f" ] || cp "$ROOT/analysis/out/probs_v5_orig_recheck_flip/$f" "$DUMP/$f"
done

dump_eval() {  # arm seed dependency-jobid-or-empty
    local arm=$1 seed=$2 dep=$3 rd; rd=$(ckpt_dir "$arm" "$seed")
    local base=(sbatch --parsable -p "$PARTITIONS" ${EXCLUDE:+-x "$EXCLUDE,tamar"} --time=4:00:00 ${dep:+--dependency=afterok:$dep})
    case "$arm" in
        d)
            for mode in with zero; do
                ej=$("${base[@]}" -J "isbi2026_evuni${OUT_TAG:+_$OUT_TAG}_${arm}_s${seed}_${mode}" "$SCRIPT_DIR/evaluate_tta_v5.sh" --batch-size "$BS" \
                     --dump-probs "$DUMP" --out-json "$DUMP/eval_${arm}_s${seed}_prior-${mode}.json" \
                     --prior-file "$PRIOR_FILE" --prior-mode "$mode" "$rd/model_swa.pth@$IMG")
                echo "    eval($mode) -> job $ej"
            done ;;
        *)
            for interp in $EVAL_INTERPS; do   # non-lanczos dumps / JSONs get an _<interp> suffix (evaluate_tta.py)
                sfx=""; [ "$interp" != lanczos ] && sfx="_$interp"
                ej=$("${base[@]}" -J "isbi2026_evuni${OUT_TAG:+_$OUT_TAG}_${arm}_s${seed}${sfx}" "$SCRIPT_DIR/evaluate_tta_v5.sh" \
                     --batch-size "$BS" --interp "$interp" --dump-probs "$DUMP" --out-json "$DUMP/eval_${arm}_s${seed}${sfx}.json" \
                     "$rd/model_swa.pth@$IMG")
                echo "    eval($interp) -> job $ej"
            done ;;
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
        c) armargs=(--triplet-lambda 0.1 --anchor-rule mean) ;;
        d) armargs=(--triplet-lambda 0.1 --anchor-rule mean "${PRIOR[@]}") ;;
        p) armargs=(--triplet-lambda 1.0 --anchor-rule mean) ;;   # sensitivity probe: arm c with a 10x triplet weight
        # class-conditional memory loss (2026-10-16, train/class_memory.py, docs/plan/2026-10-16-beat-asl-plan.md);
        # hyperparameters are the train_2_v5.py defaults, fixed before any run (tau 0.1, bank 4096, queue 128,
        # lambda 1, ratio cap 1, warm-up 1024)
        n)  armargs=(--metric-loss nca --nca-gate mean) ;;     # entropy gate per (image, class)
        na) armargs=(--metric-loss nca --nca-gate all) ;;      # control: no gate
        nr) armargs=(--metric-loss nca --nca-gate random) ;;   # control: random gate, same number of pairs
        *) echo "unknown arm: $arm (c|d|p|n|na|nr)" >&2; exit 1 ;;
    esac
    tag=""; [ "$SMOKE" = 1 ] && tag="_smoke"
    rel="checkpoint_Triplet_3/unified${IMG}${OUT_TAG:+_$OUT_TAG}${tag}/${arm}/seed_${seed}/Model_run"
    name="isbi2026_uni${OUT_TAG:+_$OUT_TAG}_${arm}_s${seed}${tag}"
    jid=$(sbatch --parsable -J "$name" -p "$PARTITIONS" ${EXCLUDE:+-x "$EXCLUDE"} -c "$CPUS" --mem="$MEM" --time="$TIME" \
          "$SCRIPT_DIR/train_2_v5.sh" --seed "$seed" --out-dir "$rel" "${COMMON[@]}" "${armargs[@]}" "${SMOKE_ARGS[@]}")
    echo "$spec -> job $jid | /home/psytp7/logs/${name}_${jid}.out"
    if [ "$SMOKE" != 1 ] && [ "$NO_EVAL" != 1 ]; then dump_eval "$arm" "$seed" "$jid"; fi
done
