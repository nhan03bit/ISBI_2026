#!/bin/bash
# ASL-only system on its own script (train/train_2_v0.py: ConvNeXt2 + asymmetric loss, nothing else), 2026-10-13.
# Counterparts on train/train_2_v5.py: Triplet Memory + Entropy (unified768/c) and + Markov prior (unified768/d,
# prior/img768_qprior_*, best = job 49375). Plan: docs/plan/2026-10-13-asl-v0-vs-entropy-v5-plan.md
#
# Recipe = job 49375 minus the Triplet Memory / prior flags: 768 px, bs 4 x accum 12 (effective 48), 4 of 8 cosine
# epochs, patience 2, drop-path 0.1, SWA of the best 3 EMA epochs, shuffle buffer 400, lr 1e-4.
#
#   SMOKE=1 ./submit_stage2_v0.sh 42            # 40-batch check first
#   ./submit_stage2_v0.sh                        # seeds in wave order: 42 123 1024 456 789
#   RESUME=1 ./submit_stage2_v0.sh 42            # continue seed 42 from <out_dir>/last.pth (written every epoch)
#   NO_EVAL=1 ./submit_stage2_v0.sh 42 123       # training only
#   OUT_TAG=seeded_trainarea TRAIN_ARGS="--train-interp area --val-interp area" EVAL_INTERPS=area ./submit_stage2_v0.sh 42
# out_dir: train/checkpoint_ASL/v0_768/seed_<S>/Model_run
# Eval: flip-TTA probability dump of model_swa.pth at 768 px into analysis/out/probs_v0_flip, eval_v0_s<S>.json;
# one eval job per EVAL_INTERPS entry (evaluate_tta.py --interp; non-lanczos JSONs get an _<interp> suffix).
PARTITIONS="${PARTITIONS:-amp48,ada24}"
EXCLUDE="${EXCLUDE-colossus}"   # EXCLUDE= (empty) allows colossus; the wrappers abort on a dead GPU
TRAIN_ARGS="${TRAIN_ARGS:-}"    # extra train_2_v0.py args, appended after the common recipe (later flags win)
EVAL_INTERPS="${EVAL_INTERPS:-lanczos}"
CPUS="${CPUS:-8}"
MEM="${MEM:-96G}"
TIME="${TIME:-36:00:00}"
SMOKE="${SMOKE:-0}"
NO_EVAL="${NO_EVAL:-0}"
RESUME="${RESUME:-0}"
IMG="${IMG:-768}"
OUT_TAG="${OUT_TAG:-}"          # e.g. seeded -> checkpoint_ASL/v0_768_seeded, probs_v0_flip_seeded (keeps older runs untouched)
ROOT=/home/psytp7/ISBI_2026
STAGE1_CKPT="${STAGE1_CKPT:-checkpoint3/Model_20260119_062652/model_best.pth}"
DUMP="$ROOT/analysis/out/probs_v0_flip${OUT_TAG:+_$OUT_TAG}"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEEDS=("$@")
[ "$#" -eq 0 ] && SEEDS=(42 123 1024 456 789)

COMMON=(--accum-steps 12 --batch-size 4 --monitor mAP --epochs 4 --patience 2 --sched-epochs 8
        --lr 1e-4 --drop-path-rate 0.1 --img-size "$IMG" --shuffle-buf 400 --swa-last-k 3
        --class-weight-order shard --stage1-ckpt "$STAGE1_CKPT")
read -r -a EXTRA <<< "$TRAIN_ARGS"
[ "$SMOKE" = 1 ] && EXTRA+=(--epochs 1 --patience 1 --limit-train-batches 40 --limit-val-batches 20)
[ "$RESUME" = 1 ] && EXTRA+=(--resume)

mkdir -p "$DUMP"
for f in keys.txt labels.npy; do
    [ -f "$DUMP/$f" ] || cp "$ROOT/analysis/out/probs_unified_flip/$f" "$DUMP/$f"
done

echo "seeds: ${SEEDS[*]} | img=$IMG | smoke=$SMOKE resume=$RESUME | partitions=$PARTITIONS exclude=$EXCLUDE"
for seed in "${SEEDS[@]}"; do
    tag=""; [ "$SMOKE" = 1 ] && tag="_smoke"
    rel="checkpoint_ASL/v0_${IMG}${OUT_TAG:+_$OUT_TAG}${tag}/seed_${seed}/Model_run"
    name="isbi2026_v0${OUT_TAG:+_$OUT_TAG}_s${seed}${tag}"
    jid=$(sbatch --parsable -J "$name" -p "$PARTITIONS" ${EXCLUDE:+-x "$EXCLUDE"} -c "$CPUS" --mem="$MEM" --time="$TIME" \
          "$SCRIPT_DIR/train_2_v0.sh" --seed "$seed" --out-dir "$rel" "${COMMON[@]}" "${EXTRA[@]}")
    echo "v0:$seed -> job $jid | /home/psytp7/logs/${name}_${jid}.out"
    if [ "$SMOKE" != 1 ] && [ "$NO_EVAL" != 1 ]; then
        for interp in $EVAL_INTERPS; do
            sfx=""; [ "$interp" != lanczos ] && sfx="_$interp"
            ej=$(sbatch --parsable -p "$PARTITIONS" -x "${EXCLUDE:+$EXCLUDE,}tamar" --time=4:00:00 --dependency=afterok:"$jid" \
                 -J "isbi2026_evv0${OUT_TAG:+_$OUT_TAG}_s${seed}${sfx}" "$SCRIPT_DIR/evaluate_tta_v5.sh" --batch-size 4 \
                 --interp "$interp" --dump-probs "$DUMP" --out-json "$DUMP/eval_v0_s${seed}${sfx}.json" \
                 "$ROOT/train/$rel/model_swa.pth@$IMG")
            echo "    eval($interp) -> job $ej (afterok:$jid)"
        done
    fi
done
