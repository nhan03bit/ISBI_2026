#!/bin/bash
# Flip-TTA probability dumps for the matched-recipe MoE / MoE+Markov / ASL-only runs
# (scripts/submit_stage2_moe.sh), each queued afterok on its training job.
# Plan: /home/psytp7/.claude/plans/purring-wandering-quiche.md (2026-10-08).
#
#   ./submit_eval_moe.sh moe:768:42:51153 moeprior:768:42:51156 ...   # run:px:seed:train_jobid
#   ./submit_eval_moe.sh regress                                      # evaluator regression check
#
# Every dump goes to analysis/out/probs_moe_flip, whose keys.txt / labels.npy are copied from
# probs_v5_orig_recheck_flip, so a dump in a different row order fails instead of mis-aligning.
#   moe / asl   model_swa@px (and model_best@640 for the 640 px run, as Triplet Memory's res_sweep)
#   moeprior    model_swa@px with --prior-mode with, and again with --prior-mode zero
#   regress     Triplet Memory push/img768_dp01_seed42 model_swa@768 -> probs_moe_regress, must
#               equal probs_w01_flip exactly (the evaluator's MoE changes must not touch v3 loads)
PARTITIONS="${PARTITIONS:-amp48,ada24}"
EXCLUDE="${EXCLUDE:-colossus}"
BS="${BS:-4}"
ROOT=/home/psytp7/ISBI_2026
OUT="$ROOT/analysis/out/probs_moe_flip"
PRIOR_FILE="${PRIOR_FILE:-$ROOT/analysis/out/patient_markov/priors.pt}"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUT"
for f in keys.txt labels.npy; do
    [ -f "$OUT/$f" ] || cp "$ROOT/analysis/out/probs_v5_orig_recheck_flip/$f" "$OUT/$f"
done

submit() {   # name dependency dump_dir out_json evaluator-args...
    local name=$1 dep=$2 dir=$3 js=$4; shift 4
    sbatch --parsable -J "$name" -p "$PARTITIONS" ${EXCLUDE:+-x "$EXCLUDE"} ${dep:+--dependency=afterok:$dep} \
        "$SCRIPT_DIR/evaluate_tta_v5.sh" --batch-size "$BS" --dump-probs "$dir" --out-json "$js" "$@"
}

for spec in "$@"; do
    if [ "$spec" = regress ]; then
        d="$ROOT/analysis/out/probs_moe_regress"; mkdir -p "$d"
        ck="$ROOT/train/checkpoint_Triplet_3/push/img768_dp01_seed42/Model_run/model_swa.pth"
        jid=$(submit isbi2026_eval_moe_regress "" "$d" "$d/eval.json" "$ck@768")
        echo "regress -> job $jid"
        continue
    fi
    IFS=: read -r run px seed dep <<< "$spec"
    rd="$ROOT/train/checkpoint_MoE/${run}_img${px}_seed${seed}/Model_run"
    js="$OUT/eval_${run}_img${px}_s${seed}"
    cks=("$rd/model_swa.pth@$px")
    [ "$px" = 640 ] && cks+=("$rd/model_best.pth@$px")
    case "$run" in
        moe|asl)
            jid=$(submit "isbi2026_eval_${run}_img${px}_s${seed}" "$dep" "$OUT" "$js.json" "${cks[@]}")
            echo "$spec -> job $jid" ;;
        moeprior)
            for mode in with zero; do
                jid=$(submit "isbi2026_eval_${run}_img${px}_s${seed}_${mode}" "$dep" "$OUT" "${js}_prior-${mode}.json" \
                      --prior-file "$PRIOR_FILE" --prior-mode "$mode" "${cks[@]}")
                echo "$spec ($mode) -> job $jid"
            done ;;
        *) echo "unknown run: $run" >&2; exit 1 ;;
    esac
done
