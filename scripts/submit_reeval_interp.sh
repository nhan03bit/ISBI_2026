#!/bin/bash
# Re-score finished Table 1 runs with another eval resize filter (evaluate_tta.py --interp), 2026-10-16.
# Why: training shrinks images with INTER_LINEAR, every eval used INTER_LANCZOS4, and the antialiased INTER_AREA
# scores best (docs/plan/2026-10-16-asl-046-ceiling-investigation.md). Same flip-TTA SWA dumps as the original
# evals, written to <original dump dir>_<interp> next to a copy of its keys.txt / labels.npy, so the rows line up
# with the lanczos dumps. Arm d is scored twice (--prior-mode with / zero).
#   ./scripts/submit_reeval_interp.sh                    # INTERP=area, seeded reruns then the 5-seed Table 1 runs
#   SETS=seeded ./scripts/submit_reeval_interp.sh        # only the seeded seed-42 reruns (v0, c, d, p)
#   SETS=unseeded ONLY="v0_s123 d_s789_prior-zero" ./scripts/submit_reeval_interp.sh   # only these evals of the set
# Eval-only (fits the A4000/A4500 nodes): 1 GPU, 4 CPUs, 16G each (an eval peaks at ~3.2G RSS), so 2 jobs run at
# once on amp16 (QOS cpu=8, mem=32G) and up to 4 on amp20.
# EXCLUDE defaults to rogue (amp20's only node): on 2026-10-16 all 8 re-evals placed there made no progress and hit
# the 3 h limit, while every one on dart / tamar finished in ~47 min.
PARTITIONS="${PARTITIONS:-amp20,amp16}"
INTERP="${INTERP:-area}"
SETS="${SETS:-seeded unseeded}"
ONLY="${ONLY:-}"            # space-separated eval names (<name>[_prior-<mode>]); empty = every eval of SETS
EXCLUDE="${EXCLUDE:-rogue}"
ROOT=/home/psytp7/ISBI_2026
O=$ROOT/analysis/out
T=$ROOT/train
PRIOR_FILE=$O/patient_markov/priors.pt

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ "$INTERP" != lanczos ] || { echo "INTERP=lanczos would rewrite the original dumps" >&2; exit 1; }

submit() {  # source-dump-dir eval-name checkpoint [prior-mode]
    local src=$1 name=$2 ckpt=$3 mode=${4:-} dst="${1}_${INTERP}" sfx="" pargs=()
    if [ -n "$mode" ]; then pargs=(--prior-file "$PRIOR_FILE" --prior-mode "$mode"); sfx="_prior-$mode"; fi
    if [ -n "$ONLY" ] && [[ " $ONLY " != *" ${name}${sfx} "* ]]; then return 0; fi
    [ -f "$ckpt" ] || { echo "missing $ckpt" >&2; exit 1; }
    mkdir -p "$dst"
    for f in keys.txt labels.npy; do [ -f "$dst/$f" ] || cp "$src/$f" "$dst/$f"; done
    jid=$(sbatch --parsable -p "$PARTITIONS" ${EXCLUDE:+--exclude="$EXCLUDE"} -c 4 --mem=16G --time=3:00:00 \
          -J "isbi2026_re${INTERP}_${name}${sfx}" \
          "$SCRIPT_DIR/evaluate_tta_v5.sh" --batch-size 4 --num-workers 4 --interp "$INTERP" \
          --dump-probs "$dst" --out-json "$dst/eval_${name}${sfx}.json" "${pargs[@]}" "$ckpt@768")
    echo "${name}${sfx} -> job $jid | $dst"
}

for set in $SETS; do
    case "$set" in
        seeded)
            submit "$O/probs_v0_flip_seeded" v0_s42 "$T/checkpoint_ASL/v0_768_seeded/seed_42/Model_run/model_swa.pth"
            U="$T/checkpoint_Triplet_3/unified768_seeded"
            submit "$O/probs_unified_flip_seeded" c_s42 "$U/c/seed_42/Model_run/model_swa.pth"
            submit "$O/probs_unified_flip_seeded" d_s42 "$U/d/seed_42/Model_run/model_swa.pth" with
            submit "$O/probs_unified_flip_seeded" d_s42 "$U/d/seed_42/Model_run/model_swa.pth" zero
            submit "$O/probs_unified_flip_seeded" p_s42 "$U/p/seed_42/Model_run/model_swa.pth" ;;
        unseeded)
            for S in 42 123 456 789 1024; do
                submit "$O/probs_v0_flip" "v0_s$S" "$T/checkpoint_ASL/v0_768/seed_$S/Model_run/model_swa.pth"
                submit "$O/probs_unified_flip" "c_s$S" "$T/checkpoint_Triplet_3/unified768/c/seed_$S/Model_run/model_swa.pth"
                case "$S" in   # d:42 and d:1024 are the original prior runs (seed 42 = job 49375)
                    42|1024) D="$T/checkpoint_Triplet_3/prior/img768_qprior_seed$S/Model_run/model_swa.pth" ;;
                    *) D="$T/checkpoint_Triplet_3/unified768/d/seed_$S/Model_run/model_swa.pth" ;;
                esac
                submit "$O/probs_unified_flip" "d_s$S" "$D" with
                submit "$O/probs_unified_flip" "d_s$S" "$D" zero
            done ;;
        *) echo "unknown set: $set (seeded|unseeded)" >&2; exit 1 ;;
    esac
done
