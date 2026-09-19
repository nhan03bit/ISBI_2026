#!/bin/bash
# Corrected-baseline SEED sweep for train_2_v3.py.
#
# Runs the fixed recipe (--class-weight-order shard) across 6 seeds to establish
# the seed distribution of the ORDERING-FIXED v3 stage-2, against the paper's
# Table I (mAP 0.414 +/- 0.002, measured WITH the class-weight ordering bug).
#
#   ./submit_stage2_v3_seedsweep.sh                    # seeds 42 86 123 456 789 1024
#   ./submit_stage2_v3_seedsweep.sh 123 456            # explicit seed list
#   SMOKE=1 ./submit_stage2_v3_seedsweep.sh
#   AUGVARY=1 ./submit_stage2_v3_seedsweep.sh          # also vary augmentation per seed
#
# aug-base-seed stays at its default (42, fixed) unless AUGVARY=1, so by default
# this isolates model-level stochasticity (weight init + dataloader shuffle) -
# matching how the paper's 5-seed std was measured.
#
# Config = the `fix_only` arm of submit_stage2_v3_sweep.sh:
#   lr 1e-4, memory 2048, metric-embed query, head-classes 5, triplet-lambda 0.1,
#   eff batch 48 (accum 6 x batch 8), 8 epochs / patience 3, monitor mAP,
#   converged Stage-1 ckpt. out_dir: train/checkpoint_Triplet_3/seedsweep/seed_<S>/Model_run
#
# NOTE seed 86 of this exact config is also running as `fix_only` in
# submit_stage2_v3_sweep.sh; seedsweep/seed_86 is an independent rerun / repro check.
PARTITIONS="${PARTITIONS:-amp48,ada24}"
# colossus silently fails torch CUDA init -> CPU fallback; excluded by default.
EXCLUDE="${EXCLUDE:-colossus}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-16:00:00}"
SMOKE="${SMOKE:-0}"
AUGVARY="${AUGVARY:-0}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoint3/Model_20260119_062652/model_best.pth}"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEEDS=("$@")
[ "$#" -eq 0 ] && SEEDS=(42 86 123 456 789 1024)

COMMON=(--accum-steps 6 --batch-size 8 --monitor mAP --epochs 8 --patience 3
        --lr 1e-4 --memory-size 2048 --metric-embed query --head-classes 5
        --triplet-lambda 0.1 --class-weight-order shard
        --stage1-ckpt "$STAGE1_CKPT")
if [ "$SMOKE" = 1 ]; then
    COMMON+=(--epochs 1 --patience 1 --limit-train-batches 40 --limit-val-batches 20 --log-every 20)
fi

echo "seeds: ${SEEDS[*]} | augvary=$AUGVARY smoke=$SMOKE | partitions=$PARTITIONS exclude=$EXCLUDE"
echo

for S in "${SEEDS[@]}"; do
    aug=()
    [ "$AUGVARY" = 1 ] && aug=(--aug-base-seed "$S")
    jid=$(sbatch --parsable \
        -J "isbi2026_v3seed_s${S}" \
        -p "$PARTITIONS" ${EXCLUDE:+-x "$EXCLUDE"} -c "$CPUS" --mem="$MEM" --time="$TIME" \
        "$SCRIPT_DIR/train_2_v3.sh" \
            --seed "$S" \
            --out-dir "checkpoint_Triplet_3/seedsweep/seed_${S}/Model_run" \
            "${COMMON[@]}" "${aug[@]}")
    echo "seed $S -> job $jid | /home/psytp7/logs/isbi2026_v3seed_s${S}_${jid}.out"
done

echo
echo "watch:  scripts/mon.py 'isbi2026_v3seed_*'    (or scripts/mon.py for the full v3 board)"
echo "curves: scripts/mon.py 'isbi2026_v3seed_*' --epochs"
