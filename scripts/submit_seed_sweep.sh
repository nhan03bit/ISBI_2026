#!/bin/bash
# Submit independent stage-2 runs at a fixed accum_steps, one per seed.
#   ./submit_seed_sweep.sh                         -> accum_steps=6, seeds 86 123 456 789 1024 (default)
#   ./submit_seed_sweep.sh 6 86 123 456 789 1024    -> explicit: first arg is accum_steps, rest are seeds
#
# Each is a separate Slurm job with its own GPU, its own job name (so its own
# %x log per README section 10) and its own out_dir, since train_2_v2.py now
# keys out_dir on both accum_steps and seed:
#   train/checkpoint_Triplet_2/accum_<N>/seed_<S>/Model_<timestamp>/
# They are fully independent - no dependencies, no shared writes.
#
# Caveat carried over from the accum_steps sweep: AUG_BASE_SEED in
# train_2_v2.py is a fixed module constant, not tied to --seed. Per-sample
# augmentation is identical across all seeds below - only weight init and
# dataloader shuffle order vary. That isolates model-level stochasticity,
# which is usually what a seed sweep is for, but it means these five runs
# will NOT diverge as much as if augmentation varied too.
#
# Resource caps (README section 6, and observed via `qos $USER` on this
# account): per-partition QOS limits concurrent per-user usage regardless of
# free hardware -
#   amp48-part  cpu=64  gpu=4  mem=640G
#   ada24-part  cpu=32  gpu=4  mem=360G
#   amp20-part  cpu=32  gpu=4  mem=160G
# Five GPUs exceeds amp48's own gpu=4 cap, so the list below lets Slurm split
# the batch across amp48 and ada24 rather than queuing the fifth job. Runtimes
# will not be directly comparable across that hardware split (A6000 48GB vs
# Ada 4500 24GB) - the accum_steps sweep found a ~0.001 mAP shift between them.
PARTITIONS="${PARTITIONS:-amp48,ada24}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -eq 0 ]; then
    ACCUM=6
    SEEDS=(86 123 456 789 1024)
else
    ACCUM="$1"; shift
    SEEDS=("$@")
fi

for S in "${SEEDS[@]}"; do
    jid=$(sbatch --parsable \
        -J "isbi2026_train2v2_accum${ACCUM}_seed${S}" \
        -p "$PARTITIONS" -c "$CPUS" --mem="$MEM" \
        "$SCRIPT_DIR/train_2_v2.sh" --accum-steps "$ACCUM" --seed "$S")
    echo "accum_steps=$ACCUM seed=$S -> job $jid | log /home/psytp7/logs/isbi2026_train2v2_accum${ACCUM}_seed${S}_${jid}.out"
done
