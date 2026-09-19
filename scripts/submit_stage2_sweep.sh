#!/bin/bash
# Submit independent stage-2 runs, one per accum_steps value.
#   ./submit_stage2_sweep.sh            -> 7 8 9 (default)
#   ./submit_stage2_sweep.sh 10 11 12   -> extends the sweep
#
# Each is a separate Slurm job with its own GPU, its own job name (so its own
# %x log) and its own out_dir, since train_2_v2.py keys out_dir on accum_steps:
#   train/checkpoint_Triplet_2/accum_<N>/Model_<timestamp>/
# They are fully independent - no dependencies, no shared writes.
#
# Resource overrides vs train_2_v2.sh's amp48 defaults:
#   -p  Slurm applies a per-partition QOS cap on concurrent per-user usage:
#         amp48-part  cpu=64  gpu=4  mem=640G
#         ada24-part  cpu=32  gpu=4  mem=360G
#         amp20-part  cpu=32  gpu=4  mem=160G
#         amp16-part  cpu=8   gpu=2  mem=32G   (also governs the cs partition)
#       amp16/cs is where the only free GPUs are, but its gpu=2 / cpu=8 cap
#       cannot hold a 3-job sweep at all - and its free nodes are 1080Ti/2080Ti
#       (11GB, no bf16) or A4000 16GB. amp20 is excluded because the pending
#       stage-1 job already holds 1 GPU + 64G of that partition's 160G cap.
#       That leaves amp48 + ada24: both fit the sweep, both are fully allocated
#       right now, so these queue rather than start immediately.
#   -c/--mem  12 CPUs x 100G x 3 would breach ada24-part's cpu=32. 8 x 64G
#       keeps the sweep under every cap (24 CPUs, 192G, 3 GPUs) and matches the
#       8 CPUs stage-1 ran with at the same num_workers=8.
PARTITIONS="${PARTITIONS:-amp48,ada24}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# accum_steps values come from the command line so the sweep can be extended
# without editing this file:  submit_stage2_sweep.sh 10 11 12
ACCUM_STEPS=("$@")
[ ${#ACCUM_STEPS[@]} -eq 0 ] && ACCUM_STEPS=(7 8 9)

for N in "${ACCUM_STEPS[@]}"; do
    jid=$(sbatch --parsable \
        -J "isbi2026_train2v2_accum${N}" \
        -p "$PARTITIONS" -c "$CPUS" --mem="$MEM" \
        "$SCRIPT_DIR/train_2_v2.sh" --accum-steps "$N")
    echo "accum_steps=$N -> job $jid | log /home/psytp7/logs/isbi2026_train2v2_accum${N}_${jid}.out"
done
