#!/bin/bash
# Read-only Triplet Memory probe (analysis/triplet_probe.py; plan docs/plan/2026-10-15-triplet-memory-debug-plan.md).
# No training: replays the v5 step on the seeded train stream for the Stage-1 init, seeded arm c (job 52650)
# and seeded ASL v0 (job 52648), writing analysis/out/triplet_probe/<tag>.json.
#   SMOKE=1 ./scripts/submit_triplet_probe.sh     # 20 warm-up + 20 batches, 30 min
#   ./scripts/submit_triplet_probe.sh             # 512 warm-up + 300 batches per checkpoint
PARTITIONS="${PARTITIONS:-amp48,ada24}"
EXCLUDE="${EXCLUDE:-colossus,tamar}"
SMOKE="${SMOKE:-0}"
ROOT=/home/psytp7/ISBI_2026
set -euo pipefail

CKPTS=(--ckpt stage1=stage1
       --ckpt c=checkpoint_Triplet_3/unified768_seeded/c/seed_42/Model_run/model_best.pth
       --ckpt asl=checkpoint_ASL/v0_768_seeded/seed_42/Model_run/model_best.pth)
MODE="${MODE:-1}"               # 2 = analysis/triplet_probe2.py (why the triplet gradient does not reach the classifier)
PROBE=$ROOT/analysis/triplet_probe.py
if [ "$MODE" = 2 ]; then
    PROBE=$ROOT/analysis/triplet_probe2.py
    CKPTS=(--ckpt stage1=stage1 --ckpt c=checkpoint_Triplet_3/unified768_seeded/c/seed_42/Model_run/model_best.pth)
fi
if [ "$MODE" = 3 ]; then         # analysis/triplet_probe3.py (why lambda 1.0 is worse: restoring-constraint test)
    PROBE=$ROOT/analysis/triplet_probe3.py
    CKPTS=()
fi
if [ "$SMOKE" = 1 ] && [ "$MODE" = 3 ]; then
    EXTRA=(--opt-steps 4 --log-every 2 --eval-every 2 --val-batches 4 --warmup 20 --static-windows 2 --lams 0,1.0
           --out-dir "$ROOT/analysis/out/triplet_probe3_smoke"); TIME=0:30:00; NAME=isbi2026_tprobe3_smoke
elif [ "$MODE" = 3 ]; then
    EXTRA=(${P3_ARGS:-}); TIME=3:00:00; NAME=isbi2026_tprobe3${P3_TAG:-}
elif [ "$SMOKE" = 1 ] && [ "$MODE" = 2 ]; then
    EXTRA=(--warmup 20 --windows 2 --val-batches 2 --out-dir "$ROOT/analysis/out/triplet_probe2_smoke"); TIME=0:30:00; NAME=isbi2026_tprobe2_smoke
elif [ "$MODE" = 2 ]; then
    EXTRA=(); TIME=4:00:00; NAME=isbi2026_tprobe2
elif [ "$SMOKE" = 1 ]; then
    EXTRA=(--warmup 20 --batches 20 --grad-batches 10 --out-dir "$ROOT/analysis/out/triplet_probe_smoke"); TIME=0:30:00; NAME=isbi2026_tprobe_smoke
else
    EXTRA=(); TIME=4:00:00; NAME=isbi2026_tprobe
fi

sbatch --parsable -J "$NAME" -p "$PARTITIONS" -x "$EXCLUDE" -G 1 -c 8 --mem=64G --time="$TIME" \
    -o "/home/psytp7/logs/%x_%j.out" --wrap "
echo \"=== \$SLURM_JOB_NAME (\$SLURM_JOB_ID) on \$(hostname) \$(date) ===\"
source $ROOT/.venv/bin/activate
cd $ROOT/train
python $PROBE ${CKPTS[*]} ${EXTRA[*]}
echo \"=== Finished \$(date) ===\""
