#!/bin/bash
# Track A Wave-2: train_2_v3.py post-45821..45825 grid.
#
#   ./submit_stage2_v3_sweep.sh              # Wave-2 grid (8 jobs, seed 86)
#   SEED=123 ./submit_stage2_v3_sweep.sh     # same grid, different seed
#   SMOKE=1 ./submit_stage2_v3_sweep.sh      # 1-epoch / 40-batch smoke of every arm
#
# What Wave-1 (jobs 45821-45825, the `q_lr*_mem*` + `legacy` grid, kept below for
# the record) established:
#   * best mAP 0.4350 (lr 1e-4) vs the v2 plateau 0.414; lr 1e-4 > lr 5e-5
#   * memory-size 2048 == 4096 (null); metric-embed query == legacy on mAP but
#     query has clearly better mECE -> fix lr 1e-4, memory 2048, query
#   * EVERY run peaks at epoch 2 then overfits for the rest of the 8-epoch budget
#     (train_loss keeps falling, val_loss + mAP get worse); the cosine LR horizon
#     is still ~90% of peak at that epoch-2 optimum
#   * class_weights + head_class_ids were computed in CSV order but applied to
#     shard-order logits (train_2_v2's reindex was dropped) -> Hydropneumothorax
#     weighted 0.58 not 2.00, "head" exclusion masking 5 tail/mid classes
#
# Wave-2 fixes the ordering bug (`--class-weight-order shard`, now the default)
# and sweeps the anti-overfit knobs added to train_2_v3.py:
#   --sched-epochs N          cosine LR horizon in epochs, decoupled from --epochs
#   --freeze-backbone-stages N freeze downsample_layers[:N]+stages[:N]
#   --llrd GAMMA              layer-wise LR decay lr*GAMMA**(depth from head)
#   --drop-path-rate R        ConvNeXt2 stochastic depth
#
# Cluster: QOS gpu=4 per partition; amp48,ada24 -> all 8 start. ~2 h/epoch,
# --time 36h covers the longest (8-epoch) arms.
#
# ---- Wave-1 grid (jobs 45821-45825), kept for reference ---------------------
#   for LR in 5e-5 1e-4; do for MEM in 2048 4096; do
#       submit "q_lr${LR}_mem${MEM}" --metric-embed query --lr "$LR" \
#              --memory-size "$MEM" --head-classes 5
#   done; done
#   submit "legacy_lr1e-4_mem4096" --metric-embed legacy --lr 1e-4 \
#          --memory-size 4096 --head-classes 5
# ---------------------------------------------------------------------------
PARTITIONS="${PARTITIONS:-amp48,ada24}"
# colossus (amp48) intermittently fails torch CUDA init ("CUDA unknown error",
# job silently falls back to CPU at ~80x slowdown) - excluded by default.
# EXCLUDE= (empty) to allow it back in.
EXCLUDE="${EXCLUDE:-colossus}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
# v3 runs reliably early-stop by ~epoch 5 (~11h wall on amp48); 16h gives margin
# while staying short enough for the backfill scheduler to slot jobs into gaps.
TIME="${TIME:-16:00:00}"
SEED="${SEED:-86}"
SMOKE="${SMOKE:-0}"
# space-separated subset of arm tags to submit; empty = all.
ARMS="${ARMS:-}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoint3/Model_20260119_062652/model_best.pth}"

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fixed from the Wave-1 findings. --class-weight-order shard is the new default
# but pinned here explicitly; the `bug_repro` arm overrides it back to csv.
COMMON=(--accum-steps 6 --batch-size 8 --monitor mAP
        --lr 1e-4 --memory-size 2048 --metric-embed query --head-classes 5
        --triplet-lambda 0.1 --class-weight-order shard
        --stage1-ckpt "$STAGE1_CKPT")
if [ "$SMOKE" = 1 ]; then
    COMMON+=(--limit-train-batches 40 --limit-val-batches 20 --log-every 20)
fi

echo "seed=$SEED smoke=$SMOKE | partitions=$PARTITIONS | stage1_ckpt=$STAGE1_CKPT"
echo

submit () {  # $1 = tag, rest = extra train_2_v3.py args (later flags win)
    local tag="$1"; shift
    if [ -n "$ARMS" ] && [[ " $ARMS " != *" $tag "* ]]; then
        echo "$tag -> skipped (not in ARMS)"
        return 0
    fi
    local smoke_ovr=()
    [ "$SMOKE" = 1 ] && smoke_ovr=(--epochs 1 --patience 1)
    local jid
    jid=$(sbatch --parsable \
        -J "isbi2026_v3s2_${tag}_s${SEED}" \
        -p "$PARTITIONS" ${EXCLUDE:+-x "$EXCLUDE"} -c "$CPUS" --mem="$MEM" --time="$TIME" \
        "$SCRIPT_DIR/train_2_v3.sh" \
            --seed "$SEED" \
            --out-dir "checkpoint_Triplet_3/sweep2/${tag}/seed_${SEED}/Model_run" \
            "${COMMON[@]}" "$@" "${smoke_ovr[@]}")
    echo "$tag -> job $jid | /home/psytp7/logs/isbi2026_v3s2_${tag}_s${SEED}_${jid}.out"
}

# --- Wave-2 grid (seed 86) -------------------------------------------------
# tag         isolates
submit fix_only   --epochs 8 --patience 3                                   # bug fix #1 alone, vs 0.435
submit bug_repro  --epochs 8 --patience 3 --class-weight-order csv           # control: must land ~= 0.435
submit sched3     --epochs 5 --patience 2 --sched-epochs 3                   # short cosine horizon
submit llrd07     --epochs 8 --patience 3 --llrd 0.7                         # layer-wise LR decay
submit freeze3    --epochs 8 --patience 3 --freeze-backbone-stages 3         # freeze early backbone
submit freeze4    --epochs 8 --patience 3 --freeze-backbone-stages 4         # freeze more
submit dp01       --epochs 8 --patience 3 --drop-path-rate 0.1              # stochastic depth
submit combo      --epochs 5 --patience 2 --sched-epochs 3 --llrd 0.7        # best-guess stack

echo
echo "watch:  squeue --me"
echo "curves: grep -H 'Epoch .*mAP=' /home/psytp7/logs/isbi2026_v3s2_*_s${SEED}_*.out"
echo "        find /home/psytp7/ISBI_2026/train/checkpoint_Triplet_3/sweep2 -name history.json | xargs -I{} sh -c 'echo {}; cat {}'"
echo
echo "gate: confirm bug_repro best-epoch mAP ~= 0.435 before trusting the rest."
