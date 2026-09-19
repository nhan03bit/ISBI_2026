#!/bin/bash
# Long "train to convergence" sweep for stage-2 at a fixed accum_steps, one job
# per seed, early stopping effectively disabled so the full mAP curve is logged.
#
#   ./submit_converge_sweep.sh                    -> accum 8, seeds 42 86 123 456 789, 30 epochs
#   ./submit_converge_sweep.sh 42 86              -> same config, only those seeds
#   ACCUM=8 EPOCHS=30 PATIENCE=999 ./submit_converge_sweep.sh   -> override via env
#
# Rationale (see the research report / job 44119 analysis):
# - train_2_v2.py has NO resume support, so this cannot continue 44119; it is a
#   fresh set of runs from the stage1_full30 checkpoint that create_model() loads.
# - A patience sweep at a FIXED seed is redundant: strict_repro makes each run
#   deterministic, so one --patience 999 run's history.json already tells you
#   what any patience in 4..9 would have selected. Instead we vary the SEED and
#   read patience off the curve afterwards.
# - --epochs also sets cosine T_max (CosineAnnealingLR(T_max=cfg["epochs"])), so
#   at EPOCHS=30 the LR decays over 30 epochs, matching the original stage-1
#   schedule shape rather than the compressed 5-epoch one.
# - strict_repro is left at its default (True): per-sample augmentation is a
#   fixed constant, not tied to --seed, so across these runs only weight init and
#   dataloader shuffle order vary. That isolates model-level stochasticity. Pass
#   --no-strict-repro below if you want webdataset shuffling + fuller divergence.
#
# Wall time: 44119 ran ~2 h/epoch on an A6000. 30 epochs ~= 60 h on amp48, more
# on ada24's slower Ada 4500. train_2_v2.sh hard-codes #SBATCH --time=15:00:00,
# which would kill the job at ~epoch 7 with no checkpoint to resume - so this
# script overrides it to 120 h. Every partition here is MaxTime=UNLIMITED.
#
# QOS caps (per-user, per-partition, concurrent): amp48 gpu=4, ada24 gpu=4.
# Five seeds over amp48,ada24 lets Slurm place ~4+1 rather than queue the fifth
# behind a 60 h run. Hardware is NOT homogeneous across that split (the earlier
# sweep saw ~0.001 mAP shift A6000 vs Ada) - keep that in mind comparing seeds.
PARTITIONS="${PARTITIONS:-amp48,ada24}"
CPUS="${CPUS:-8}"
MEM="${MEM:-64G}"
TIME="${TIME:-120:00:00}"
ACCUM="${ACCUM:-8}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-999}"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -eq 0 ]; then
    SEEDS=(42 86 123 456 789)
else
    SEEDS=("$@")
fi

echo "accum_steps=$ACCUM  epochs=$EPOCHS  patience=$PATIENCE  time=$TIME"
echo "seeds: ${SEEDS[*]}"
echo

for S in "${SEEDS[@]}"; do
    jid=$(sbatch --parsable \
        -J "isbi2026_train2v2_accum${ACCUM}_seed${S}_ep${EPOCHS}" \
        -p "$PARTITIONS" -c "$CPUS" --mem="$MEM" --time="$TIME" \
        "$SCRIPT_DIR/train_2_v2.sh" \
            --accum-steps "$ACCUM" --seed "$S" \
            --epochs "$EPOCHS" --patience "$PATIENCE")
    echo "accum=$ACCUM seed=$S -> job $jid | log /home/psytp7/logs/isbi2026_train2v2_accum${ACCUM}_seed${S}_ep${EPOCHS}_${jid}.out"
done
