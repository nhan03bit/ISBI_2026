#!/bin/bash
#SBATCH -J "isbi2026_train2v2_tuned"
#SBATCH -p amp48
#SBATCH -G 1
#SBATCH -c 12
#SBATCH --mem=100G
#SBATCH --time=45:00:00
#SBATCH -o /home/psytp7/logs/%x_%j.out

# NOTE: keep the #SBATCH directives above as one contiguous block with no
# comments interleaved - this cluster's submit filter rejects the job otherwise.
#
# Pinned to -p amp48 (colossus / juggernaut = NVIDIA RTX A6000 48GB). The
# ada24 node (petra) is an RTX 4500 Ada and is deliberately excluded here.
#
# Jobs 44648 (100G) and 45700 (150G, same config otherwise) were both
# OOM-killed mid-epoch-1 (iter 1823/12913 and 2720/12913) - never produced
# a checkpoint. Root cause found in train_2_v2.py: make_webdataset()'s
# per-worker .shuffle() buffer was sized for small thumbnails (4000
# samples), but this dataset's raw pre-decode images average ~6.65MB, so
# 4000 * 8 workers ~= 213GB just to fill the buffers - more --mem only
# bought a few thousand more iterations before hitting the same wall.
# Fixed at the source (SAMPLE_SHUFFLE_BUFSIZE = 1000, ~53GB total) rather
# than by keeping the memory request oversized here.
#
# Conservative single-epoch config expected to land mAP in [0.42, 0.49]
# vs the eb48/accum sweep's mAP ~0.41 plateau:
#   - strict_repro left on (default; no --no-strict-repro)
#   - peak LR 3e-5 (back down from the 1e-4 tuned attempt)
#   - 1 epoch, patience 1
#   - gamma_neg 2.0 (back down from 4.0)
#   - effective batch 48 (batch 8 x accum 6)

set -u

echo "=== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) started $(date) ==="
echo "args: $*"
nvidia-smi

source /home/psytp7/ISBI_2026/.venv/bin/activate
cd /home/psytp7/ISBI_2026/train

python train_2_v2.py \
    --stage1-ckpt checkpoint3/Model_20260119_062652/model_best.pth \
    --num-classes 30 \
    --lr 3e-5 \
    --embedding-dim 768 \
    --weight-decay 1e-2 \
    --memory-size 96 \
    --epochs 1 \
    --patience 1 \
    --min-delta 0.0 \
    --gamma-neg 2.0 \
    --gamma-pos 0.0 \
    --monitor mAP \
    --batch-size 8 \
    --accum-steps 6 \
    --num-workers 8 \
    --val-num-workers 2 \
    --log-every 500 \
    --margin 0.5 \
    --disc 0 \
    --seed 42 \
    "$@"
    # --train-shards-dir / --val-shards-dir intentionally omitted: cfg[...]
    # hardcodes /data/psytp7/wds_shards_{train,val}_raw in train_2_v2.py, so
    # these flags are dead and would be silently ignored anyway.
    # use_amp / strict_repro left at their argparse defaults (True / True) -
    # only --no-amp / --no-strict-repro would change them.

echo "=== Finished $(date) ==="
