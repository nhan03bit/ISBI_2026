#!/bin/bash
#SBATCH -J "isbi2026_stage1"
#SBATCH -p amp48
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o /home/psytp7/logs/isbi2026_stage1_%j.out

set -euo pipefail

PROJECT_DIR="/home/psytp7/ISBI_2026"
cd "$PROJECT_DIR"

# --- Build: environment setup ---
source "$PROJECT_DIR/.venv/bin/activate"

# --- Run ---
# train.py has no CLI args: config (paths, batch size, epochs, etc.) is set
# in the cfg dict inside main(). Shard dirs are read relative to $PROJECT_DIR
# (./wds_shards_train_raw, ./wds_shards_val_raw), so we must run from here.
#
# To observe a submitted job:
#   squeue --me                                        # confirm it's running, get <jobid>
#   tail -f "$PROJECT_DIR/logs/latest.log"              # live training log (unbuffered)
#   tail -f /home/psytp7/logs/isbi2026_stage1_<jobid>.out   # raw Slurm stdout/stderr
python train.py
