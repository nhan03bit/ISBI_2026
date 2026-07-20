#!/bin/bash
#SBATCH -J "convert_dataset"
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH -o /home/psytp7/logs/convert_dataset_%j.out

# convert_dataset.py is CPU-only (image I/O + tensor packing, no GPU ops),
# so this targets the "cpu" partition (Pietro) rather than a GPU node.
# Leaving out --nodelist lets Slurm pick whichever node in the partition
# is free instead of hardcoding one.

set -e

source ~/envs/padchest_wds/bin/activate
cd /data/psytp7

echo "=== Starting convert_dataset.py at $(date) on $(hostname) ==="
python convert_dataset.py
echo "=== Finished at $(date) ==="
