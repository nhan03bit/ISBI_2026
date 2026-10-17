#!/bin/bash
#SBATCH -J isbi2026_memeval_5seed
#SBATCH -p cpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o /home/psytp7/logs/%x_%j.out

# NOTE: keep the #SBATCH directives above as one contiguous block with no comments interleaved (the cluster's
# submit filter rejects the job otherwise).
#
# Route 1 replication (docs/plan/2026-10-16-beat-asl-plan.md section 2c): memory_readout_eval.py over the 5 Table 1
# Entropy probes, then the pre-registered seed-level verdict (analysis/memory_readout_seeds.py). CPU only.
#   sbatch --dependency=afterok:<probe job ids> scripts/memory_eval_5seed.sh

set -u
echo "=== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) started $(date) ==="
source /home/psytp7/ISBI_2026/.venv/bin/activate
cd /home/psytp7/ISBI_2026
python -I analysis/memory_readout_eval.py --tags entropy_c_s42 entropy_c_s123 entropy_c_s456 entropy_c_s789 \
    entropy_c_s1024 --out analysis/out/memory_readout_eval_5seed.json \
    || { echo "memory_readout_eval FAILED" >&2; exit 1; }
python -I analysis/memory_readout_seeds.py || { echo "memory_readout_seeds FAILED" >&2; exit 1; }
echo "=== Finished $(date) ==="
