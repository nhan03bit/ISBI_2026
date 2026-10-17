#!/bin/bash
#SBATCH -J isbi2026_task1_download
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH -o /home/psytp7/logs/%x_%j.out

# NOTE: keep the #SBATCH directives above as one contiguous block with no comments interleaved (the cluster's
# submit filter rejects the job otherwise).
#
# Downloads the CXR-LT 2026 Task-1 evaluation images (Task-1.zip, 31.5 GB) from the shared "submission" Drive
# folder via rclone, verifies the checksum against Drive, lists the archive, refuses it if any member path is
# absolute or contains "..", then extracts into a new empty directory. The archive is third-party data: nothing
# inside it is executed. These are the leaderboard images: never join them to the PadChest CSV (it holds their
# labels). Prediction: scripts/predict_task1.sh.
#   sbatch scripts/download_task1_gdrive.sh

set -uo pipefail

RCLONE=/home/psytp7/.local/bin/rclone
FOLDER_ID="1413LT8vchVhHC7ubLa1xxvMedbU3xL0M"
SRC="gdrive,root_folder_id=${FOLDER_ID}:"
BASE=/data/psytp7/isbi_task1
ZIPDIR="$BASE/zip"
OUT="$BASE/extracted"

echo "=== Job $SLURM_JOB_NAME ($SLURM_JOB_ID) on $(hostname) started $(date) ==="
mkdir -p "$ZIPDIR"
if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "$OUT exists and is not empty; refusing to extract over it" >&2
    exit 2
fi

"$RCLONE" copy "${SRC}Task-1.zip" "$ZIPDIR" --transfers 1 --drive-chunk-size 128M \
    --stats 2m --stats-log-level NOTICE --log-file "/home/psytp7/logs/rclone_task1_${SLURM_JOB_ID}.log" \
    || { echo "rclone copy FAILED (see log)" >&2; exit 1; }
"$RCLONE" check "$SRC" "$ZIPDIR" --include "Task-1.zip" \
    --log-file "/home/psytp7/logs/rclone_check_task1_${SLURM_JOB_ID}.log" \
    || { echo "rclone check FAILED: download does not match Drive" >&2; exit 1; }
ls -l "$ZIPDIR/Task-1.zip"

unzip -l "$ZIPDIR/Task-1.zip" > "$BASE/Task-1.zip.listing.txt" || { echo "unzip -l FAILED" >&2; exit 1; }
head -5 "$BASE/Task-1.zip.listing.txt"; tail -3 "$BASE/Task-1.zip.listing.txt"
/home/psytp7/ISBI_2026/.venv/bin/python -I - "$ZIPDIR/Task-1.zip" <<'EOF' || { echo "archive has absolute or '..' member paths; not extracting" >&2; exit 1; }
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
bad = [n for n in names if n.startswith(("/", "\\")) or ".." in n.replace("\\", "/").split("/")]
print(f"members: {len(names)} | unsafe: {len(bad)}", bad[:5])
sys.exit(1 if bad else 0)
EOF

mkdir -p "$OUT"
unzip -q -n "$ZIPDIR/Task-1.zip" -d "$OUT" || { echo "unzip FAILED" >&2; exit 1; }
echo "extracted files: $(find "$OUT" -type f | wc -l)"
find "$OUT" -maxdepth 2 | head -20
find "$OUT" -type f ! -iname "*.png" | head -20
echo "=== Finished $(date) ==="
