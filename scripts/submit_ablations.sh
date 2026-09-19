#!/bin/bash
# A4 ablations. Base config = the fixed paper recipe (converged Stage-1, 1 epoch,
# eff batch 48, binary entropy, shard-order class weights, mean anchor gate,
# pos_min_shared 1, easy mining, lambda_tri 0.1). Each axis varies ONE knob.
#
#   ./submit_ablations.sh anchor          # {max, mean, all, random}
#   ./submit_ablations.sh mining          # {easy, hard, semihard}
#   ./submit_ablations.sh memory          # {48, 96, 512, 2048}
#   ./submit_ablations.sh pos             # pos_min_shared {1, 2}
#   ./submit_ablations.sh lambda          # {0.05, 0.1, 0.2}
#   ./submit_ablations.sh entropy         # {softmax, binary}
#   ./submit_ablations.sh all             # everything
#   SEEDS="86 123" ./submit_ablations.sh anchor    # >1 seed for the key axes
#   ./submit_ablations.sh anchor --dry-run

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AXIS="${1:-all}"; DRY=0; [ "${2:-}" = "--dry-run" ] && DRY=1

PARTITIONS="${PARTITIONS:-amp48,ada24}"
SEEDS="${SEEDS:-86}"
CKPT="checkpoint3/Model_20260119_062652/model_best.pth"

BASE=(--lr 3e-5 --embedding-dim 768 --weight-decay 1e-2 --epochs 1 --patience 1
      --min-delta 0.0 --gamma-neg 2.0 --gamma-pos 0.0 --monitor mAP --batch-size 8
      --accum-steps 6 --num-workers 8 --val-num-workers 2 --log-every 500 --margin 0.5
      --disc 0 --entropy-mode binary --anchor-rule mean --pos-min-shared 1
      --mining easy --lambda-tri 0.1 --class-weight-order shard --memory-size 96)

fire() {  # <tag> <extra args...>
    local tag="$1"; shift
    for S in $SEEDS; do
        local out="checkpoint_Triplet_2/ablate/${tag}/seed_${S}/Model_run"
        local name="isbi2026_s2ab_${tag}_seed${S}"
        if [ "$DRY" = 1 ]; then echo "would submit $name : $*"; continue; fi
        local jid
        jid=$(sbatch --parsable -J "$name" -p "$PARTITIONS" -c 8 --mem=64G --time=12:00:00 \
            "$SCRIPT_DIR/train_2_v2.sh" --stage1-ckpt "$CKPT" --num-classes 30 \
            --seed "$S" --out-dir "$out" "${BASE[@]}" "$@")
        echo "$name -> job $jid"
    done
}

run_axis() {
case "$1" in
  anchor)  for r in max mean all random; do fire "anchor_${r}" --anchor-rule "$r"; done ;;
  mining)  for m in easy hard semihard;  do fire "mining_${m}" --mining "$m"; done ;;
  memory)  for k in 48 96 512 2048;      do fire "memory_${k}" --memory-size "$k"; done ;;
  pos)     for p in 1 2;                 do fire "pos_${p}"     --pos-min-shared "$p"; done ;;
  lambda)  for l in 0.05 0.1 0.2;        do fire "lambda_${l}"  --lambda-tri "$l"; done ;;
  entropy) for e in softmax binary;      do fire "entropy_${e}" --entropy-mode "$e"; done ;;
  *) echo "unknown axis: $1" >&2; exit 1 ;;
esac
}

# NOTE: later --flag wins in argparse, so "${BASE[@]}" then the override is fine.
if [ "$AXIS" = "all" ]; then
    for a in anchor mining memory pos lambda entropy; do run_axis "$a"; done
else
    run_axis "$AXIS"
fi

echo
echo "per-class eval after:  for d in train/checkpoint_Triplet_2/ablate/*/seed_*/Model_run; do sbatch scripts/eval_perclass.sh \$d/model_best.pth 512 res512; done"
