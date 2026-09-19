#!/bin/bash
# Submit per-class / head-medium-tail evaluation jobs for the A1 + A2 tables.
#
# Produces, next to each checkpoint:
#   eval_result.<tag>.json   (macro + group_mAP + per_class AP/AUC + val_pos)
#   preds.<tag>.npz          (probs [N,30], labels [N,30], class_names)
#
# A2 resolution control: the converged Stage-1 checkpoint is evaluated at BOTH
# 384 (the [b12] number) and 512 (what the proposed method uses).
#
#   ./submit_eval_perclass.sh              # submit everything
#   ./submit_eval_perclass.sh --dry-run    # just print what would be submitted

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

S1_CONV="train/checkpoint3/Model_20260119_062652/model_best.pth"
S1_UNDER="train/checkpoint3/stage1_full30/model_best.pth"

# proposed Stage-2, converged Stage-1 lineage (Table 1)
PROP_CONV=(
    train/checkpoint_Triplet_2/accum_6/seed_86/Model_20260827_095451/model_best.pth
    train/checkpoint_Triplet_2/accum_6/seed_123/Model_20260827_095859/model_best.pth
    train/checkpoint_Triplet_2/accum_6/seed_456/Model_20260827_100901/model_best.pth
    train/checkpoint_Triplet_2/accum_6/seed_789/Model_20260827_101532/model_best.pth
    train/checkpoint_Triplet_2/accum_6/seed_1024/Model_20260827_101922/model_best.pth
)
# proposed Stage-2, under-trained Stage-1 lineage (Table 2)
PROP_UNDER=(
    train/checkpoint_Triplet_2/accum_6/seed_86/Model_20260829_055214/model_best.pth
    train/checkpoint_Triplet_2/accum_6/seed_123/Model_20260829_055214/model_best.pth
    train/checkpoint_Triplet_2/accum_6/seed_456/Model_20260829_055214/model_best.pth
    train/checkpoint_Triplet_2/accum_6/seed_789/Model_20260829_055214/model_best.pth
    train/checkpoint_Triplet_2/accum_6/seed_1024/Model_20260829_072211/model_best.pth
)

submit() {  # <ckpt> <img> <tag> <jobname>
    local ckpt="$1" img="$2" tag="$3" name="$4"
    if [ ! -f "$ckpt" ]; then echo "MISSING: $ckpt" >&2; return 1; fi
    if [ "$DRY" = 1 ]; then
        echo "would submit: $name  |  $ckpt  @${img}  tag=$tag"
        return 0
    fi
    local jid
    jid=$(sbatch --parsable -J "$name" "$SCRIPT_DIR/eval_perclass.sh" "$ckpt" "$img" "$tag")
    echo "$name -> job $jid"
}

# --- A2: resolution control on the converged Stage-1 ---
submit "$S1_CONV"  384 res384 "isbi2026_eval_s1conv_res384"
submit "$S1_CONV"  512 res512 "isbi2026_eval_s1conv_res512"

# --- Stage-1 under-trained (512 to match proposed) ---
submit "$S1_UNDER" 512 res512 "isbi2026_eval_s1under_res512"

# --- A1: proposed, both lineages, at 512 ---
i=0
for c in "${PROP_CONV[@]}";  do s=$(echo "$c" | grep -oP 'seed_\K[0-9]+'); submit "$c" 512 res512 "isbi2026_eval_propconv_seed${s}"; i=$((i+1)); done
for c in "${PROP_UNDER[@]}"; do s=$(echo "$c" | grep -oP 'seed_\K[0-9]+'); submit "$c" 512 res512 "isbi2026_eval_propunder_seed${s}"; done

echo
echo "when done, build the tables:"
echo "  python analysis/per_class_report.py \\"
echo "    --cond stage1_conv_384 train/checkpoint3/Model_20260119_062652/preds.res384.npz \\"
echo "    --cond stage1_conv_512 train/checkpoint3/Model_20260119_062652/preds.res512.npz \\"
echo "    --cond proposed_conv   train/checkpoint_Triplet_2/accum_6/seed_*/Model_20260827_*/preds.res512.npz \\"
echo "    --baseline stage1_conv_512 --out analysis/out/perclass_conv"
