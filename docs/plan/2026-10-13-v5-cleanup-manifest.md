# v5 cleanup manifest (2026-10-13)

`train/train_2_v5.py` loses `--arch moe`, `--moe-*`, `--expert-noise` and `--no-triplet`, so it can only train the Triplet Memory
(entropy gate) and Entropy + Markov prior systems. ASL only now trains exclusively on `train/train_2_v0.py`.
Plan: `docs/plan/2026-10-13-asl-v0-vs-entropy-v5-plan.md`.

## State frozen before the edit
- git HEAD: `6d9cf746b4f03903056d22a84441ea8aefe4b697`, local tag `v5-pre-moe-removal` (not pushed).
- The four files below were clean in git at that commit.

| file | sha256 |
|---|---|
| train/train_2_v5.py | 3715df415f7ae6459c40f7cb42132cf0cda61675212cfb5612470076f1aea092 |
| train/convnext_moe.py | a4d7b44535d3c182b6b86f603ca7f38d9b2d7c40e3b74748f3ee9cc4479926f4 |
| scripts/submit_stage2_moe.sh | e44c44aae925ef70cbf3b00af8beea6721271348d791cce9a6f1a5971c42fb0d |
| scripts/submit_stage2_v5_unified.sh | 4a9ac70218d9ebf28b3f04d068d62ec6a759c3564836bab22a47cb79dc0bd32a |

Reproduce an old run: `git checkout v5-pre-moe-removal -- train/train_2_v5.py scripts/submit_stage2_v5_unified.sh`.

## Run directories made with the removed flags (stay on disk, not reported)
- `train/checkpoint_MoE/{asl_img768_seed{42,1024}, moe_img768_seed{42,1024}, moeprior_img768_seed{42,1024}, moe_img{640,896,1024}_seed86}`
- `train/checkpoint_Triplet_3/unified768/a/seed_{123,456,789}` (ASL via `--no-triplet`; cancelled / incomplete 2026-10-13)
- `train/checkpoint_Triplet_3/unified768/e/seed_{42,123}` (ASL + prior, `--no-triplet`)
- Arms b (every image an anchor) and f (stronger triplet) in `unified768/{b,f}` were made with flags that stay in v5, but the arms are not part of the three-system study.

## Not removed
`train/convnext_moe.py`, `evaluate/evaluate_tta_v5.py` (they only read old checkpoints) and all existing probability dumps under `analysis/out/`.
