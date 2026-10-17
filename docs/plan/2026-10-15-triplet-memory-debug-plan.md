# Debug plan: is Triplet Memory in train_2_v5.py actually changing the model?

## Context
ASL only (train_2_v0.py) and Triplet Memory + Entropy (train_2_v5.py, λ=0.1) give the same mAP: seeded seed-42 flip-TTA 0.4605 vs 0.4625, 5-seed 0.4617 vs 0.4612. The user suspects the memory path is not used, or that a bug reduces v5 to ASL only.

**Already established (read-only):**
- The path executes: triplets on ~80% of steps, ~1.2 per step (history.json `triplet_stats`).
- The anchor is `last_h.mean(1)` (`ml_decoder.py:398`), which feeds the classifier, so gradient can reach the decoder and backbone.
- Shard class order and head classes are correct.
- The sampler (`utils_update.sample_triplets_v13`) works on synthetic data.

**Leading hypothesis (H1), not yet measured on real embeddings:** easy mining (`utils_update.py:121-122`) picks the closest positive and farthest negative. With margin 0.5 that hinge is almost always satisfied, so L_tri ≈ 0 and so is its gradient. The train-loss gap vs ASL suggests L_tri ≈ 0.003 per step, but that is an indirect estimate across different training trajectories.

**Goal:** measure, on the real data and checkpoints, how much gradient the triplet term actually sends into shared weights. Then decide whether there is a bug or a design choice that zeroes the term.

## Hypotheses to test
| # | Hypothesis | Test (diagnostic below) |
|---|---|---|
| H1 | Easy-mined hinge is inactive, so L_tri ≈ 0 | Fraction of active hinges (`d(a,p) − d(a,n) + 0.5 > 0`) and mean L_tri, easy vs hard vs semihard |
| H2 | Even when active, λ·∇L_tri is negligible next to ∇L_cls on shared params | ‖λ∇L_tri‖ / ‖∇L_cls‖ and their cosine, over the backbone, decoder and query parameters |
| H3 | Embeddings collapse (mean-pooling 30 queries gives near-identical vectors), so mining is degenerate | Mean pairwise cosine of `last_h.mean(1)`; spread of d(a,p) and d(a,n) |
| H4 | Memory bank is stale or mis-populated (labels/embeddings misaligned, fp16 cast, `filled` wrong) | Assert bank labels match the batches written; mem `filled` = 2048 after warm-up; distance from bank entries to recomputed embeddings |
| H5 | Entropy gate uses softmax entropy (the `calculate_entropy_from_logits` default) although the docstring calls binary the training default | Overlap of anchors chosen under softmax vs binary entropy. This affects anchor choice only, not whether the term is live. |

## Step 1: read-only GPU diagnostic (no training, ~30–45 min on one GPU)
New script **`analysis/triplet_probe.py`**. It only reads checkpoints and data, and reuses:
- `train/train_2_v5.py`: `create_model`, `make_webdataset`, `preprocess_batch`, `AsymmetricLoss` and the class-weight / head-class computation from `main()` (shard order; copy the logic, don't refactor v5)
- `train/utils_update.py`: `CrossBatchMemoryV2`, `sample_triplets_v13`, `calculate_entropy_from_logits`, `_mine_pair`

For each checkpoint:
- (a) the Stage-1 init `checkpoint3/Model_20260119_062652/model_best.pth`, the state at the start of training
- (b) seeded arm c `train/checkpoint_Triplet_3/unified768_seeded/c/seed_42/Model_run/model_best.pth` (job 52650)
- (c) seeded ASL `train/checkpoint_ASL/v0_768_seeded/seed_42/Model_run/model_best.pth`, as a reference for embedding geometry without triplet training

What the script does:
1. Build the model at 768 px in `train()` mode (as during training), batch 4, seeded train loader.
2. Warm up the memory for 512 batches with no grad, so the 2048-entry bank is full.
3. For the next 300 batches, run forward plus two separate backward passes (no optimizer step) to get:
   - L_cls, and L_tri under easy, hard and semihard mining (same anchors)
   - the fraction of active hinges, d(a,p), d(a,n)
   - grad-norm ratio and cosine on shared parameters
   - embedding pairwise cosine
   - softmax- vs binary-gate anchor overlap
   - the H4 consistency checks
4. Write `analysis/out/triplet_probe/<ckpt>.json` plus a small summary table.

Launcher **`scripts/submit_triplet_probe.sh`**: one sbatch job on `amp48,ada24` (exclude colossus), 4 h limit. This follows the pattern of the `evaluate_tta_v5.sh` jobs in `scripts/submit_stage2_v5_unified.sh`.

**Decision rule:**
- **H1 confirmed:** easy active-hinge fraction <10% and ‖λ∇L_tri‖/‖∇L_cls‖ <1% on (b). Root cause is the mining and weighting design, not a wiring bug.
- **Investigate further:** H4 shows misaligned or wrong bank entries. That is a real bug, so fix `CrossBatchMemoryV2` / the v5 update call.
- **Large but no effect:** hard mining gives a non-trivial ratio (e.g. >10%) but easy mining does not. Go to Step 2.

## Step 2 (only if Step 1 confirms H1): one training arm, ~12 h
Add arm `h` to `scripts/submit_stage2_v5_unified.sh`. It is arm c plus `--mining semihard`, using the existing flag (`train_2_v5.py:1309`); no code change. Seed 42, `OUT_TAG=seeded`, with the existing flip-TTA eval chained.

Compare against seeded ASL (0.4605) and c (0.4625). Treat the result as n=1 and say so; seeded noise is unmeasured. Submit only after the user OKs the GPU time.

## Optional logging improvement (proposed, not part of the debug)
In `train_one_epoch` (`train_2_v5.py:897-1009`), accumulate per-epoch mean L_tri and active-hinge fraction into `history["triplet_stats"]`. Future runs then report the term directly instead of having it inferred from the train-loss gap.

## Verification
- **Smoke first:** run the probe with `--batches 20 --warmup 20` in a short job to confirm it loads all three checkpoints and writes JSON.
- **Sanity checks on the output:**
  - Hard mining must give an active fraction ≥ easy.
  - L_tri recomputed from the logged distances must match `TripletMarginLoss`.
  - The bank labels check (H4) must pass for any trustworthy conclusion.
- **Write-up:** the findings go to `docs/plan/2026-10-15-triplet-memory-debug.md` (a study, so `docs/plan/` per policy) in the debug-report template. Update memory `stage2-triplet-embedding-miswired.md` or create a new memory if H1 is confirmed.
- **Untouched:** no change to `train_2_v5.py`, `utils_update.py` or `paper.tex` in Step 1.
