# Why does every Stage-2 arm, ASL included, land at ~0.46 internal mAP? (2026-10-16)

**Question.** ASL only (`train_2_v0.py`) and every `train_2_v5.py` arm score about 0.46 internal-val mAP across many trainings. Is there a bug in `train_2_v5.py` that caps them all?

**Answer.**
- **No code bug caps the score.** Labels, decoding, normalisation and the loss all check out. v0 and v5 share the same image pipeline, so anything there affects both arms equally.
- **One real but small pipeline defect:** training resizes images with `cv2.INTER_LINEAR`, but validation and evaluation use `cv2.INTER_LANCZOS4`. Fixing it at eval time alone is worth +0.003 to +0.006 mAP.
- **The rest of the 0.46 is systematic, not variance.** A 5-seed ensemble adds only 0.0055. What holds the number down:
  - lateral and portable views score about 0.41
  - labels are per study, so each image is scored against findings that may only be visible in another view
  - the model overfits after epoch 2 (train-set mAP 0.71 against val 0.46, on a same-recipe model)
- **The leaderboard is a different distribution** (frontal only, radiologist-labelled, one image per study), so internal 0.46 is not the leaderboard number.

Related: the Triplet Memory debug report [2026-10-15-triplet-memory-debug.md](2026-10-15-triplet-memory-debug.md) explains why Entropy equals ASL. This document is about the level both arms share.

**Tools and outputs** (no training; nothing launched beyond eval):
- `analysis/resize_probe.py`, job 52921 (amp16; juggernaut was down), output in `analysis/out/resize_probe/`
- the other checks are offline on the saved flip-TTA dumps in `analysis/out/probs_v0_flip{,_seeded}/` and `probs_unified_flip_seeded/`
- PadChest CSV metadata columns only (Projection, MethodLabel, StudyID), and only for internal-val images

## Checks that rule out a bug
| # | Hypothesis | Test | Result |
|---|---|---|---|
| B1 | Labels misaligned with images | Eval dump labels against `val_fold_1.csv`; train shard 10 `cls` against `train_fold_1.csv`, matched by `fname` | **Eliminated:** 0 / 17,270 val and 0 / 3,305 train mismatches; folds share no filenames |
| B2 | 16-bit PNGs crushed by the 8-bit `IMREAD_COLOR` decode | Raw `IMREAD_UNCHANGED` against the decode, 30 train images | **Eliminated:** values use the full 0–65,519 range (2–4k grey levels); 8-bit keeps 230–256 levels, mean about 120 |
| B3 | Normalisation differs between train, val, eval or v0/v5 | Code read | **Eliminated:** ImageNet mean/std everywhere; v0 and v5 decode/augment/normalise identically (diffed) |
| B4 | Correlated minibatches from shard order | Shard sample order and lag-1 label Jaccard | **Minor:** shards are in CSV (filename = study) order, so study-mates are adjacent (lag-1 Jaccard 0.44 against 0.19 random). The 400-sample buffer breaks that up; not a cap |
| B5 | ASL implementation | Code read | **Eliminated:** standard (γ− 2, γ+ 0, clip 0.05, class weights on positives, shard order) |
| B6 | Run-to-run variance hides a higher mean | 5 unseeded ASL seeds (flip-TTA SWA) and their ensemble | **Eliminated:** seeds 0.4585–0.4656, ensemble 0.4672 (+0.0055 over the mean 0.4617) |

## What the 0.46 is made of (seeded ASL, flip-TTA SWA unless stated)
1. **View mix.** The internal val is 30% laterals.

   | Subset | n | ASL | Entropy c | Markov, history | Markov, withheld |
   |---|---|---|---|---|---|
   | All views | 17,270 | 0.4605 | 0.4625 | 0.4807 | 0.4619 |
   | Frontal (PA, AP, AP-horizontal) | 11,973 | 0.4803 | 0.4850 | 0.4994 | 0.4854 |
   | PA only | 9,780 | 0.4881 | 0.4949 | 0.4989 | 0.4935 |
   | Non-frontal | 5,297 | 0.4055 | 0.4011 | 0.4277 | 0.4002 |

   The arm ranking is the same on frontal images as on all views.
2. **Labels are per study (per report), and each image is scored alone.** Every image of a study carries identical labels (checked on all val studies), so a lateral is labelled with findings that may only be visible on the PA.
   - On the 8.7% of val images whose study has another val image, averaging the study's predictions raises mAP from 0.512 to 0.576.
   - Laterals gain the most (0.448 → 0.531), PA 0.523 → 0.548.
   - **This is not usable on the leaderboard,** which has one frontal image per study.
3. **Label source is not the cap.** On common classes, physician-labelled images score 0.462 and NLP-labelled images 0.470. On frontal images, physician-labelled ones score 0.496.
4. **Generalisation gap.** On the 103,281 training images, a 768 px model with the same recipe (job 49000, v3 λ 0.1, SWA, flip-TTA; eval job 49407) scores 0.713, against 0.46 on val. Val loss bottoms at epoch 2 in every arm. The recipe stops at 4 of 8 cosine epochs, with the LR still about 54% of peak. A fully annealed schedule at 768 px with this recipe was never run; the older short-horizon runs used 512 px or a continuation, so they are not comparable.

## Pipeline defect: the train/val resampling filters differ
`deterministic_random_resized_crop` (`train_2_v5.py:113,120`) downsizes the ~2,800–3,000 px originals with `cv2.INTER_LINEAR`, which does not antialias at a ~4× downscale. `deterministic_val_resize` (`train_2_v5.py:364`) and `evaluate/evaluate_tta.py:53` use `cv2.INTER_LANCZOS4`. The same pairing appears in `train_2_v0.py`–`v4.py`, the Stage-1 `train.py` and `evaluate/evaluate.py`.

**Eval-only test** (`resize_probe.py`, job 52921): one decode per val image, three resizes, SWA weights, no TTA, full val of 17,270 images. 95% CIs from a paired bootstrap over images, 300 resamples.

| Model | Lanczos (current) | Linear (the training filter) | Area (antialiased) |
|---|---|---|---|
| Seeded ASL | 0.4567 | 0.4609, +0.0042 [+0.0024, +0.0068], 28/30 classes up | 0.4630, +0.0063 [+0.0013, +0.0133], 19/30 up |
| Seeded Entropy c | 0.4590 | 0.4624, +0.0034 [+0.0020, +0.0053], 26/30 up | 0.4642, +0.0052 [+0.0016, +0.0106], 21/30 up |

- On frontal images only, area gains +0.0105 (ASL) and +0.0066 (c). mAUC and mECE improve slightly as well.
- The gain is real and costs nothing at eval time, but it is about 1% relative, so it does not explain the plateau.
- Training with `INTER_AREA` (antialiased in both training and eval) has not been tested.

## Status: recommendations 1–4 launched (2026-10-16, user: "continue as recommended")
**Code changes.** Defaults are unchanged: a CPU check confirmed the v0 train/val output is byte-identical to v5, and eval Lanczos output is identical to before.
- `evaluate/evaluate_tta.py` and `evaluate_tta_v5.py`: `--interp lanczos|linear|area`, default lanczos. Non-lanczos dumps get an `_<interp>` suffix, and the JSON records `interp`.
- `train/train_2_v0.py`: `--train-interp` (default linear) and `--val-interp` (default lanczos).
- `scripts/train_2_v0.sh` and `scripts/evaluate_tta_v5.sh`: abort if CUDA is unavailable (a dead colossus GPU otherwise falls back to CPU silently).
- `scripts/submit_stage2_v0.sh`: `TRAIN_ARGS`, `EVAL_INTERPS`; `EXCLUDE=` (empty) now allows colossus.
- New `scripts/submit_reeval_interp.sh` (area re-evaluation of every Table 1 run).
- New `analysis/eval_interp_table.py` (lanczos vs area, all-view and frontal, seeded table, 5-seed contrasts, training arms against the seeded ASL baseline).

**Jobs.**
- **Area re-evaluation**, jobs 52942–52966 (amp20/amp16, 16G):
  - seeded v0 / c / d-with / d-zero / p
  - the 5-seed unseeded Table 1 runs (v0, c, d-with, d-zero)
  - output in `analysis/out/probs_*_area/`
- **Training arms** (seeded ASL v0, seed 42; juggernaut down, so colossus is allowed):
  - 52967: `--train-interp area --val-interp area` → eval 52968 (area)
  - 52969: `--sched-epochs 4` → evals 52970 (lanczos) and 52971 (area)
- **Frontal-only 5-seed means** (Lanczos, existing dumps): ASL 0.4836, Entropy 0.4834, Markov with history 0.5020, withheld 0.4853. Entropy − ASL is −0.0002, and Markov with history − ASL +0.0184 [+0.0118, +0.0249]: the same conclusions as on all views.

## Recommendations (as written before launch)
1. **Eval filter.**
   - Switch the val/eval resize to `INTER_AREA`, or at least to `INTER_LINEAR` to match training, in `deterministic_val_resize` and `evaluate/evaluate_tta.py`.
   - Re-evaluate all Table 1 arms together, so the comparison stays like-for-like.
   - Check which resize the leaderboard submission used: the repo has no submission script; `evaluate.py` / `evaluate_tta.py` use Lanczos.
2. **Dev metric.** Report frontal-only internal mAP next to the all-view number. It matches the leaderboard's frontal-only test set, and lateral images should not drive model selection.
3. **One training arm to test consistent antialiasing.** Train and val with `INTER_AREA` inside the random-resized crop (about 12 h, seeded ASL v0; compare against seeded ASL 0.4605).
4. **Schedule.** One seeded 768 px run with `--sched-epochs 4`, so the LR anneals within the 4 trained epochs (about 12 h). Treat it as n = 1 until the seeded noise floor is known (same as the repeat-ASL item in the triplet report).
5. **Beyond `train_2_v5.py`.** The systematic part (views, study-level labels, overfitting after epoch 2) is set by the Stage-1 initialisation and the data, not by a Stage-2 code path. Larger gains would need a different starting point (for example CXR-specific pretraining) rather than another Stage-2 loss.
