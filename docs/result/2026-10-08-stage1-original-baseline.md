# Stage-1 original baseline ("without entropy") — 2026-10-08

**Directory:** `train/checkpoint3/Model_20260119_062652/` (original), `checkpoint/stage1_full30/` and `train/checkpoint3/stage1_full30/` (Aug-2026 retrain), `checkpoint/stage1_buggy_posenc_20260716/`, `checkpoint/stage1_v4_22k768dp/`
**Script / recipe:** `/home/psytp7/ISBI_2026/train.py` (ConvNeXt2 + ML-Decoder, Asymmetric Loss only; no triplet, no MoE, no class weights)
**Slurm job(s):** 41730, 42717 (retrain), 34342 / 34546 / 34709 (July runs), 49043 (v4 Stage 1), 47073 (512 px eval of the original checkpoint). Logs in `/home/psytp7/logs/<name>_<jobid>.out`; the same run logs (without progress bars) are in `/home/psytp7/ISBI_2026/logs/train_*.log`.
**Interpretation / comparison with Stage 2:** [docs/plan/2026-10-08-entropy-vs-moe-baseline.md](../plan/2026-10-08-entropy-vs-moe-baseline.md)

All numbers below were read from `history.json`, `eval_result.stage1.json` or the run logs on 2026-10-08. Everything is internal validation (the 17,270-image fold), never leaderboard. Metrics are the Trainer's macro mAP / mAUC / mF1 / mECE (mECE: higher is worse).

## TL;DR
- The paper's Stage-1 row (0.385 / 0.395 / 0.875 / 0.134, 384 px, cited to b12) is exactly `train/checkpoint3/Model_20260119_062652/history.json`, best epoch 6: 0.3849 / 0.3946 / 0.8746 / 0.1338. It is a **training-time** validation number from the January 2026 run, at 384 px.
- The same checkpoint scored by `evaluate/evaluate.py` at **512 px** gives 0.3688 / 0.3689 / 0.8643 / 0.1408 (job 47073). That is the number the repo can actually re-measure. No 384 px re-evaluation of the checkpoint exists, so the 0.016 mAP gap is not yet decomposed into resolution vs. anything else.
- The Aug-2026 retrain of `train.py` (jobs 41730, 42717) reaches only 0.3771 at epoch 7. It does not reproduce 0.3849. It is the "under-trained" Stage 1 of the paper's Table 2.
- The right "original / without entropy" baseline is `Model_20260119_062652/model_best.pth`: it is the checkpoint every Stage-2 row is initialised from.

## Configuration
| Setting | Original `Model_20260119_062652` | Retrain `stage1_full30` (41730 / 42717) |
|---|---|---|
| Date | run dir stamp 2026-01-19 06:26:52; files dated 2026-01-20 | 2026-08-25 (41730), 2026-08-26 to 08-29 (42717) |
| Code | **not preserved** (first `train.py` in git is 2026-07-13). No job log or `args:` line survives (oldest log in `/home/psytp7/logs` is 2026-07-05) | `train.py` after the 2026-07-19 patience change, root `convnext.py` as of 2026-08-25 10:29 (MoE plumbing removed) |
| Image size | 384 (b12: "384x384 during training and evaluation") | 384 (`SIZE = 384`, log has no `img_size` line; CLI absent at the time) |
| Optimizer / LR | b12: "Adam, initial LR 4e-5, wd 0.01". `history.json` `lr` list is exactly 3.9e-5 x cosine, T_max 30, stepped per epoch (3.8893e-05, 3.8574e-05, 3.8046e-05, 3.7314e-05, 3.6387e-05, 3.5276e-05), so the peak LR was **3.9e-5**, not 4e-5 | AdamW, lr 4e-5, wd 1e-2, `CosineAnnealingLR(T_max=30)` per epoch |
| Loss | ASL (b12). Class weights: none known | ASL gamma_neg 2.0, gamma_pos 0.0, clip 0.05, no class weights |
| Batch / AMP | unknown | 32, bf16 |
| Init | ImageNet (b12). The much better epoch-1 mAP (0.3603 vs 0.2851) hints at a different init or recipe, not verifiable | ImageNet-1k, drop-path 0, seed 42 |
| Epochs | b12: "trained end-to-end for 10 epochs". `history.json` holds 6 rows with best = last (see caveat) | 41730: patience 3, early-stopped at epoch 10. 42717: patience 30, ran all 30 epochs |
| Stage-1 bugs | class-weight-order bug and triplet-embedding bug do not apply (no Stage-2 code involved) | same |

## Results
### Original checkpoint, per epoch (`history.json`, 384 px, training-time validation)
| Epoch | train_loss | val_loss | mAP | mAUC | mF1 | mECE |
|---|---|---|---|---|---|---|
| 1 | 0.0539 | 0.0548 | 0.3603 | 0.8767 | 0.3545 | 0.1832 |
| 2 | 0.0517 | 0.0548 | 0.3754 | 0.8803 | 0.3680 | 0.1811 |
| 3 | 0.0499 | 0.0525 | 0.3826 | 0.8833 | 0.3832 | 0.1693 |
| 4 | 0.0481 | 0.0531 | 0.3831 | 0.8834 | 0.3905 | 0.1547 |
| 5 | 0.0461 | 0.0564 | 0.3825 | 0.8757 | 0.3856 | 0.1517 |
| **6 (best)** | 0.0437 | 0.0571 | **0.3849** | **0.8746** | **0.3946** | **0.1338** |

`best_epoch` 6, `best_metric` 0.38490042.

### All Stage-1 runs
| Run (job) | Res | Best epoch | Best mAP | mAUC / mF1 / mECE at best | Final epoch (mAP / mAUC / mF1 / mECE) | Status |
|---|---|---|---|---|---|---|
| **Original** `Model_20260119_062652`, history.json (Jan 2026) | 384 | 6 (last row) | **0.3849** | 0.8746 / 0.3946 / 0.1338 | not recorded | log lost |
| **Original, re-evaluated** `eval_result.stage1.json` (47073, 2026-09-10) | **512** | n/a | **0.3688** | 0.8643 / 0.3689 / 0.1408 | n/a | COMPLETED, 17,270 images |
| Retrain 42717 `stage1_full30` | 384 | 7 | 0.3771 | 0.8793 / 0.3760 / 0.1476 | ep 30: 0.3740 / 0.8262 / 0.4054 / 0.0364 | COMPLETED, 30/30 |
| Retrain 41730 `stage1` | 384 | 7 | 0.3771 | 0.8793 / 0.3760 / 0.1476 | ep 10: 0.3691 / 0.8595 / 0.3728 / 0.1095 (early stop, patience 3) | COMPLETED |
| July 34546 `stage1_buggy_posenc_20260716` | 384 | 8 | 0.2980 | 0.8794 / 0.3065 / 0.1655 | ep 11: 0.2956 / 0.8725 / 0.3328 / 0.1265 (early stop) | COMPLETED |
| July 34342 (log `train_20260713_163906.log`) | 384 | 8 | 0.2980 | same numbers as 34546 through epoch 10 | ep 10: 0.2964 / 0.8738 / 0.3234 / 0.1401 | TIMEOUT after 24 h in epoch 11 |
| July 34709 (log `train_20260719_000809.log`) | 384 | 25 | 0.3401 | not read per epoch; ep-30 0.8593 / 0.3991 / 0.0573 | ep 30: 0.3393 / 0.8593 / 0.3991 / 0.0573 | COMPLETED, 30/30 |
| v4 `stage1_v4_22k768dp` (49043) | 768 | 15 | 0.4140 | 0.8813 / 0.4388 / 0.1003 | ep 15 = best | COMPLETED; 22k init, dp 0.1, wd 0.03, bs 8, 15 ep / patience 5 (`args:` line). Later variant, not the original |
| Failed or cancelled launches 34335, 34338, 34339, 41677 | n/a | n/a | n/a | n/a | n/a | FAILED / CANCELLED, no result |

The 41730 and 42717 trajectories agree through epoch 10 to the printed precision (same seed; mECE at epoch 10 is 0.1095 vs 0.1096), and the `history.json` copies in `checkpoint/stage1`, `checkpoint/stage1_full30` and `train/checkpoint3/stage1_full30` are the same 7 rows.

### Retrain 42717 per epoch (Slurm log, all 30 epochs; `history.json` stops at epoch 7)
| Epoch | mAP | mAUC | mF1 | mECE |
|---|---|---|---|---|
| 1 | 0.2851 | 0.8426 | 0.2753 | 0.1817 |
| 3 | 0.3492 | 0.8788 | 0.3448 | 0.1853 |
| 5 | 0.3730 | 0.8854 | 0.3628 | 0.1718 |
| 6 | 0.3767 | 0.8807 | 0.3696 | 0.1635 |
| **7 (best)** | **0.3771** | 0.8793 | 0.3760 | 0.1476 |
| 10 | 0.3691 | 0.8595 | 0.3728 | 0.1096 |
| 15 | 0.3630 | 0.8400 | 0.3882 | 0.0649 |
| 20 | 0.3624 | 0.8301 | 0.3890 | 0.0437 |
| 25 | 0.3644 | 0.8209 | 0.3943 | 0.0390 |
| 30 | 0.3740 | 0.8262 | 0.4054 | 0.0364 |

mAP peaks at epoch 7 and then overfits (val_loss 0.0545 at epoch 7, 0.0877 at epoch 30; train_loss 0.0113 at epoch 30). mECE falls steadily with overfitting while mAUC falls.

### Per-class groups, original checkpoint at 512 px (`eval_result.stage1.json`, groups from `evaluate/class_groups.py`)
| Group | mAP |
|---|---|
| head (3 classes) | 0.7051 |
| medium (20) | 0.3765 |
| tail (7) | 0.2028 |
| extreme tail (6) | 0.2298 |
| category average | 0.4281 |

No per-class evaluation of the original checkpoint exists at 384 px.

## Comparison to baselines
- vs paper Table 1 Stage 1 (0.385 / 0.395 / 0.875 / 0.134): matches `history.json` best epoch to the printed precision. It does not match the 512 px re-evaluation (0.3688 / 0.3689 / 0.8643 / 0.1408).
- vs Aug-2026 retrain (0.3771): the original is +0.0078 mAP higher and has a different learning curve (epoch-1 mAP 0.3603 vs 0.2851; epoch-1 train_loss 0.0539 vs 0.0676).
- vs MoE / entropy Stage 2 and the delta table: see the plan document linked above.

## Caveats / known-bug flags
- **`history.json` is only written when the monitored metric improves** (`Trainer._save_best` in `train.py`). It therefore ends at the best epoch and does not show later epochs. The original's "6 epochs, best = last" is consistent with b12's "10 epochs" (epochs 7-10 not improving), but that is an inference. The same truncation applies to `stage1_full30` (7 rows for a 30-epoch run). For final-epoch numbers use the Slurm log.
- **The original run's code, args and Slurm log are not in the repo.** Recipe values above are from `history.json` (`lr`), b12, and the current `train.py` defaults. The peak LR reconstruction (3.9e-5, T_max 30) is exact to 5 significant digits, but it differs from b12's stated 4e-5.
- **The original's validation split is not recorded.** b12 says ~103,000 / ~17,000. The re-evaluation and all Stage-2 runs use the 17,270-image fold (`/data/psytp7/wds_shards_val_raw`). It is plausible but unverified that 0.3849 was measured on the identical fold.
- **Best epoch is selected on the scoring fold** for every row (0.3849 and 0.3771 are max-over-epochs of validation mAP). Small optimism, not quantified here.
- **Val split is not patient-disjoint** (77% of val images share a patient with train, 53% share a StudyID, 30% lateral; project memory `data-split-patient-leakage`). Applies equally to Stage 1 and Stage 2.
- **Positional-encoding code state is unclear.** `convnext.py` switched `PositionalEncoding2D` to `PositionalEncodingPermute2D` in commit 0eccffe (2026-07-19, "fix: legacy positional encoding class"), and the 2026-08-25 working copy and HEAD use `PositionalEncoding2D` again with the MoE plumbing removed. In the installed `positional-encodings` 6.0.4 the two classes give different outputs for a (B,C,H,W) tensor (checked on a toy tensor). The July runs (0.298-0.340) are labelled "buggy_posenc" by the directory name, and the August runs with a different file reach 0.377. I did not verify which code difference causes the gap, nor which class the January run used. `train/convnext.py` (used by Stage 2 and `evaluate.py`) uses `PositionalEncoding2D`. If the January run used the other class, both the 512 px re-evaluation and every Stage-2 initialisation inherit that mismatch. This is untested.
- The buggy-posenc runs (34342, 34546, 34709) are **not** candidates for the baseline: 0.298 and 0.340.
- Stage-1 `train.py` has no class weights and no triplet term, so the class-weight-order and metric-embedding bugs (Stage-2 `train_2_v2.py`) do not touch any run in this report.

## Notable
- The paper's 0.385 is a training-time, 384 px number whose code and log are lost. The repo's own measurement of the same checkpoint at 512 px is 0.3688. Resolution-matched Stage-1 numbers differ by 0.016 mAP, and 0.385 is not reproducible by retraining (0.3771).
- Stage-1 peak LR in the original run was 3.9e-5 (from `history.json` `lr`), not the 4e-5 stated in b12 and in `train.py`.

## Next step
- Run the 384 px evaluation of the original checkpoint that `scripts/submit_eval_perclass.sh` was designed to produce but that was never submitted (no `res384` job in `sacct`): `sbatch scripts/eval_perclass.sh train/checkpoint3/Model_20260119_062652/model_best.pth 384 res384` (about 35 min). If it returns about 0.385, the 0.016 gap is resolution; if it returns about 0.369, the January training-time number came from a different fold or code state.
- See the plan document for the Stage-2 controls.
