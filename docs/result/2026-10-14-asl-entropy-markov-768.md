# ASL only vs Triplet Memory + Entropy vs Entropy + Markov prior, 768 px, 5 seeds — 2026-10-14

**Findings and next steps:** [docs/plan/2026-10-14-asl-entropy-markov-768-findings.md](../plan/2026-10-14-asl-entropy-markov-768-findings.md)
**Plan:** [docs/plan/2026-10-13-asl-v0-vs-entropy-v5-plan.md](../plan/2026-10-13-asl-v0-vs-entropy-v5-plan.md); parity validation: [docs/plan/2026-10-13-v0-v5-parity-validation.md](../plan/2026-10-13-v0-v5-parity-validation.md)
**Directory:** `train/checkpoint_ASL/v0_768/seed_S/Model_run/` (row A); `train/checkpoint_Triplet_3/unified768/{c,d}/seed_S/Model_run/` and `train/checkpoint_Triplet_3/prior/img768_qprior_seed{42,1024}/Model_run/` (rows C, D)
**Script / recipe:** row A `train/train_2_v0.py` only; rows C, D `train/train_2_v5.py` only (pre-cleanup v5, git tag `v5-pre-moe-removal`); never pooled
**Evaluation:** `analysis/unified_benchmark.py` -> `analysis/out/unified_benchmark.json`, `unified_table.tex`, `unified_macros.tex` (copied to `docs/paper/isbi_submit/generated/`). Internal validation split, n = 17,270 rows, SWA weights, flip TTA, 768 px. Not the leaderboard/test set.

## Slurm jobs
| Row | Training | Eval (flip TTA dumps) |
|---|---|---|
| A, ASL only | v0: 52402 (s42), 52404 (s123), 52408 (s456), 52410 (s789), 52406 (s1024), all COMPLETED 0:0, 12-13 h each | 52403, 52405, 52409, 52411, 52407, COMPLETED, `analysis/out/probs_v0_flip/eval_v0_s*.json` |
| C, Entropy | 51832 (s42), 51843 (s123), 51857 (s456), 51871 (s789), 51883 (s1024) | `evuni_c_*`, `analysis/out/probs_unified_flip/eval_c_s*.json` |
| D, Entropy + Markov | 49375 (s42), 51845 (s123), 51859 (s456), 51873 (s789), 49376 (s1024) | `evuni_d_*` with and zero (history available / withheld) |

Logs: `/home/psytp7/logs/isbi2026_v0_s*_524*.out`, `isbi2026_evv0_s*_524*.out`, `isbi2026_uni_{c,d}_s*_518*.out`, `isbi2026_v5prior_qprior_s{42,1024}_4937{5,6}.out`.

## Configuration (checked against the `args:` line and config dict of every log)
| Setting | A (v0, 5 logs) | C (5 logs) | D (5 logs) | Job 49375 (reference) |
|---|---|---|---|---|
| Stage-1 checkpoint | checkpoint3/Model_20260119_062652/model_best.pth | same | same | same |
| Image size / shuffle buf / rrc min | 768 / 400 / 0.9 | same | same | same |
| bs x accum | 4 x 12 (eff. 48) | same | same | same |
| Epochs / sched epochs / patience / warmup | 4 / 8 / 2 / 0.05 | same | same | same |
| lr / wd / drop-path | 1e-4 / 0.01 / 0.1 | same | same | same |
| ASL gamma_neg / gamma_pos | 2.0 / 0.0 | same | same | same |
| EMA / SWA | 0.999 / top-3 EMA epochs ([2,3,4]) | same | same | same |
| class-weight-order | shard (fixed); class-weight line md5-identical in the 8 logs compared (49375, 5 v0, C s42, D s123) | shard | shard | shard |
| metric-embed / memory / triplet-lambda | none (no triplet code) | query / 2048 / 0.1, `--anchor-rule mean` | same + prior (dropout 0.3, lr x10) | query / 2048 / 0.1, prior |
| Stage-1 load | missing=0 unexpected=0 | missing=0 | missing=2 (the two prior parameters) | missing=2 |
| LR schedule | warmup 860 / total 17,216 steps | same | same | same |

Differences found: (1) v0 has no triplet/memory/prior arguments (by design). (2) The 49375 and 49376 `args:` lines carry no `--anchor-rule`; the v5 default is `mean`, equal to the explicit `--anchor-rule mean` of the other C/D runs, so no difference. (3) 49375 `--out-dir` / argument order differ cosmetically. (4) v0 completes all 4 epochs with no early stopping (best epoch 3 for all five seeds); D s123 and s456 early-stopped at epoch 4 (best epoch 2), SWA still [2,3,4]. Nothing else differs.

Known-bug check: `--class-weight-order shard` in all 15 runs (post-fix; Hydropneumothorax weight 2.000). `--metric-embed query` in all C and D runs (post-fix); not applicable to A. The 384-vs-512 resolution confound does not apply (all 768 px). The 7.6% triplet-utilisation and softmax-entropy questions apply to C and D only.

## Per-run training summaries (validation, no TTA, from logs / history.json)
| Run | best-epoch EMA mAP (epoch) | SWA mAP | SWA mAUC | SWA mF1 | SWA mECE |
|---|---|---|---|---|---|
| A s42 | 0.4597 (3) | 0.4609 | 0.9079 | 0.4765 | 0.1337 |
| A s123 | 0.4558 (3) | 0.4574 | 0.9095 | 0.4664 | 0.1316 |
| A s456 | 0.4551 (3) | 0.4561 | 0.9101 | 0.4707 | 0.1324 |
| A s789 | 0.4586 (3) | 0.4586 | 0.9109 | 0.4709 | 0.1328 |
| A s1024 | 0.4489 (3) | 0.4510 | 0.9102 | 0.4598 | 0.1328 |
| C s42 / s123 / s456 / s789 / s1024 | 0.4554 / 0.4590 / 0.4521 / 0.4598 / 0.4582 (all epoch 3) | 0.4578 / 0.4609 / 0.4552 / 0.4604 / 0.4597 | | | |
| D s42 (49375) | 0.4793 (3) | 0.4807 | 0.9151 | 0.4863 | 0.1342 |
| D s123 / s456 / s789 / s1024 | 0.4749 (2) / 0.4753 (2) / 0.4754 (3) / 0.4760 (2) | 0.4778 / 0.4741 / 0.4769 / 0.4786 | | | |

Job 49375 check: per-epoch EMA values in its `history.json` are mAP [0.4587, 0.4781, 0.4793, 0.4735], mAUC [0.9121, 0.9147, 0.9125, 0.9107], mF1 [0.4596, 0.4898, 0.4898, 0.4848], mECE [0.1521, 0.1452, 0.1349, 0.1211], val_loss [0.0392, 0.0382, 0.0385, 0.0395]. The pasted line "0.4735 0.4848 0.1211 0.0395" is the **epoch-4** row (mAP, mF1, mECE, val_loss); 0.4848 is mF1, not mAUC (epoch-4 mAUC is 0.9107). Best is epoch 3 (0.4793), SWA 0.4807 (mAUC 0.9151).

## Results: unified table (flip TTA, SWA, internal val, n = 5 seeds, mean ± sd)
| Row (script) | n | mAP | mAUC | mF1 | mECE | head AP | medium AP | tail AP |
|---|---|---|---|---|---|---|---|---|
| ASL only (v0) | 5 | 0.4617 ± 0.0031 | 0.9110 ± 0.0011 | 0.4730 ± 0.0045 | 0.1325 ± 0.0007 | 0.7632 ± 0.0004 | 0.4620 ± 0.0007 | 0.3317 ± 0.0123 |
| Triplet Memory + Entropy (v5) | 5 | 0.4612 ± 0.0028 | 0.9114 ± 0.0007 | 0.4722 ± 0.0047 | 0.1349 ± 0.0005 | 0.7644 ± 0.0006 | 0.4621 ± 0.0004 | 0.3287 ± 0.0115 |
| + Markov prior, history available (v5) | 5 | 0.4805 ± 0.0027 | 0.9150 ± 0.0010 | 0.4930 ± 0.0033 | 0.1329 ± 0.0008 | 0.7714 ± 0.0008 | 0.4806 ± 0.0007 | 0.3553 ± 0.0116 |
| + Markov prior, history withheld (v5) | 5 | 0.4611 ± 0.0036 | 0.9115 ± 0.0005 | 0.4750 ± 0.0058 | 0.1355 ± 0.0008 | 0.7629 ± 0.0008 | 0.4596 ± 0.0005 | 0.3361 ± 0.0153 |

sd = sample sd (ddof 1) over 5 seeds. Head/medium/tail = 3 / 20 / 7 classes from `evaluate/class_groups.py` (head: Normal, cardiomegaly, pleural effusion; tail incl. Hydropneumothorax, Mass, Pneumothorax, Subcutaneous Emphysema, azygos lobe, calcified densities, pneumoperitoneo). The two D rows are the same 5 trained models scored with and without patient history. Independently recomputed from the raw probabilities (`ea.tm_metrics`, own sign-flip / t-CI code): every mean and sd equals `unified_table.tex`; per-seed ASL values equal `eval_v0_s*.json` (s42 0.4656 / 0.9093 / 0.4807 / 0.1335; s123 0.4614; s456 0.4590; s789 0.4641; s1024 0.4585 mAP).

Per-seed flip-TTA mAP:
| Seed | A | C | Dwith | Dzero |
|---|---|---|---|---|
| 42 | 0.4656 | 0.4605 | 0.4839 | 0.4624 |
| 123 | 0.4614 | 0.4653 | 0.4803 | 0.4603 |
| 456 | 0.4590 | 0.4577 | 0.4769 | 0.4566 |
| 789 | 0.4641 | 0.4621 | 0.4792 | 0.4599 |
| 1024 | 0.4585 | 0.4604 | 0.4820 | 0.4663 |

## Paired contrasts (x − y, by seed, n = 5; 95% t-CI; exact two-sided sign-flip p; wins = seeds with x > y)
mAP, with Holm over the five mAP p-values and the paired-t MDE (80% power):
| Contrast | mean diff | 95% CI | p | wins | Holm p | MDE |
|---|---|---|---|---|---|---|
| C − A | −0.0005 | [−0.0049, +0.0038] | 0.750 | 2/5 | 1.000 | 0.0058 |
| Dwith − A | +0.0187 | [+0.0149, +0.0225] | 0.0625 | 5/5 | 0.3125 | 0.0051 |
| Dzero − A | −0.0006 | [−0.0066, +0.0054] | 0.875 | 1/5 | 1.000 | 0.0080 |
| Dwith − C | +0.0193 | [+0.0151, +0.0234] | 0.0625 | 5/5 | 0.3125 | 0.0056 |
| Dzero − C | −0.0001 | [−0.0053, +0.0050] | 0.9375 | 2/5 | 1.000 | 0.0069 |

With n = 5 the smallest attainable two-sided exact sign-flip p is 2/32 = 0.0625, so no contrast can reach p < 0.05 and Holm over five gives at best 0.3125. The CI, wins and MDE carry the information. Also: Dwith − Dzero (same models) = +0.0194 mAP, 5/5 seeds (+0.0215, +0.0200, +0.0203, +0.0193, +0.0158).

Other metrics (mean diff [95% CI], wins; for mECE a "win" means higher = worse):
| Contrast | mAUC | mF1 | mECE | head | medium | tail |
|---|---|---|---|---|---|---|
| C − A | +0.0004 [−0.0017, +0.0025] 3/5 | −0.0008 [−0.0110, +0.0093] 3/5 | +0.0024 [+0.0012, +0.0035] 5/5 | +0.0012 [+0.0004, +0.0021] 5/5 | +0.0001 [−0.0005, +0.0007] 2/5 | −0.0030 [−0.0200, +0.0140] 2/5 |
| Dwith − A | +0.0041 [+0.0019, +0.0063] 5/5 | +0.0200 [+0.0124, +0.0276] 5/5 | +0.0004 [−0.0002, +0.0010] 4/5 | +0.0082 [+0.0067, +0.0096] 5/5 | +0.0186 [+0.0172, +0.0201] 5/5 | +0.0236 [+0.0099, +0.0373] 5/5 |
| Dzero − A | +0.0005 [−0.0011, +0.0021] 2/5 | +0.0020 [−0.0077, +0.0117] 3/5 | +0.0029 [+0.0023, +0.0035] 5/5 | −0.0003 [−0.0017, +0.0011] 2/5 | −0.0024 [−0.0037, −0.0012] 0/5 | +0.0044 [−0.0191, +0.0278] 1/5 |
| Dwith − C | +0.0037 [+0.0024, +0.0049] 5/5 | +0.0208 [+0.0169, +0.0247] 5/5 | −0.0020 [−0.0031, −0.0008] 0/5 | +0.0069 [+0.0060, +0.0079] 5/5 | +0.0185 [+0.0176, +0.0195] 5/5 | +0.0266 [+0.0078, +0.0454] 5/5 |
| Dzero − C | +0.0001 [−0.0007, +0.0010] 2/5 | +0.0028 [−0.0042, +0.0098] 2/5 | +0.0006 [−0.0006, +0.0017] 4/5 | −0.0015 [−0.0025, −0.0005] 0/5 | −0.0025 [−0.0033, −0.0018] 0/5 | +0.0074 [−0.0144, +0.0292] 3/5 |

(Tail wins in the C/Dzero rows: the tail group is 7 classes, sd across seeds 0.012-0.015, driven by Hydropneumothorax with 38 training positives.) The exact p values for these other-metric contrasts are in `unified_benchmark.json` / `unified_macros.tex`; all are multiples of 1/16 (0.0625 minimum).

## Same-seed rerun noise floor (job 49375/49376 vs reruns 49855/49856; identical args, only `--out-dir` differs)
| Seed | best EMA mAP orig | rerun | SWA mAP (no TTA) orig | rerun | flip-TTA single-model mAP, WITH, orig | rerun | WITH diff | ZERO orig | ZERO rerun |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.4793 | 0.4740 | 0.4807 | 0.4731 | 0.4839 | 0.4760 | −0.0079 | 0.4624 | 0.4560 |
| 1024 | 0.4760 | 0.4761 | 0.4786 | 0.4764 | 0.4820 | 0.4786 | −0.0034 | 0.4663 | 0.4622 |

(From `train/.../prior/*/history.json` and [docs/result/2026-10-05-v5-rerun-repro.md](2026-10-05-v5-rerun-repro.md); the table's D row uses the original runs.) Run-to-run noise at the same seed is 0.0001-0.008 mAP, as large as a seed-to-seed sd (0.003). Substituting the s42 rerun for the original would lower the D(with) mean by about 0.0016 (0.0079/5), a mean of about 0.479; the conclusion would not change.

## Comparison to baselines
- Stage-1 backbone (mAP 0.385 internal, 384 px era): not on the same recipe; not compared quantitatively.
- Published MoE (0.405 internal at 384 px, 0.4599 official test): different recipe and resolution, not comparable with this table.
- Paper 5-seed entropy triplet (0.414 ± 0.002, 512/384 px era): different recipe/resolution.
- Earlier same-recipe numbers: 49375 SWA no-TTA 0.4807; reruns SWA 0.4731/0.4764 (above).

## Caveats / known-bug flags
- Post-fix runs only (shard weights, `--metric-embed query`). C and D ran on the pre-cleanup v5 (tag `v5-pre-moe-removal`); the working-tree `train/train_2_v5.py` is modified and uncommitted, and `train/train_2_v0.py` is untracked.
- Residual script difference (stated per plan): ConvNeXt2 still computes, and discards, its legacy embedding output in v0 (shared `convnext.py`; no parameters, no effect on the loss). The webdataset sample-buffer shuffle is **unseeded** in both scripts (`.shuffle(400)` seeded from pid + time), so same-seed reruns differ (noise floor above). Parity of v0 and v5 was shown at `--shuffle-buf 1` ([parity report](../plan/2026-10-13-v0-v5-parity-validation.md)).
- Val split is not patient-disjoint (77% of val images share a patient with train, 30% lateral); `has_prior` is 0.334 on val. Dwith is leakage-prone and not test-like; Dzero is the test-like row.
- Eval logs print `libpng error: PNG input buffer is incomplete` (also in the C/D eval logs); a decoder warning that appears in every eval log checked (v0 and v5), so it does not differ between rows; the dumps hold 17,270 rows while the prior-file table in the training log counts 17,274 val images (not investigated here).
- Only one split; absolute numbers likely optimistic. Internal val, not leaderboard.

## Next step
See the findings file. No per-class table from a fresh `eval_result*.json` exists; the head/medium/tail columns are computed from the flip-TTA dumps.
