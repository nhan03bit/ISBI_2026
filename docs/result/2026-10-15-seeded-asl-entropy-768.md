# Seeded rerun (seed 42, 768 px): ASL only vs Triplet Memory + Entropy vs + Markov prior, plus lambda-1.0 probe — 2026-10-15

**Findings and next steps:** [docs/plan/2026-10-15-seeded-asl-vs-entropy-findings.md](../plan/2026-10-15-seeded-asl-vs-entropy-findings.md)
**Previous (unseeded, 5 seeds) experiment:** [docs/result/2026-10-14-asl-entropy-markov-768.md](2026-10-14-asl-entropy-markov-768.md)
**Directories:** `train/checkpoint_ASL/v0_768_seeded/seed_42/Model_run/` (A); `train/checkpoint_Triplet_3/unified768_seeded/{c,d,p}/seed_42/Model_run/`
**Script / recipe:** A = `train/train_2_v0.py`; C, D, P = `train/train_2_v5.py` (working tree, MoE code removed, `SeededSampleShuffle` present in both scripts). Never pooled across scripts.
**Evaluation:** `evaluate_tta_v5.sh` dumps in `analysis/out/probs_v0_flip_seeded/` and `analysis/out/probs_unified_flip_seeded/` (SWA weights, flip TTA, 768 px, internal validation split, n = 17,270 rows, same `keys.txt` / `labels.npy` as the unseeded dumps; row order verified identical). Head/medium/tail columns and per-class AP are recomputed by me from the dumps with `analysis/entropy_ablation.tm_metrics` and `evaluate/class_groups.py` (3 / 20 / 7 classes); mAP/mAUC/mF1/mECE equal the `eval_*.json` files to 4 decimals. Not the leaderboard or test set.

## Slurm jobs (sacct)
| Run | Training | Elapsed | Eval | State |
|---|---|---|---|---|
| A ASL only | 52648 `isbi2026_v0_seeded_s42` | 12:47:05 | 52649 (36:24) | COMPLETED 0:0 |
| C Entropy (lambda 0.1) | 52650 | 12:45:36 | 52651 (36:28) | COMPLETED 0:0 |
| D Entropy + Markov prior | 52652 | 12:42:38 | 52653 with history (36:37), 52654 withheld (36:41) | COMPLETED 0:0 |
| P probe (lambda 1.0) | 52655 | 11:48:22 | 52656 | COMPLETED 0:0 (eval finished 12:19; added 2026-10-15 after this report was first written) |

Logs: `/home/psytp7/logs/isbi2026_{v0,uni}_seeded_*_5264{8}..5265{5}.out`, `isbi2026_ev{v0,uni}_seeded_*_5264{9}..5265{6}.out`.

## Configuration (from the `args:` line of each log)
| Setting | A | C | D | P |
|---|---|---|---|---|
| Stage-1 checkpoint | `checkpoint3/Model_20260119_062652/model_best.pth` (missing=0 unexpected=0; D missing=2 = the two prior parameters) | same | same | same |
| Image size / shuffle buf | 768 / 400 (**seeded sample shuffle**) | same | same | same |
| Seed | 42 | 42 | 42 | 42 |
| lr / bs x accum (eff.) | 1e-4 / 4 x 12 (48) | same | same | same |
| Epochs / sched epochs / patience / drop-path | 4 / 8 / 2 / 0.1 | same | same | same |
| SWA | `--swa-last-k 3` ([2,3,4]) | same | same | same |
| class-weight-order | shard (fixed; Hydropneumothorax 2.000) | shard | shard | shard |
| metric-embed / memory / triplet-lambda / anchor-rule | none | query / 2048 / 0.1 / mean | same + prior file (dropout 0.3, lr x10) | query / 2048 / **1.0** / mean |
| Epochs run / best epoch | 4 / 3 | 4 / 4 | 4 (early stop flagged at epoch 4) / 2 | 4 / 4 |

Known-bug check: shard class weights in all four logs (post-fix); `--metric-embed query` in C, D, P (post-fix; n/a for A). The shuffle seeding fix is what distinguishes this experiment from the unseeded one (`SeededSampleShuffle`, seeded from `--seed`, DataLoader worker id and epoch, in both `train_2_v0.py` and `train_2_v5.py`); the working-tree `train_2_v5.py` and `train_2_v0.py` are uncommitted. Mining is `easy` and the anchor rule `mean` in the `triplet_stats` of C, D and P (this is the known "easy, not hard" confound; the paper's max-entropy anchor is not what these runs use).

## Table 1 candidate: seeded rerun, seed 42, flip TTA, SWA weights, internal validation, n = 1 run per row
| Row | mAP | mAUC | mF1 | mECE | head AP | medium AP | tail AP |
|---|---|---|---|---|---|---|---|
| ASL only (v0) | 0.4605 | 0.9103 | 0.4715 | 0.1327 | 0.7627 | 0.4620 | 0.3269 |
| Triplet Memory + Entropy (v5) | 0.4625 | 0.9104 | 0.4753 | 0.1356 | 0.7628 | 0.4617 | 0.3362 |
| + Markov prior, history available | 0.4807 | 0.9120 | 0.4931 | 0.1338 | 0.7701 | 0.4819 | 0.3532 |
| + Markov prior, history withheld | 0.4619 | 0.9089 | 0.4727 | 0.1362 | 0.7617 | 0.4603 | 0.3378 |

Probe P (not a Table 1 row), same scoring: mAP 0.4557, mAUC 0.9116, mF1 0.4648, mECE **0.1542**, head 0.7619, medium 0.4583, tail 0.3171 (`eval_p_s42.json`, job 52656; group APs recomputed with the same helpers, which reproduce the A and C rows above exactly). P − A: mAP −0.0048, mECE +0.0215; P − C: mAP −0.0068, mECE +0.0186.

The two Markov rows are one trained model (job 52652) scored with and without history. Sources: `analysis/out/probs_*_seeded/eval_*.json` for the four aggregate metrics; group APs recomputed from the dumps.

## Same table, previous UNSEEDED seed-42 runs (same scoring; this is the pasted 768 px Table 1, reproduced to 3 decimals)
| Row | mAP | mAUC | mF1 | mECE | head | medium | tail |
|---|---|---|---|---|---|---|---|
| ASL only | 0.4656 | 0.9093 | 0.4807 | 0.1335 | 0.7628 | 0.4623 | 0.3476 |
| Entropy | 0.4605 | 0.9124 | 0.4670 | 0.1346 | 0.7645 | 0.4621 | 0.3257 |
| Markov, history | 0.4839 | 0.9162 | 0.4926 | 0.1341 | 0.7723 | 0.4799 | 0.3716 |
| Markov, withheld | 0.4624 | 0.9121 | 0.4742 | 0.1366 | 0.7638 | 0.4591 | 0.3423 |

(Unseeded runs: `train/checkpoint_ASL/v0_768/seed_42`, `train/checkpoint_Triplet_3/unified768/c/seed_42`, `train/checkpoint_Triplet_3/prior/img768_qprior_seed42` = job 49375.)

## Entropy minus ASL (C − A), seed 42, per metric
| Metric | Seeded C − A | Unseeded C − A | 5-seed mean C − A (unseeded) [95% CI] | 5-seed sd of paired diff |
|---|---|---|---|---|
| mAP | **+0.0020** | −0.0051 | −0.0005 [−0.0049, +0.0038] | 0.0035 |
| mAUC | +0.0001 | +0.0031 | +0.0004 [−0.0017, +0.0025] | 0.0017 |
| mF1 | +0.0038 | −0.0137 | −0.0008 [−0.0110, +0.0093] | 0.0082 |
| mECE (higher = worse) | **+0.0029** | +0.0011 | +0.0024 [+0.0012, +0.0035], 5/5 seeds | 0.0009 |
| head AP | +0.0001 | +0.0017 | +0.0012 [+0.0004, +0.0021] | 0.0007 |
| medium AP | −0.0003 | −0.0002 | +0.0001 [−0.0005, +0.0007] | 0.0005 |
| tail AP | +0.0093 | −0.0219 | −0.0030 [−0.0200, +0.0140] | 0.0137 |

Other seeded contrasts vs ASL (seed 42): D(with) − A = mAP +0.0201, mAUC +0.0017, mF1 +0.0216, mECE +0.0011, head +0.0074, medium +0.0199, tail +0.0263. D(zero) − A = mAP +0.0013, mAUC −0.0014, mF1 +0.0011, mECE +0.0036, head −0.0010, medium −0.0017, tail +0.0109. D(with) − D(zero), same model: +0.0188 mAP (unseeded: +0.0215; 5-seed range +0.0158 to +0.0215).

Uncertainty on the seeded C − A mAP from resampling validation rows (300 paired bootstraps over the 17,270 rows, same rows for both models; captures validation-sampling noise only, not training noise, and it is optimistic because the split is not patient-disjoint): +0.0022, 95% [−0.0012, +0.0080]. By group: head +0.0000 [−0.0008, +0.0009], medium −0.0002 [−0.0012, +0.0008], tail +0.0100 [−0.0042, +0.0346]. For comparison the unseeded seed-42 pair: −0.0053 [−0.0148, +0.0033].

## Per-class AP (seeded, flip TTA), tail classes and where the group differences come from
| Class (group) | val positives | A | C | C − A seeded | C − A unseeded | D(zero) − A seeded | A seeded − A unseeded |
|---|---|---|---|---|---|---|---|
| Hydropneumothorax (tail) | 7 | 0.2310 | 0.2833 | +0.0523 | −0.1055 | +0.0967 | −0.1407 |
| Mass (tail) | 49 | 0.0709 | 0.0742 | +0.0033 | −0.0072 | −0.0017 | −0.0003 |
| Pneumothorax (tail) | 71 | 0.3757 | 0.3667 | −0.0090 | +0.0238 | −0.0178 | +0.0191 |
| Subcutaneous Emphysema (tail) | 29 | 0.6076 | 0.5849 | −0.0227 | +0.0010 | −0.0054 | +0.0207 |
| azygos lobe (tail) | 58 | 0.5694 | 0.5698 | +0.0005 | +0.0095 | +0.0065 | +0.0168 |
| calcified densities (tail) | 140 | 0.1250 | 0.1272 | +0.0021 | +0.0018 | +0.0089 | −0.0083 |
| pneumoperitoneo (tail) | 9 | 0.3086 | 0.3471 | +0.0385 | −0.0765 | −0.0106 | −0.0523 |
| head: Normal / cardiomegaly / pleural effusion | 6917 / 1833 / 2082 | 0.8760 / 0.6483 / 0.7639 | 0.8757 / 0.6487 / 0.7640 | −0.0003 / +0.0003 / +0.0002 | −0.0006 / +0.0029 / +0.0028 | | |

Seeded tail C − A = +0.0093 is the sum over seven classes of (+0.0523, +0.0033, −0.0090, −0.0227, +0.0005, +0.0021, +0.0385)/7; Hydropneumothorax and pneumoperitoneo (7 and 9 validation positives) contribute +0.0130 of it, and the other five tail classes average −0.0052. In the unseeded pair the same two classes moved the other way (−0.1055, −0.0765) and produced tail C − A = −0.0219. The full 30-class table (all classes, seeded and unseeded) was printed by the analysis scripts `/tmp/claude-72139/-home-psytp7-ISBI-2026/deaa42c1-12f8-4dea-90bf-8caf0c1b9dd0/scratchpad/an.py` and `an2.py` (session scratchpad, not kept in the repo); it can be regenerated from the dumps with `analysis/entropy_ablation.tm_metrics`. The 20 medium classes show |C − A| < 0.006 for every class (largest +0.0053 pleural thickening, −0.0034 Hernia).

## Do the seeded runs reproduce the unseeded seed-42 runs? (single-model flip-TTA mAP)
| Row | Unseeded s42 | Seeded s42 | Difference | Unseeded 5-seed range (mean) |
|---|---|---|---|---|
| A | 0.4656 | 0.4605 | −0.0051 | 0.4585-0.4656 (0.4617) |
| C | 0.4605 | 0.4625 | +0.0020 | 0.4577-0.4653 (0.4612) |
| D with history | 0.4839 | 0.4807 | −0.0032 | 0.4769-0.4839 (0.4805) |
| D withheld | 0.4624 | 0.4619 | −0.0005 | 0.4566-0.4663 (0.4611) |

No: the seeded runs do not reproduce the unseeded seed-42 numbers (differences up to 0.005 mAP, per-class AP of Hydropneumothorax by 0.14), which is expected, because the seeded shuffle is a different sample stream from the earlier process-time-seeded one. All four seeded values lie inside the unseeded 5-seed range. Prediction similarity of the SWA flip-TTA outputs (mean over the 30 classes of the logit correlation across the 17,270 images): A seeded vs A unseeded 0.970, C seeded vs C unseeded 0.966, **A seeded vs C seeded 0.983** (A unseeded vs C unseeded 0.969); mean |Δp| 0.0162 / 0.0165 / **0.0097** / 0.0167. So seeding makes the A and C runs more alike (common data order), but C still differs from A (triplet term, and RNG streams other than the shuffle that were not shown to be fixed). Whether a second seeded run of the same config reproduces itself bit-for-bit or within what tolerance has **not been tested** (no repeated seeded run exists), so the seeded-pair noise floor is unknown; the only floor available is the unseeded same-seed rerun gap (0.0001-0.0079 mAP, [2026-10-14 result](2026-10-14-asl-entropy-markov-768.md)) and the seed-to-seed sd of the paired C − A difference (0.0035).

## Training-time validation metrics (SWA weights, NO TTA, from `history.json` / log `SWA weights ->` line; not comparable to the flip-TTA table above)
| Run | best-epoch EMA mAP (epoch) | SWA mAP | SWA mAUC | SWA mF1 | SWA mECE | SWA val_loss |
|---|---|---|---|---|---|---|
| A seeded | 0.4547 (3) | 0.4567 | 0.9090 | 0.4711 | 0.1327 | 0.0394 |
| C seeded (lambda 0.1) | 0.4569 (4) | 0.4590 | 0.9091 | 0.4695 | 0.1356 | 0.0394 |
| D seeded | 0.4770 (2) | 0.4772 | 0.9109 | 0.4918 | 0.1339 | 0.0387 |
| **P seeded (lambda 1.0, probe)** | 0.4505 (4) | **0.4535** | 0.9105 | 0.4648 | **0.1543** | 0.0398 |
| A unseeded s42 | 0.4597 (3) | 0.4609 | 0.9079 | 0.4765 | 0.1337 | 0.0395 |
| C unseeded s42 | 0.4554 (3) | 0.4578 | 0.9115 | 0.4629 | 0.1346 | 0.0394 |
| D unseeded s42 (49375) | 0.4793 (3) | 0.4807 | 0.9151 | 0.4863 | 0.1342 | 0.0388 |

Log lines (`Done. Best epoch` / `SWA weights ->`) match `history.json` to the printed precision for all four seeded runs (A 0.4547399878 / 0.4567; C 0.4568668 / 0.4590; D 0.4770450 / 0.4772; P 0.4505424 / 0.4535). Flip TTA adds about +0.003 to +0.005 mAP over these (A +0.0038, C +0.0035, D(with) +0.0035, seeded).

Per-epoch EMA mAP: A [0.4361, 0.4546, 0.4547, 0.4526]; C [0.4364, 0.4552, 0.4551, 0.4569]; P [0.4291, 0.4490, 0.4498, 0.4505]; D [0.4579, 0.4770, 0.4759, 0.4722]. Per-epoch EMA mECE: C [0.1526, 0.1454, 0.1362, 0.1240]; P [0.1625, 0.1600, 0.1501, 0.1399]; A [0.1514, 0.1437, 0.1338, 0.1213].

## Probe P (lambda 1.0) versus C (lambda 0.1), training-time only
- SWA no-TTA: mAP 0.4535 vs 0.4590 (−0.0055), mF1 0.4648 vs 0.4695, mAUC 0.9105 vs 0.9091 (+0.0014), val_loss 0.0398 vs 0.0394, **mECE 0.1543 vs 0.1356 (+0.0187)**. Per-epoch EMA mECE is higher at every epoch (+0.0099, +0.0146, +0.0139, +0.0159).
- Best-epoch EMA mAP 0.4505 vs 0.4569 (−0.0064); both best at epoch 4.
- Triplet yield is the same for both weights (`triplet_stats`, epoch 4): steps with at least one triplet 20,821 / 25,822 (80.6%) for P versus 20,650 (80.0%) for C; anchors 58,394 vs 56,546; triplets 31,577 vs 30,869. This is 80% of steps, not the 7.6% of the paper, and not the "13%" figure; the stats here count steps with at least one triplet under `--anchor-rule mean` with `easy` mining, so they are a different quantity from the paper's utilisation (open confound, not resolved here).
- Flip-TTA eval for P (job 52656): mAP 0.4557 vs C 0.4625 (−0.0068), mECE 0.1542 vs 0.1356 (+0.0186); the training-time picture holds under flip TTA.
- Why the triplet term barely acts in C (gradient probes, offline analysis): [docs/plan/2026-10-15-triplet-memory-debug.md](../plan/2026-10-15-triplet-memory-debug.md).
- n = 1: the mAP gap (−0.0055) is about the size of the seed noise (paired sd 0.0035 flip-TTA, same-seed rerun gap up to 0.008), so it is not a distinguishable effect. The mECE gap (+0.0187) is 20 times the seed sd of mECE differences (0.0009) and is the clear signal.

## Comparison to baselines
- vs unseeded 5-seed means ([2026-10-14](2026-10-14-asl-entropy-markov-768.md): A 0.4617 ± 0.0031, C 0.4612 ± 0.0028, D with 0.4805, D withheld 0.4611): seeded A −0.0012, C +0.0013, D with +0.0002, D withheld +0.0008, all within 0.5 sd (A: 0.4 sd, C: 0.5 sd).
- vs Stage-1 backbone (0.385 internal, 384 px era), MoE (0.405 internal / 0.4599 test) and the paper's 5-seed entropy triplet (0.414 ± 0.002): different recipe/resolution, not compared quantitatively.
- vs prior best on this recipe family: D with history (leakage-prone) is 0.4807 here, equal to the earlier 49375 SWA no-TTA 0.4807 by coincidence (this is flip TTA of a retrain; its no-TTA SWA is 0.4772).

## Caveats / known-bug flags
- n = 1 per row (one seed). No contrast is a statistical claim; the bootstrap is over validation rows only.
- Post-fix runs only (shard class weights, `--metric-embed query`). Easy mining, `--anchor-rule mean`, and the 7.6%/13%/80% utilisation discrepancy remain open paper confounds; this experiment does not resolve them. Resolution is 768 px for every row (no 384-vs-512 confound). The softmax-vs-sigmoid entropy question applies to C, D, P only.
- Val split is not patient-disjoint (77% of val images share a patient with train, 53% share their StudyID): the D(with) numbers are leakage-prone and not test-realistic; D(zero) is the test-like row.
- Eval logs carry the same `libpng` truncated-PNG warnings as before; all four evals have n_rows 17,270.
- Tail AP is seven classes, two of which have 7 and 9 validation positives; the tail column is not stable.

## Notable
- The sign of the ASL-vs-Entropy gap flipped between the unseeded and seeded seed-42 pair (−0.0051 → +0.0020 mAP; tail −0.0219 → +0.0093), with both deltas inside the seed noise (paired sd 0.0035). The only metric whose sign is the same in the seeded pair, the unseeded pair and all five unseeded seeds is mECE (Entropy worse by +0.0011 to +0.0034).
- Job 52655 (lambda 1.0) shows the memory path can change the model (mECE +0.019, mAP −0.006), but not in a good direction.

## Next step
P flip-TTA is now in (above). See the findings file for the recommendation and the debug report for why the triplet term is near-inactive.
