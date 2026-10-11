# v5 patient-prior (B-early) reproducibility rerun — 2026-10-05

**Directory:** `train/checkpoint_Triplet_3/prior/img768_qprior_seed{42,1024}_rerun/Model_run/` (reruns); originals `train/checkpoint_Triplet_3/prior/img768_qprior_seed{42,1024}/Model_run/`
**Script / recipe:** `train/train_2_v5.py` (v3 push-wave recipe at 768 px + `PriorQueryConditioner`, see `train/prior_conditioning.py`); SWA over epochs 2-4
**Slurm jobs:**
- Training reruns: 49855 (s42), 49856 (s1024). Originals: 49375 (s42), 49376 (s1024). All COMPLETED 0:0 on node juggernaut (sacct; 11:30:09 / 11:30:09 for the reruns, 11:35:37 / 11:12:20 for the originals).
- Evals (2-seed SWA ensemble, flip TTA, full 17,270-row internal val): 49857 orig-with, 49859 orig-zero, 49858 rerun-with, 49860 rerun-zero. All COMPLETED 0:0.
- Offline analysis (this report): `analysis/v5_rerun_compare.py` and `analysis/v5_rerun_excl_hydro.py` (srun on the general partition); outputs `analysis/out/v5_rerun/{run.log,res.json,run_excl_hydro.log,res_excl_hydro.json}`.

**Logs:** `/home/psytp7/logs/isbi2026_v5prior_qprior_s42_rerun_49855.out`, `..._s1024_rerun_49856.out`, `..._s42_49375.out`, `..._s1024_49376.out`, `isbi2026_ev_v5_{orig,rerun}_{with,zero}_{49857,49859,49858,49860}.out`.
**Claims check (what a paper can say):** [docs/plan/2026-10-05-v5-rerun-claims-check.md](../plan/2026-10-05-v5-rerun-claims-check.md).
**Earlier reports:** [Markov A/B training 2026-10-01](2026-10-01-markov-ab-training.md), [verdict](../plan/2026-10-01-markov-ab-verdict.md), [0.4722 provenance](../plan/2026-10-02-047-provenance-and-paper-corrections.md), [post-hoc HMM](../plan/2026-10-01-markov-hmm-posthoc.md), [v4 result](2026-09-27-v4-markov-result.md).

All numbers are INTERNAL val (17,270 rows, not patient-disjoint, 30% lateral). Nothing here is leaderboard/test. "WITH" = the previous report's labels are supplied as a prior (a longitudinal simulation); "ZERO" = prior zeroed, which is the leaderboard/test condition (test patients have no history).

## Configuration
Each run's `args:` line (L2 of its log) was compared token by token with its original after normalising `--out-dir`. **The args lines are identical** for s42 (49375 vs 49855) and s1024 (49376 vs 49856); the only difference is `--out-dir ..._rerun`.

| Setting | Value (all four runs) |
|---|---|
| Stage-1 checkpoint | checkpoint3/Model_20260119_062652/model_best.pth |
| Image size | 768 |
| Seeds | 42, 1024 (original and rerun use the same seeds) |
| LR / batch / accum | 1e-4 / 4 / 12, epochs 4 of sched-epochs 8, patience 2, drop-path 0.1, swa-last-k 3 |
| metric-embed / memory-size / triplet-lambda | query / 2048 / 0.1 |
| class-weight-order | shard (fixed) |
| Prior | `--prior-file analysis/out/patient_markov/priors.pt --prior-dropout 0.3 --prior-lr-mult 10` (same file) |

Same seeds and same args still gave different models: training is not bit-reproducible here (cause not investigated; `cudnn.deterministic=True` is set at `train_2_v5.py:1106`, so the nondeterminism is elsewhere, e.g. data-loader/shuffle or atomics). The evaluation, however, is deterministic: the orig-recheck dumps (jobs 49857/49859) reproduce the 2026-10-01 dumps with max |d| = 0.0, and `tta_v5_orig_recheck_{with,zero}.json` equal `tta_ab_v5_{with,zero}.json` (0.4867 / 0.4693). So every orig-vs-rerun difference below comes from training. Row order: `keys.txt` is identical (same md5) in `probs_ab_flip`, `probs_v5_orig_recheck_flip`, `probs_v5_rerun_flip`; `labels.npy` identical; keys equal the reconstructed `val_key_order.npy` (checked in `v5_rerun_compare.py`); the v3 s42 dump in `probs_w01_flip` matches `probs_ab_flip` with max |d| = 0.0.

## Results

### Single models, training log (EMA best epoch, then SWA of epochs 2-4; no TTA; WITH prior)
| Run | Job | Best ep | Best mAP | SWA mAP | SWA mAUC | SWA mF1 | SWA mECE |
|---|---|---|---|---|---|---|---|
| orig s42 | 49375 | 3 | 0.4793 | 0.4807 | 0.9151 | 0.4863 | 0.1342 |
| rerun s42 | 49855 | 2 (early-stop line at ep 4) | 0.4740 | 0.4731 | 0.9132 | 0.4853 | 0.1328 |
| orig s1024 | 49376 | 2 (early-stop line at ep 4) | 0.4760 | 0.4786 | 0.9141 | 0.4903 | 0.1324 |
| rerun s1024 | 49856 | 3 | 0.4761 | 0.4764 | 0.9158 | 0.4849 | 0.1329 |
| v3 baseline s42 (49000, no prior) | 49000 | 3 | 0.4585 | 0.4600 | 0.9092 | 0.4685 | 0.1354 |
| v3 baseline s1024 (48999, no prior) | 48999 | 3 | 0.4564 | 0.4582 | 0.9104 | 0.4741 | 0.1354 |

history.json agrees with the log `Done. Best` / `SWA weights ->` lines for all four v5 runs (checked; best epoch, best mAP and SWA metrics identical).

Per-epoch EMA mAP (history.json): orig s42 .4587 / .4781 / .4793 / .4735; rerun s42 .4583 / .4740 / .4728 / .4637; orig s1024 .4607 / .4760 / .4747 / .4759; rerun s1024 .4594 / .4733 / .4761 / .4697. Epoch 1 agrees to 0.0004 to 0.0013, the runs diverge from epoch 2 (s42 rerun is 0.004-0.010 lower at epochs 2-4).

Run-to-run (same seed, SWA): s42 0.4807 vs 0.4731 (gap 0.0076); s1024 0.4786 vs 0.4764 (gap 0.0022). Four-run SWA mean 0.4772, sample s.d. 0.0032 (best-epoch mean 0.4764, s.d. 0.0022). Reference v3 seed spread: 0.0018 (768 px SWA, two seeds), s.d. 0.0025 (512 px, six seeds).

### Ensemble evals (2-seed SWA uniform average of probabilities, flip TTA, n = 17,270)
Metrics from the `tta_v5_*.json` files; mAP re-computed from the dumps agrees to 4 decimals.

| Ensemble | Prior | mAP | mAUC | mF1 | mECE |
|---|---|---|---|---|---|
| orig (49857 / 49859) | WITH | 0.4867 | 0.9170 | 0.4980 | 0.1332 |
| orig | ZERO | 0.4693 | 0.9131 | 0.4811 | 0.1358 |
| rerun (49858 / 49860) | WITH | 0.4797 | 0.9169 | 0.4904 | 0.1328 |
| rerun | ZERO | 0.4611 | 0.9128 | 0.4706 | 0.1352 |
| 4-model (orig + rerun), recomputed by me | WITH | 0.4847 | n/c | n/c | n/c |
| 4-model | ZERO | 0.4667 | n/c | n/c | n/c |
| v3 matched 2-seed (push 768 s42 + s1024 SWA, flip), recomputed | none | 0.4659 | n/c | n/c | n/c |
| v3 greedy-3 (`ens_select_w01_flip.json`, selected on val) | none | 0.4722 | n/c | n/c | n/c |

n/c = not computed here. The v3 matched 2-seed 0.4659 and greedy-3 0.4722 were recomputed from the dumps and agree with the earlier reports.

Single-model flip-TTA full-val mAP (recomputed from dumps): orig WITH 0.4839 (s42) / 0.4820 (s1024); orig ZERO 0.4624 / 0.4663; rerun WITH 0.4760 / 0.4786; rerun ZERO 0.4560 / 0.4622; v3 0.4650 (s42) / 0.4626 (s1024).

### Q1. Does the rerun reproduce v5?
Quantified: it reproduces the **direction** (history helps, zero history does not) and the **size of the prior's own contribution**, but not the headline 0.4807 / 0.4867.

| Quantity (ensemble, flip, full val) | orig | rerun | rerun - orig (paired, patient-clustered bootstrap, 300) |
|---|---|---|---|
| WITH mAP | 0.4867 | 0.4797 | -0.0070 [-0.0143, -0.0011] (image-level -0.0070 [-0.0148, -0.0010]) |
| ZERO mAP | 0.4693 | 0.4611 | -0.0081 [-0.0144, -0.0006] (image-level -0.0081 [-0.0175, -0.0002]) |
| Prior's contribution (WITH - ZERO, same models) | +0.0175 [+0.0110, +0.0245] | +0.0186 [+0.0124, +0.0247] | +0.0011 |
| WITH - ZERO on has-prior & frontal rows (n 4061) | +0.0329 [+0.0124, +0.0515] | +0.0317 [+0.0088, +0.0502] | -0.0012 |
| mAP on no-prior rows (n 11,511; same for WITH and ZERO) | 0.4571 | 0.4489 | -0.0082 [-0.0171, +0.0030] |

- The run-to-run gap of 0.007-0.008 on the ensemble is 3-4x the v3 seed s.d. (about 0.002) and its CI just excludes 0. It is a gap in the **base image model**, not in how the prior is used (the WITH-ZERO contribution is +0.0175 vs +0.0186, and no-prior rows drop by the same 0.008).
- **The gap is almost entirely one class: Hydropneumothorax** (7 positives in the 17,270 val rows). Its ensemble AP is orig WITH 0.4657 / ZERO 0.4950 and rerun WITH 0.2854 / ZERO 0.2834 (v3 matched 0.3384). Per-class dAP rerun - orig is -0.1803 (WITH) and -0.2117 (ZERO); every other class is within about 0.023 (largest others: azygos lobe -0.016, Subcutaneous Emphysema -0.013 / -0.023, Pneumothorax +0.011 / +0.015). One class moves macro mAP by dAP/30, so Hydropneumothorax alone accounts for 0.0060 of the 0.0070 WITH gap.
- Excluding Hydropneumothorax (29-class mean; a post-hoc diagnostic I chose after seeing the per-class table, so treat as exploratory): orig WITH 0.4874, rerun WITH 0.4864, orig ZERO 0.4684, rerun ZERO 0.4672, v3 matched 0.4703. Run-to-run gap -0.0010 [-0.0033, +0.0013] (WITH) and -0.0011 [-0.0038, +0.0019] (ZERO): inside the v3 seed s.d. WITH - ZERO is +0.0190 (orig) and +0.0192 (rerun). Singles excluding Hydropneumothorax: orig WITH 0.4849 / 0.4839, rerun WITH 0.4839 / 0.4836; v3 0.4696 / 0.4651.
- Hydropneumothorax AP of single models ranges from 0.2465 (rerun s42 ZERO) to 0.5065 (orig s1024 ZERO). The same noise applies to every full-val mAP comparison in the campaign, including the v3 baselines (0.3321 s42, 0.3893 s1024). That was not flagged in earlier reports.
- Other classes: orig vs rerun per-class dAP on head 3 / medium 20 classes is within +-0.006 for every class (mean -0.0003 head, -0.0004 medium WITH).

### Paired comparisons against the no-prior v3 matched 2-seed flip ensemble (0.4659)
Paired bootstrap of the macro-mAP difference, 300 resamples, resampling patients with all their rows (image-level agrees to within about 0.001 on the full-val rows, see `res.json`). Subset n: all 17,270; has-prior & frontal 4,061; frontal & patient-unseen 3,572.

| Comparison | all (30 cls) | all excl. Hydropneumothorax | has-prior & frontal | frontal & patient-unseen |
|---|---|---|---|---|
| rerun WITH - v3 | +0.0138 [+0.0042, +0.0209] | +0.0161 [+0.0104, +0.0215] | +0.0321 [+0.0153, +0.0483] | -0.0046 [-0.0215, +0.0075] |
| orig WITH - v3 | +0.0208 [+0.0132, +0.0285] | +0.0171 [+0.0119, +0.0227] | +0.0351 [+0.0178, +0.0509] | -0.0105 [-0.0372, +0.0022] |
| 4-model WITH - v3 | +0.0188 [+0.0104, +0.0278] | +0.0180 [+0.0125, +0.0235] | +0.0349 [+0.0170, +0.0504] | -0.0061 [-0.0239, +0.0042] |
| rerun ZERO - v3 | -0.0048 [-0.0129, +0.0004] | -0.0031 [-0.0064, -0.0001] | +0.0004 [-0.0093, +0.0150] | -0.0060 [-0.0214, +0.0064] |
| orig ZERO - v3 | +0.0033 [-0.0033, +0.0074] | -0.0019 [-0.0056, +0.0010] | +0.0022 [-0.0063, +0.0146] | -0.0116 [-0.0366, +0.0008] |
| 4-model ZERO - v3 | +0.0008 [-0.0060, +0.0063] | -0.0011 [-0.0041, +0.0014] | +0.0039 [-0.0049, +0.0187] | -0.0070 [-0.0247, +0.0031] |

Against the greedy-3 v3 ensemble (0.4722, mixed 768/1024/640 px, selected on this fold, so optimistic): ZERO ensembles are lower. Full val, 30 classes: orig ZERO -0.0030 [-0.0104, +0.0042], rerun ZERO -0.0111 [-0.0153, -0.0056], 4-model ZERO -0.0055 [-0.0106, +0.0018]. Excluding Hydropneumothorax: -0.0105 [-0.0135, -0.0069], -0.0116 [-0.0157, -0.0076], -0.0096 [-0.0129, -0.0055]. On has-prior & frontal rows: -0.0100 to -0.0139 (30-class and 29-class), all CIs exclude 0. On frontal & patient-unseen: -0.0004 to -0.0060, CIs include 0. The with-prior ensembles (0.4797 to 0.4867) are above 0.4722 on full val.

### Head / medium / tail dAP vs v3 matched 2-seed (full val, canonical split from `evaluate/class_groups.py`: head 3, medium 20, tail 7)
This split is the canonical `frequency_groups()` (the same function `markov_ab_compare.py` calls), which resolves the 2026-10-01 caveat that it had not been cross-checked. Tail positives in val are small: Hydropneumothorax 7, pneumoperitoneo 9, Subcutaneous Emphysema 29, Mass 49, azygos lobe 58, Pneumothorax 71, calcified densities 140.

| Arm - v3 matched | head | medium | tail (7 cls) | tail excl. Hydropneumothorax (6 cls, derived) |
|---|---|---|---|---|
| orig WITH | +0.0074 (3/3 up) | +0.0192 (20/20 up) | +0.0311 (5/7 up) | +0.0151 |
| rerun WITH | +0.0072 (3/3 up) | +0.0188 (20/20 up) | +0.0024 (4/7 up) | +0.0117 |
| orig ZERO | -0.0010 (0/3 up) | -0.0019 (4/20 up) | +0.0202 (3/7 up) | -0.0025 |
| rerun ZERO | -0.0013 (0/3 up) | -0.0024 (5/20 up) | -0.0132 (2/7 up) | -0.0062 |

(The last column is the mean of the six non-Hydropneumothorax tail-class dAPs from `res.json`; Hydropneumothorax dAP vs v3 is +0.1273 / -0.0531 / +0.1566 / -0.0551 for orig WITH / rerun WITH / orig ZERO / rerun ZERO.)

- The medium/head gain from the prior is stable between runs (medium +0.019 in both, all 20 classes up, with prior). With the prior zeroed, head and medium classes are 0.001-0.002 below v3 in both runs (point estimates; mostly classes down).
- The tail is not reproducible at this sample size: the orig-vs-rerun tail difference is -0.029 (WITH) and -0.033 (ZERO), carried by Hydropneumothorax. Even the six other tail classes move by +0.0151 vs +0.0117 (WITH) and -0.0025 vs -0.0062 (ZERO) against v3 between orig and rerun; none is a stable tail effect. No tail claim should rest on v5.
- Largest per-class WITH gains over v3 (rerun, similar in orig): fracture +0.067, bronchiectasis +0.046, calcified densities +0.046, Pneumothorax +0.047, pleural thickening +0.034, Mass +0.042. ZERO per-class deltas vs v3 are within +-0.01 except tail.

### Seen vs unseen patients (this run's dumps)
Subsets: 33.3% of val images (5,759) have a strictly earlier study in train U val; 77.0% share a patient with train (13,304 patient-seen, 3,966 patient-unseen); frontal & patient-unseen n = 3,572.

| Ensemble mAP | all | has-prior (5,759) | no-prior (11,511) | has-prior & frontal | patient-unseen (3,966) | frontal & patient-unseen (3,572) | patient-seen (13,304) |
|---|---|---|---|---|---|---|---|
| orig WITH | 0.4867 | 0.5302 | 0.4571 | 0.5200 | 0.4689 | 0.4703 | 0.4886 |
| orig ZERO | 0.4693 | 0.4871 | 0.4571 | 0.4871 | 0.4679 | 0.4692 | 0.4678 |
| rerun WITH | 0.4797 | 0.5278 | 0.4489 | 0.5170 | 0.4759 | 0.4761 | 0.4811 |
| rerun ZERO | 0.4611 | 0.4865 | 0.4489 | 0.4854 | 0.4747 | 0.4748 | 0.4601 |
| v3 matched 2-seed | 0.4659 | 0.4871 | 0.4563 | 0.4850 | 0.4795 | 0.4808 | 0.4635 |
| v3 greedy-3 | 0.4722 | 0.4978 | 0.4616 | 0.4989 | 0.4743 | 0.4752 | 0.4703 |

- On frontal & patient-unseen rows the prior adds almost nothing: WITH - ZERO is +0.0011 [-0.0016, +0.0029] (orig) and +0.0014 [-0.0009, +0.0030] (rerun), and WITH is below the no-prior v3 matched ensemble (-0.0105 [-0.0372, +0.0022] orig, -0.0046 [-0.0215, +0.0075] rerun, CIs include 0).
- The whole gain sits on has-prior rows (WITH - ZERO +0.0431 on has-prior orig, +0.0413 rerun; derived from the table). No-prior rows are identical between WITH and ZERO by construction.

## Comparison to baselines
- vs Stage-1 backbone (internal mAP 0.385, from the task brief, not re-measured here): all v5 arms are far above it, but v3 already accounts for that.
- vs MoE Stage-2 (internal 0.405 / test 0.4599) and the entropy paper's 5-seed result (0.414 +- 0.002, 512 px, pre-fix): not like-for-like (768 px, v3 recipe).
- vs matched no-prior v3 (single SWA 0.4600 / 0.4582; 2-seed flip ensemble 0.4659): WITH-history +0.0131 / +0.0182 (rerun singles, derived, no TTA) and +0.0207 / +0.0204 (orig singles); ensemble +0.0138 (rerun) / +0.0208 (orig). ZERO-history: -0.0048 (rerun) / +0.0033 (orig), CIs include 0.
- vs greedy-3 v3 (0.4722, no Markov, no metadata): with history above (+0.0075 rerun, +0.0145 orig on full val, derived); without history below (see paired table).
- vs the campaign's earlier best at the time of the original runs (v5 s42 SWA 0.4807): not reproduced; the reproducible single-run level is 0.4731 to 0.4807 (WITH).

## Caveats / known-bug flags
- Post-fix recipe in all four runs: `--metric-embed query` and `--class-weight-order shard` are present in every `args:` line. No known-bug flag applies.
- The WITH numbers use history from train U val labels (previous report), which does not exist for test images. Leaderboard effect of WITH is 0 by construction.
- Internal val is not patient-disjoint (77.0% of val images share a patient with train, 53% their StudyID). ZERO numbers are therefore still optimistic for a patient-disjoint test set, and the frontal & patient-unseen subset (n 3,572) is the better proxy.
- Hydropneumothorax has 7 val positives; macro mAP over 30 classes is dominated by its noise in any run-to-run comparison. The excl.-Hydropneumothorax rows are exploratory (the exclusion was chosen after seeing the per-class deltas).
- Bootstrap: 300 resamples, patient-clustered; image-level only for full-val rows. One val fold, two seeds per arm, and the 4-model ensemble mixes orig and rerun members (not independent of the pairs above).
- The v3 baseline is a 2-seed ensemble from the same recipe, not trained in the same session as the v5 reruns.
- The excl.-Hydropneumothorax script is `analysis/v5_rerun_excl_hydro.py` (a derivative of `v5_rerun_compare.py`, nulling class index 2); its "mAP by subset" printout is a stub of zeros and should be ignored.
- Paper-confound items (384 vs 512 px, softmax entropy, 7.6% vs 13% triplet use) do not bear on this comparison; all arms are 768 px with softmax entropy and about 80% triplet hit rate.

## Notable
- **The original v5 headline (single 0.4807, ensemble 0.4867) did not reproduce.** Same seeds and args gave 0.4731 (s42) and 0.4764 (s1024) single SWA, ensemble WITH 0.4797 and ZERO 0.4611. The best estimate of the single-model WITH level is the 4-run mean 0.4772 (s.d. 0.0032), the 4-model ensemble 0.4847.
- **Cause of most of the gap: Hydropneumothorax (7 val positives) AP 0.47 vs 0.29.** Without that class the reruns match the originals to 0.001, and the prior's own gain (+0.019 on 29 classes, +0.0175 to +0.0186 on 30) is reproducible. This is a noise source for every full-val mAP comparison in the campaign.
- **The zero-history result is weaker than reported on 2026-10-01.** orig ZERO was +0.0033 vs the matched v3; the rerun ZERO is -0.0048 (CI [-0.0129, +0.0004]); on 29 classes both are below v3 (-0.0019, -0.0031). There is no evidence that v5 helps without history.
- The v3 greedy-3 ensemble (0.4722) is above every ZERO ensemble in this batch, so the "0.47 level" is reached by v3 with no patient information; see the claims check.

## Next step
- If a v5 headline number is still wanted: report the 4-run mean (or the 4-model ensemble 0.4847 WITH / 0.4667 ZERO) with the spread, not 0.4807 / 0.4867.
- A per-class eval via `eval_perclass.sh` for the v3 and v5 SWA checkpoints would give per-class AP with the standard flow (not run; per-class numbers above are from my dumps analysis).
- Because Hydropneumothorax and pneumoperitoneo (7 and 9 val positives) swing mAP, a stability check for any further comparison should report the 29-class (or head+medium) mAP beside the 30-class mAP.
- Reproducibility of training itself (same seed, different result) is unexplained; if exact reproduction matters, find the nondeterminism source before more single-seed comparisons.
