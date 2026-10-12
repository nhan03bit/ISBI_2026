# Why the entropy-gated triplet shows no measurable gain — 2026-10-12

**Directory:** `train/checkpoint_Triplet_3/{ablation512/{asl,nogate,randgate,gate},seedsweep}/seed_*/Model_run/` (512 px arms A/B/B'/C); `train/checkpoint_Triplet_3/push/img768_dp01_seed*/` and `train/checkpoint_MoE/{asl,moe}_img768_seed*/` (768 px)
**Script / recipe:** `train/train_2_v3.py` (512 px arms; Triplet Memory at 768 px), `train/train_2_v5.py --no-triplet` (ASL and MoE at 768 px). Sampler and bank: `train/utils_update.py` (`CrossBatchMemoryV2`, `sample_triplets_v13`).
**Slurm job(s):**
- 512 px ablation, seeds in the order 42 / 86 / 123 / 456 / 789 / 1024. A (asl): 51227 / 51209 / 51215 / 51233 / 51239 / 51221. B (nogate): 51229 / 51211 / 51217 / 51235 / 51241 / 51223. B' (randgate): 51231 / 51213 / 51219 / 51237 / 51243 / 51225. C originals: 47097 / 46472 / 46473 / 47098 / 47099 / 46476. C reruns on today's code: 51245 (s86), 51247 (s1024). Evals 51765-51774 and earlier chains.
- 768 px: Triplet Memory (TM) 49000 (s42), 48999 (s1024); MoE 51153 / 51154; ASL 51158 / 51159.
- Logs: `/home/psytp7/logs/isbi2026_abl_<arm>_s<S>_<jobid>.out`, `isbi2026_v3seed_s<S>_<jobid>.out`, `isbi2026_v3push_img768_dp01_s<S>_<jobid>.out`, `isbi2026_{asl,moe}_img768_s<S>_<jobid>.out`.

Related: [`docs/result/2026-10-11-moe-matched-eval.md`](../result/2026-10-11-moe-matched-eval.md) (768 px matched MoE / ASL / TM results) and [`docs/plan/2026-10-08-entropy-ablation-isbi-submit-plan.md`](2026-10-08-entropy-ablation-isbi-submit-plan.md) (pre-registration of the 512 px ablation). Per-run metrics are quoted below only as evidence. The formal results report of the ablation (plan Step 4, `docs/result/<date>-entropy-ablation-512.md`) is a separate document and is not written here. Another workspace, `/home/psytp7/ISBI_entropy_ablation` (768 px seed-42 three-arm rerun, jobs 51761-51764), exists and was not touched.

**Status of this document.** The coordinator stopped the investigation early (usage limit). Everything computed before that is reported with its tag. Items that were planned and not finished are marked **NOT TESTED**. No file under `train/`, `evaluate/`, `analysis/`, `scripts/` or `docs/paper/` was edited, no Slurm job was submitted, and no GPU work was done.

Tags used throughout: **ESTABLISHED** = measured from an artifact; **INFERRED** = follows from measured numbers plus code; **UNKNOWN** = needs a diagnostic that cannot be run from existing artifacts; **NOT TESTED** = planned but not run.

## Question
The entropy-gated triplet + cross-batch memory bank Stage-2 objective ("Triplet Memory", arm C) shows no measurable improvement over an ASL-only Stage-2 control (arm A) and over a matched-recipe sparse MoE, at the same seeds. Why?

## Data and method
- Arms at 512 px, 6 seeds (42, 86, 123, 456, 789, 1024), fixed Stage-1 checkpoint `checkpoint3/Model_20260119_062652/model_best.pth` and recipe (`--accum-steps 6 --batch-size 8 --lr 1e-4 --epochs 8 --patience 3 --memory-size 2048 --metric-embed query --head-classes 5 --class-weight-order shard`): A = ASL only (`--triplet-lambda 0`), B = triplet + bank, every image an anchor (`--anchor-rule all`), B' = random anchors, same count as the gate (`--anchor-rule random`), C = entropy gate (`--anchor-rule mean`, the six `seedsweep` runs). Two C reruns (s86, s1024) on today's code.
- All 26 `preds.<tag>.npz` / `eval_result.<tag>.json` files exist (checked 2026-10-12 01:47 BST; the last two, randgate s123 and s1024, were written at 01:41:51). `analysis/out/entropy_ablation.json` was not used as an input and `analysis/entropy_ablation.py` was not run.
- Validation checks of my own tooling: `eval_result` `val_mAP` vs `history.json` `best_metric`: max |diff| = 1.32e-05 (24 runs); my tie-grouped AP reproduces `eval_result` per-class AP to 7.2e-08 on all 26 runs x 30 classes. Class weights recomputed from `data/CXRLT_2026_training_filtered.csv` (N = 103,303) equal the weights printed in the C-rerun log; head-5 indices [6, 24, 17, 9, 12] (Normal, pleural effusion, cardiomegaly, Support Devices, aortic elongation) equal the logged `head_class_ids`.
- `history.json` stores per-epoch arrays only up to the best epoch (it is written when the monitor improves), so per-epoch trajectories beyond the best epoch come from the Slurm logs (`Epoch NN/8 | train_loss=... val_loss=... | mAP=...`, `triplet hit rate`, `anchors/step`). The six original C logs predate the `anchors/step` logging; their per-epoch triplets/step are proxied by the two C reruns (hit-rate lines of original and rerun logs agree within 0.3 pp).
- Scratch scripts (session scratchpad `/tmp/claude-72139/-home-psytp7-ISBI-2026/8ec7043b-bc5c-4ac8-9b4f-e9917bc3ba54/scratchpad/`, not committed): `common.py`, `meta.py`, `apfun.py`, `h1_power.py`, `h2_dose.py`, `h3_prep.py`, `h3_gate.py`, `h4_coverage.py`, `h4_perclass.py`, `h5_loss.py`, `h7_epochs.py`, `h7_768.py`, `h8_pred.py`, `h8_ens_perm.py`, `dup_scan.py`.

## 1. Verdict (ranked)

1. **[ESTABLISHED] The ablation cannot resolve effects below about 0.005 mAP, and "same seed" buys almost no variance reduction.** Paired n = 6 minimum detectable effect (80% power, two-sided 0.05) is 0.0043-0.0076 mAP; the observed point estimates are +0.0014 (C-A), +0.0027 (B-A), -0.0014 (C-B), 0.0000 (C-B'). Pooled within-arm s.d. is 0.0033; seeds explain only about 20% of the variance. The data bound the entropy-specific gain at C-B <= +0.0021 and C-B' <= +0.0049 (upper 95% limits). "No improvement" means "no entropy-specific gain above about 0.002-0.005", not "exactly zero".
2. **[INFERRED] The triplet term carries almost no loss mass, so it is close to a no-op.** The logged train loss exceeds the ASL-only train loss by only 0.0002-0.0005 (0.5-2.4% of the loss), i.e. a mean triplet term of 0.002-0.005 against a margin of 0.5. Code reading gives the mechanism: easy mining (closest positive, farthest negative) over a 2,048 bank satisfies a 0.5 margin through order statistics alone. The hinge-active fraction and the gradient ratio are UNKNOWN (no GPU measurement exists).
3. **[ESTABLISHED] The softmax-entropy gate is a "not Normal-only / higher-loss image" filter.** It does not select multi-finding images (Spearman with number of labels +0.04) and does not favour tail images (pass probability 0.561 vs 0.533). It gives +14% more valid triplets than a random gate of equal size and 58% higher mean per-image ASL loss among gated images, and still C-B' = 0.0000.
4. **[INFERRED] Little triplet signal reaches the tail.** 2,105 of 103,303 training images (2.04%) carry a tail label, about 4.2% of valid triplets; the bank holds an own-class positive for 5 of 7 tail classes with probability >= 0.97 but only 0.53 / 0.64 for the two rarest. Tail AP differences between arms are large in size (B-A +0.0150) but unresolvable (paired s.d. 0.010-0.022, MDE 0.016-0.031).
5. **[INFERRED] The best epoch (2 of 8, LR still about 90% of peak) is selected before a regulariser can show.** Arms separate only late (C val loss 0.0003-0.0006 below A at epochs 3-5; C-B mAP at epoch 5 +0.0030, uncorrected p = 0.044 among 25 tests), after early stopping has chosen epoch 2.
6. **[ESTABLISHED as a pattern, not significant] The only positive-leaning signal is "having triplets at all", not entropy.** mAP rises with triplets/step (A 0, B' 2.26, C 2.51, B 3.86): per-seed slope +0.00068 per triplet/step, p = 0.26 (Page trend p = 0.037, permutation p = 0.06 one-sided); 6-seed ensemble B-A +0.0039 (permutation p = 0.056; tail +0.0165, p = 0.041, uncorrected over 24 tests).
7. **[INFERRED] A likely cause of the weak pairing: the in-worker sample shuffle is not seeded by `--seed`** (installed webdataset 1.0.2 seeds it from pid + time). The plan's statement "arms with the same seed see the same data order" is therefore doubtful; empirically unverified.
8. **[REFUTED for these runs]** Triplet utilisation (hit rate 93-99% of steps at 512 px, not 7.6%), the class-weight order (`shard`), and bank bookkeeping (no defect found by reading `CrossBatchMemoryV2`; numerics NOT TESTED) do not explain the null.
9. **Published-number relationship (verified in `ISBI2026.pdf`).** The ASL-only 512 px control, 0.4381 +/- 0.0042, is about +0.033 above the published internal Stage-2 mAP 0.405 (not a match; recipes differ). See H9 and section 4.
10. **NOT TESTED:** see "Not measured / NOT TESTED" (bank numerics, data-order sibling effect, null-geometry simulation, formal mECE test).

## 2. Findings by hypothesis

### H1 Statistical power and noise — verdict: SUPPORTED (ESTABLISHED)

Per-run `best_metric` (computed from `history.json`; best epoch in brackets when not 2):

| Arm | s42 | s86 | s123 | s456 | s789 | s1024 | mean +/- s.d. |
|---|---|---|---|---|---|---|---|
| A (ASL) | 0.445371 | 0.436875 | 0.436038 | 0.437659 | 0.432843 | 0.439781 | 0.4381 +/- 0.0042 |
| B (all anchors) | 0.439251 | 0.440185 | 0.441997 | 0.440866 | 0.442268 | 0.440235 [3] | 0.4408 +/- 0.0012 |
| B' (random gate) | 0.447600 | 0.439655 | 0.440132 | 0.437572 | 0.436754 | 0.435170 [1] | 0.4395 +/- 0.0044 |
| C (entropy gate) | 0.441404 | 0.438524 | 0.438915 [3] | 0.437519 | 0.436946 | 0.443368 | 0.4394 +/- 0.0025 |
| C rerun | | 0.438556 | | | | 0.439952 | |

Your provisional numbers are reproduced exactly. Paired by seed (n = 6; exact sign-flip p; MDE = smallest true paired difference detected with 80% power at two-sided 0.05, exact non-central t):

| Contrast | mean | s.d. of diffs | 95% t-CI | p (sign-flip) | wins | MDE (80%) |
|---|---|---|---|---|---|---|
| C-B (pre-registered primary) | -0.0014 | 0.0033 | [-0.0048, +0.0021] | 0.375 | 2/6 | 0.0048 |
| C-B' | -0.0000 | 0.0047 | [-0.0049, +0.0049] | 0.969 | 2/6 | 0.0067 |
| B-A | +0.0027 | 0.0053 | [-0.0028, +0.0082] | 0.250 | 5/6 | 0.0076 |
| C-A | +0.0014 | 0.0030 | [-0.0018, +0.0045] | 0.344 | 4/6 | 0.0043 |
| B'-A | +0.0014 | 0.0033 | [-0.0021, +0.0049] | 0.375 | 4/6 | 0.0047 |
| B-B' | +0.0013 | 0.0051 | [-0.0040, +0.0067] | 0.531 | 5/6 | 0.0073 |

Source: `history.json` best_metric via `h1_power.py`. Repeated-measures ANOVA (arm effect, seeds as blocks): F(3,15) = 0.825, p = 0.501 (Friedman p = 0.241). Pooled within-arm s.d. 0.0033; unpaired MDE for 6 runs per arm 0.0060.

Does "same seed" help? Between-arm correlation of per-seed mAP: A-B -0.887, A-B' +0.706, A-C +0.712, B-B' -0.536, B-C -0.640, B'-C +0.166. Mean pairwise correlation -0.080 against a permutation null of 0.001 +/- 0.183 (one-sided p = 0.62). Var(d)/(Var_x + Var_y) is below 1 for C-A (0.38), B'-A (0.29), C-B' (0.86) and above 1 for C-B (1.49), B-A (1.45), B-B' (1.27). Variance components: seed s.d. 0.0015, residual s.d. 0.0030, so about 20% of the variance is shared by seed. Pairing is not demonstrably better than an unpaired design here (the negative correlations with B are inflated by B's tiny s.d.; with n = 6 each correlation is itself very noisy).

Noise floor at a fixed seed with identical config (INFERRED to be about 0.002-0.003 per run): C original vs rerun at s86 +0.000033 and at s1024 -0.003416; per-epoch C original vs rerun differences (2 seeds x 5 epochs) have RMS 0.0041, implying a same-seed run s.d. of 0.0029; the two identical-args v5-prior reruns at 768 px (jobs 49375/49855, 49376/49856, found by `dup_scan.py`) differ by -0.0053 (s42) and +0.00003 (s1024) in best-epoch mAP. RMS of the four best-metric rerun differences is 0.0032, so a run s.d. of 0.0022. With so few reruns the interval on this s.d. is wide, but it is of the same size as the between-seed s.d. of arm C (0.0025).

Seeds needed to detect a true paired effect with 80% power (paired s.d. 0.0041 average / 0.0053 worst): 0.003 mAP: 17 / 27 seeds; 0.002: 36 / 57; 0.001: 135 / 221; 0.0005: 533 / 877.

Reading: the primary contrast C-B was designed to detect effects of about 0.005, which is larger than any plausible entropy-specific effect. The upper limit of the C-B interval (+0.0021) is smaller than the B-A triplet effect itself (+0.0027).

### H2 Dose-response in triplet signal — verdict: SUGGESTIVE, NOT SIGNIFICANT; entropy adds nothing beyond count

Dose = mean triplets/step over epochs 1..best epoch (computed from the per-epoch `anchors/step triplets/step` log lines; C uses the two reruns' per-epoch means as proxy):

| Arm | triplets/step | anchors/step | valid fraction of anchors | loss-mass T (see H5) | mean mAP |
|---|---|---|---|---|---|
| A | 0 | 0 | | 0 | 0.4381 |
| B' | 2.256 | 4.643 | 0.486 | 0.0027 | 0.4395 |
| C | 2.512 | 4.613 | 0.545 | 0.0035 | 0.4394 |
| B | 3.859 | 7.998 | 0.482 | 0.0023 | 0.4408 |

Tests across arms, within seed (source `h2_dose.py`):

| Test | result |
|---|---|
| per-seed slope of mAP on triplets/step | +0.00068 per triplet/step, 95% CI [-0.00069, +0.00205], t-test p = 0.260, positive in 5/6 seeds |
| seed-fixed-effects OLS, triplets/step | slope +0.00068 (SE 0.00041), p = 0.120 |
| same, anchors/step / loss-mass T | slope +0.00033 (SE 0.00020), p = 0.116 / slope +0.537 (SE 0.432), p = 0.231 |
| permutation of arm labels within seed (100,000), linear dose trend | one-sided p = 0.060, two-sided p = 0.119 |
| Page's L, ordered A < B' < C < B (Monte Carlo) | L = 163, one-sided p = 0.037 |
| Friedman (any difference) | p = 0.241; mean within-seed ranks A / B' / C / B = 1.83 / 2.50 / 2.33 / 3.33 |

What this does and does not establish at n = 6: it does not establish a dose-response (most tests have p between 0.06 and 0.26; the one test below 0.05, Page's L, is one of several trend tests and its ordering was fixed after seeing the provisional table). It is compatible with a small positive effect of the number of triplets (about +0.0026 mAP from A to B). Key reading, ESTABLISHED: C-B' = 0.0000 (s.d. 0.0047), so entropy ranking adds nothing beyond a random subset of equal size. B-A is the only positive-leaning contrast, and it concerns "more triplets", not "better-chosen anchors". The variance pattern (s.d. of mAP A 0.0042, B' 0.0044, C 0.0025, B 0.0012; Levene p = 0.39, Bartlett p = 0.049) falls with dose but is weak evidence at n = 6 per arm (95% interval for the s.d. of B: [0.0007, 0.0028]).

### H3 Is softmax entropy a meaningful gate for ASL-trained sigmoid logits? — verdict: PARTLY. It selects harder, not-Normal images; it does not select multi-finding or tail images

Method: logits recovered as logit(clip(p, 1e-7, 1 - 1e-7)) from `preds.<tag>.npz` of the six C runs (17,270 validation images; also A, B, B' for comparison, which gave the same correlations to +/-0.01). Softmax entropy H over the 30 logits as in `utils_update.py` L72-75. Per-image ASL loss recomputed with the training formula and class weights. Source `h3_prep.py`, `h3_gate.py`. Proxy caveat: the training-time gate sees augmented, seen images in train mode at the live weights; these are held-out images at the selected EMA weights.

Validation of the code reading (ESTABLISHED consistency): a label-only model of the sampler (positives = share >= 1 non-head label, negatives = share no label, bank = 2,048 iid training label signatures) predicts B's triplets/step as 3.862 (validation anchors) / 3.860 (training anchors) against the logged 3.858-3.859, and the fraction of anchors with a valid triplet, 0.4827, equals the fraction of images that have a non-head label, 0.4829 (training 0.4826).

| What H (softmax entropy; C runs, mean over 6) tracks | value |
|---|---|
| Spearman with number of labels | +0.036 (s.d. over runs 0.006) |
| Spearman with number of non-head labels | +0.196 |
| Spearman with per-image ASL loss / per-image AP / top-1 correct | +0.466 / -0.440 / -0.444 |
| Spearman of the Bernoulli entropy with number of labels / loss | +0.290 / +0.601 (H vs Bernoulli entropy +0.794) |
| AUC of H for: has non-head label / >= 2 labels / top-25% loss | 0.623 / 0.530 / 0.595 |
| AUC of H for Normal-only (n = 6,891 = 39.9%) / frontal | 0.366 / 0.387 |
| mean H: Normal-only vs others; non-head vs head-only; frontal vs lateral | 1.340 vs 1.820; 1.851 vs 1.421; 1.512 vs 1.892 (max log 30 = 3.401) |
| within-stratum Spearman(H, loss) for 1 / 2 / 3 / >= 4 labels | +0.644 / +0.415 / +0.415 / +0.314 |
| variance of H explained by label count / by the 30 label indicators | R2 0.000 / 0.207 |

Gate simulation on 200,000 random batches of 8 validation images (entropy of each C run's val logits; mean over the six runs):

| Quantity | simulated | logged (C rerun, epochs 1-5) |
|---|---|---|
| anchors/step | 4.271 | 4.684, 4.576, 4.327, 3.904, 3.401 |
| triplets/step: B / B' / C | 3.858 / 2.078 / 2.371 | B 3.858; B' 2.274, 2.228, 2.097, 1.903, 1.662; C 2.548, 2.494, 2.366, 2.177, 1.973 |
| P(valid triplet): gated / not gated / random same count / all | 0.555 / 0.399 / 0.486 / 0.482 | C 0.544 (ep1), 0.547 (ep3), 0.580 (ep5); B' 0.485; B 0.482 (ratios of logged values) |
| P(non-head label): gated / not gated | 0.556 / 0.399 | |
| mean n labels: gated / not gated | 1.48 / 1.52 | |
| mean per-image ASL loss: gated / not gated / all | 0.0489 / 0.0310 / 0.0405 | |
| step hit rate (>= 1 triplet): B / B' / C | 0.994 / 0.924 / 0.958 | B 0.994; B' 0.941, 0.936; C 0.9665, 0.9635 (ep1-2) |

Reading (ESTABLISHED on the proxy): the gate selects the 53% of images the model is least sure are Normal-only; those carry 58% more per-image loss, which is real "difficulty", but it is not "more findings" and not "tail". That explains the logged 55% vs 48%: the gate's extra valid triplets are the non-head-label images it keeps. P(pass gate | tail-labelled image) = 0.561 vs 0.533 otherwise; per tail class (val positives in brackets): Hydropneumothorax 0.443 (7), Mass 0.583 (49), Pneumothorax 0.462 (71), Subcutaneous Emphysema 0.381 (29), azygos lobe 0.376 (58), calcified densities 0.689 (140), pneumoperitoneo 0.628 (9). The gate also prefers lateral views (higher entropy), while the test set is frontal only (memory note). P(gate | lateral) was not computed (NOT TESTED).

### H4 Bank coverage and tail reach — verdict: SUPPORTED (INFERRED) as a limit on tail signal

Bank 2,048, prevalence from `data/CXRLT_2026_training_filtered.csv` (N = 103,303), groups from `evaluate/class_groups.py` (head 3 / medium 20 / tail 7). Source `h4_coverage.py`.

| Tail class | train pos | prevalence | E[bank pos] | P(bank >= 1 own-class) | P(batch of 8 has >= 1) | P(valid triplet) | own-class share of candidates |
|---|---|---|---|---|---|---|---|
| Hydropneumothorax | 38 | 0.00037 | 0.75 | 0.529 | 0.0029 | 0.789 | 0.457 |
| pneumoperitoneo | 52 | 0.00050 | 1.03 | 0.643 | 0.0040 | 0.849 | 0.428 |
| Subcutaneous Emphysema | 179 | 0.00173 | 3.55 | 0.971 | 0.0138 | 1.000 | 0.084 |
| Mass | 296 | 0.00287 | 5.87 | 0.997 | 0.0227 | 0.999 | 0.441 |
| azygos lobe | 346 | 0.00335 | 6.86 | 0.999 | 0.0265 | 0.999 | 0.694 |
| Pneumothorax | 420 | 0.00407 | 8.33 | 1.000 | 0.0321 | 1.000 | 0.587 |
| calcified densities | 835 | 0.00808 | 16.55 | 1.000 | 0.0629 | 1.000 | 0.500 |

"Own-class share" = fraction of bank candidates (those sharing any non-head label with the anchor) that share the anchor's own tail class; if the closest positive were random among candidates this would be the probability that a tail anchor is matched through its own tail class. The real selection is by embedding distance, so this is only an indicator (UNKNOWN how much closer own-class positives are).

- 2,105 training images (2.038%) carry >= 1 tail label; P(a batch of 8 contains one) = 0.152. They account for about 4.2% of valid triplets in B (2,089 of 49,843 per epoch) and about 4.4% in C (the gate passes them with probability 0.561). 49,843 valid triplets per epoch in B agrees with the logged 3.858 triplets/step x 12,913 steps.
- Coverage is therefore adequate for 5 of 7 tail classes (bank almost always holds an own-class positive) and poor only for the two rarest (38 and 52 positives; 38 and 52 anchors per epoch). The bottleneck is the number of tail anchors (a few hundred per class per epoch), together with H5 (few triplets are active).

Group AP at the selected epoch (mean +/- s.d. over 6 seeds; `eval_result.<tag>.json`, groups via `class_groups.py`):

| Group | A | B | B' | C |
|---|---|---|---|---|
| head (3) | 0.7545 +/- 0.0016 | 0.7534 +/- 0.0031 | 0.7552 +/- 0.0013 | 0.7544 +/- 0.0026 |
| medium (20) | 0.4441 +/- 0.0010 | 0.4430 +/- 0.0017 | 0.4438 +/- 0.0019 | 0.4439 +/- 0.0013 |
| tail (7) | 0.2854 +/- 0.0179 | 0.3004 +/- 0.0071 | 0.2919 +/- 0.0171 | 0.2919 +/- 0.0103 |
| extreme tail (6) | 0.3183 +/- 0.0212 | 0.3354 +/- 0.0083 | 0.3257 +/- 0.0201 | 0.3243 +/- 0.0117 |

Paired by seed (mean [95% t-CI], p, wins; MDE for the tail group): tail C-B -0.0086 [-0.0203, +0.0031] p = 0.119, 1/6, MDE 0.0159; tail C-B' -0.0000 [-0.0160, +0.0159] p = 0.997, 4/6; tail B-A +0.0150 [-0.0077, +0.0378] p = 0.150, 5/6, MDE 0.0308; tail C-A +0.0065 [-0.0061, +0.0191] p = 0.244, 5/6, MDE 0.0171; tail B'-A +0.0065 [-0.0038, +0.0168] p = 0.167, 5/6. Head and medium contrasts all lie within +/-0.0011 with CIs spanning zero (for example head C-B +0.0010 [-0.0038, +0.0058], medium C-B +0.0008 [-0.0019, +0.0035]).

Class-bootstrap vs seed-aware intervals for the tail group: C-A class-bootstrap only [+0.0021, +0.0110] but seed + class double bootstrap [-0.0067, +0.0167]; B-A [-0.0030, +0.0337] vs [-0.0088, +0.0468]. A class-only bootstrap would wrongly suggest a C-A tail effect. Per tail class (C-A, seed-averaged AP, paired t): Pneumothorax +0.014 (t = +4.5), azygos lobe +0.015 (+2.4), calcified densities +0.010 (+2.9), Mass +0.003, Subcutaneous Emphysema -0.003, Hydropneumothorax +0.003 (+0.2), pneumoperitoneo +0.004 (+0.2); B-A Pneumothorax +0.018 (+3.4), Subcutaneous Emphysema -0.025 (-2.4). With 30 classes and several contrasts a few |t| > 2 are expected by chance; Pneumothorax is hypothesis-generating only. The tail s.d. across seeds falls with dose (A 0.0179, B' 0.0171, C 0.0103, B 0.0071); not tested formally.

### H5 Triplet-loss magnitude and optimisation — verdict: loss mass ESTABLISHED tiny; inactivity INFERRED; gradient ratio UNKNOWN

The logged train loss is ASL + 0.1 x triplet (`train_2_v3.py` L871-872, accumulated at L892 before the /accum division at L880), so (train loss of an arm - train loss of A at the same seed and epoch) / 0.1 estimates the mean triplet term per step T (steps without a triplet count as 0). Source `h5_loss.py`; history.json (full precision) for epochs <= best epoch, 4-decimal log values after.

| Epoch | A train ASL | B gap (SE) | B' gap (SE) | C gap (SE) | T: B / B' / C |
|---|---|---|---|---|---|
| 1 | 0.03642 | +0.000198 (0.000015) | +0.000216 (0.000015) | +0.000303 (0.000032) | 0.0020 / 0.0022 / 0.0030 |
| 2 | 0.03443 | +0.000254 (0.000030) | +0.000331 (0.000038) | +0.000402 (0.000056) | 0.0025 / 0.0033 / 0.0040 |
| 3 | 0.03037 | +0.000184 (0.000031) | +0.000283 (0.000048) | +0.000451 (0.000072) | 0.0018 / 0.0028 / 0.0045 |
| 4 | 0.02517 | +0.000233 (0.000049) | +0.000350 (0.000034) | +0.000533 (0.000067) | 0.0023 / 0.0035 / 0.0053 |
| 5 | 0.01975 | +0.000200 (0.000045) | +0.000320 (0.000037) | +0.000467 (0.000049) | 0.0020 / 0.0032 / 0.0047 |

- ESTABLISHED: the gap is 0.5-1.0% (B) and 0.8-2.4% (C) of the ASL loss; epoch 1 is the cleanest estimate, because weights are still nearly identical across arms (T = 0.0020 for B, 95% CI 0.0016-0.0024). Margin 0.5 is 100-250 times larger than T.
- ESTABLISHED: C's triplet term exceeds B' and B (paired train-loss gaps C-B' +0.000087, SE 0.000043, p = 0.097 at epoch 1 and +0.000072, p = 0.051 at epoch 2; C-B +0.000105, p = 0.024 and +0.000148, p = 0.021): gated anchors have harder triplets than random anchors, consistent with H3, and still no mAP gain.
- ESTABLISHED: T does not fall across epochs for B (0.0018-0.0025) and rises for C (0.0030 to 0.0053). The triplet term is not being optimised away.
- The same size at 768 px (TM minus ASL, 2 seeds, epochs 1-4, 4-decimal logs): gaps 0.0001-0.0003, so T about 0.001-0.003, with hit rates of 79.5-82.3% at batch 4.
- Val ASL loss (ESTABLISHED, paired): arms with a triplet are slightly lower than A at later epochs; C: -0.00004 (ep1), -0.00007, -0.00030 (SE 0.00011), -0.00055 (0.00012), -0.00060 (0.00018) (ep5); B' -0.00028 and B -0.00017 at ep5.
- INFERRED activity: T = P(hinge active) x E[hinge | active] (x hit rate, 0.994 for B and 0.9665 for C at epoch 1). If active hinges averaged 0.25, P(active) would be about 0.8% (B) and 1.2% (C) at epoch 1; if they averaged 0.05, about 4% and 6%; at epoch 4 C would be 2.2% and 11%. So at least about 90% (most likely about 95-99%) of mined triplets give no gradient, unless nearly all hinges are tiny, which the data cannot exclude (UNKNOWN; needs D1).
- Mechanism (INFERRED from code, simulation NOT TESTED): `sample_triplets_v13` L303-304 takes the closest positive and the farthest negative ("hardest_*" in the code, easy mining). The minimum distance over tens of positives (2,048 x mean fraction 0.026-0.047 for tail anchors; the negative pool is larger) and the maximum over a much larger negative pool differ by more than 0.5 on the unit sphere even if the embedding carried no label information, as long as distances are spread. The loss is then satisfied by order statistics, independent of structure.
- Gradient scale (analytic): for L2-normalised embeddings the gradient of an active hinge w.r.t. the anchor is (a-p)/|a-p| - (a-n)/|a-n| (norm <= 2, independent of the hinge value), times lambda = 0.1 / (number of triplets in the step) and / 6 for accumulation, then projected by F.normalize. Compared with ASL gradients over 8 x 30 logits, the relative gradient at the shared encoder is UNKNOWN (needs D2).

### H6 `CrossBatchMemoryV2` and bank mechanics — verdict: no defect found by code reading; numerics NOT TESTED

| Item | Finding | Class |
|---|---|---|
| `utils_update.py` L24-25 | comment says "half precision", buffer is `float32` | comment/implementation mismatch, harmless |
| L36 `.half()` then copy into float32 buffer (L40-49) | stored embeddings carry fp16 rounding (relative step 2^-10, error <= 2^-11 per coordinate) | harmless by scale (INFERRED); error vs distance gaps of order 0.1-1 not measured (NOT TESTED) |
| L39-52 ring buffer: write at `ptr`, wrap branch L43-49, `filled = min(size, filled + B)` | correct FIFO by reading; with B = 8 and size 2,048 the wrap branch runs only after a partial batch (each worker's last batch can be < 8) | correct by reading; unit test NOT TESTED |
| L253-258 `sample_triplets_v13` | slices `[:mem_count]` (empty slots never used); step 1 returns before counting anchors, so epoch-1 `anchors/step` is very slightly low | harmless |
| `train_2_v3.py` L852-859 sample, L893 update | bank updated after sampling: no self-match within a step | correct |
| bank not reset between epochs | an anchor in the first 256 steps of an epoch can match its own previous-epoch embedding: 2,048 / 103,303 = 1.98% of images x first 256 of 12,913 steps (1.98%) = about 0.04% of anchors | negligible (arithmetic) |
| staleness | 2,048 slots = 256 micro-batches = 42.7 optimizer steps (accum 6; at 768 px: 512 micro-batches / 12 = 42.7) at lr up to 1e-4; no momentum encoder; entries come from the train-mode, augmented forward pass | drift UNKNOWN (needs D3) |
| L303-304 | `hardest_pos`/`hardest_neg` are closest positive / farthest negative (easy mining), as the file's own `_mine_pair` docstring says | misleading names, by design |
| L286-291 | positives need >= 1 shared non-head label; negatives share no label of any kind; Normal-only and head-only anchors can never get a triplet (51.7% of validation images) | by design, bounds the signal |
| `train_2_v3.py` L841-850, L609, L861-869 | embedding = L2-normalised mean of the 30 decoded frozen-query vectors; only the anchor has gradient; `TripletMarginLoss(margin 0.5, p 2)` mean over the step's valid triplets; steps without a triplet add 0 | standard XBM-style; no wiring bug |

Data-order hypothesis (new, NOT TESTED): shards are written in `train_fold_1.csv` row order with no shuffle (`convert_dataset.py` L80-110; sorted file names put images of one study/patient on adjacent rows, e.g. rows 1-2 and 3-4 of the CSV), and each worker reads its shards sequentially through a 400-sample buffer (`train_2_v3.py` L1083-1088; worker w gets shards w and w+8 of 11). A batch and the 2,048 bank therefore come from windows of consecutive rows, so same-study sibling views (identical labels, near-identical anatomy) may often be in the bank and be the closest positive. That would make positives trivially close and hinges inactive (consistent with H5). The size of the effect needs the data-order simulation D5.

### H7 Epoch selection and schedule — verdict: SUPPORTED as a contributing factor (INFERRED)

Mean val mAP by epoch (logs; n = 6 per arm and epoch except B' epoch 5 n = 5):

| Epoch | A | B | B' | C |
|---|---|---|---|---|
| 1 | 0.4320 | 0.4324 | 0.4335 | 0.4332 |
| 2 | 0.4381 | 0.4399 | 0.4393 | 0.4394 |
| 3 | 0.4341 | 0.4366 | 0.4356 | 0.4363 |
| 4 | 0.4259 | 0.4264 | 0.4272 | 0.4289 |
| 5 | 0.4169 | 0.4159 | 0.4194 | 0.4190 |

Best epoch is 2 in 20 of 24 runs (3 for B s1024 and C s123, 1 for B' s1024). LR at the end of epoch 2 is 8.95e-05 (89.5% of peak 1e-4; C s42 `history.json`). Runs stop at epoch 5-6 (patience 3), before the cosine decay of an 8-epoch horizon; at 768 px the recipe is `--epochs 4 --sched-epochs 8` (LR about 50% of peak at the end). Evaluated weights are the EMA (decay 0.999, about 1,000 optimizer steps = 0.46 epoch of 2,152 steps).

Paired contrasts (mean [95% CI], p_t, wins): at epoch 2: C-A +0.0013 [-0.0019, +0.0045] p = 0.331; B-A +0.0018 p = 0.508; C-B -0.0005 p = 0.833. Mean of epochs 3-5: C-A +0.0025 [-0.0013, +0.0063] p = 0.152; C-B +0.0018 [-0.0001, +0.0036] p = 0.059, 5/6; B'-A +0.0024 p = 0.224; B-A +0.0007 p = 0.673. Single epochs: the only nominal hit is C-B at epoch 5, +0.0030 [+0.0001, +0.0060], p = 0.044 (25 epoch-contrast tests; Bonferroni threshold 0.002). No arm leads at any epoch after correction.

Selection bias (max over epochs) is small and not equal across arms: best-epoch mean minus epoch-2 mean is 0.0000 (A), +0.0009 (B), +0.0002 (B'), 0.0000 (C), so the B-A contrast is +0.0027 with best-epoch selection and +0.0018 at fixed epoch 2.

Reading: the arms differ most at epochs 3-5 (lower val loss, slightly higher mAP for the gated arms), exactly when the model overfits (A val loss 0.0404 at epoch 2 to 0.0511 at epoch 5), but early stopping selects epoch 2 on a metric where they do not differ. A regulariser that slows overfitting cannot raise the best-epoch value. Whether a late/low-LR phase would show a benefit is UNKNOWN (no run reaches it).

768 px (2 seeds, 4 epochs; logs). Per-epoch mAP, mean of two seeds, ASL / MoE / TM: epoch 1 0.4367 / 0.4387 / 0.4386; epoch 2 0.4541 / 0.4563 / 0.4559; epoch 3 0.4558 / 0.4546 / 0.4575; epoch 4 0.4516 / 0.4493 / 0.4547. SWA (epochs 2-4, no TTA) mAP per seed (s42 / s1024): ASL 0.4594 / 0.4574, MoE 0.4560 / 0.4585, TM 0.4600 / 0.4582 (identical to the result report). TM leads at epochs 3-4 by 0.0017 and 0.0031 over ASL, with 2 seeds this is not interpretable.

### H8 Prediction-level effect — verdict: between-arm shifts are no larger than between-seed shifts (ESTABLISHED); small calibration/late-loss shifts exist

Source `h8_pred.py`, `h8_ens_perm.py` (all 26 runs, same 17,270 images).

| Pair category | n pairs | RMS prob diff | RMS logit diff | RMS per-class-AP diff |
|---|---|---|---|---|
| same seed, different arm | 36 | 0.0342 | 0.438 | 0.0208 |
| same arm, different seed | 60 | 0.0359 | 0.447 | 0.0211 |
| different arm and seed | 180 | 0.0359 | 0.448 | 0.0230 |
| same arm and seed (C rerun) | 2 | 0.0316 | 0.410 | 0.0262 |

The identical-config rerun is almost as far from its twin as two different seeds: chaotic run noise dominates, and the arm is invisible in single-run predictions. (The logit RMS within arm C across seeds, 0.621 vs 0.36-0.41 for the other arms, is probably inflated by the runs whose selected epoch differs; not tested.)

Exact permutation test on arm-mean predictions (6 vs 6 seeds, 462 splits of the 12 runs): RMS of the difference of mean probabilities, observed vs null mean: A vs C 0.01382 vs 0.01461 (p = 0.965); A vs B 0.01370 vs 0.01462 (p = 0.974); B vs C 0.01372 vs 0.01548 (p = 0.985); B' vs C 0.01457 vs 0.01485 (p = 0.600); A vs B' 0.01408 vs 0.01386 (p = 0.277). The arm-mean difference is no larger than a random split would produce.

Six-seed probability-average ensembles (no TTA): mAP A 0.4472, B 0.4511, B' 0.4486, C 0.4505 (single-run means 0.4381 / 0.4408 / 0.4395 / 0.4394; ensemble tail AP 0.3045 / 0.3210 / 0.3104 / 0.3157). Permutation p-values of the ensemble differences (mAP; tail): C-A +0.0032 (0.268; +0.0112, 0.318); B-A +0.0039 (0.056; +0.0165, 0.041); B'-A +0.0014 (0.647; +0.0059, 0.647); C-B -0.0006 (0.777; -0.0053, 0.537); C-B' +0.0018 (0.558; +0.0053, 0.665); B-B' +0.0025 (0.392; +0.0106, 0.366). The two nominal hits (B-A tail 0.041, B-A mAP 0.056) are among 24 tests and are not significant after correction.

Other metrics at the selected epoch (paired, eval_result): mAUC C-A -0.0005 [-0.0029, +0.0019]; B'-A +0.0010 [-0.0002, +0.0021] p = 0.078; mF1 C-A +0.0033 [-0.0007, +0.0074] p = 0.090 (5/6 wins), B-A +0.0029. mECE at the selected epoch is not usable for contrasts because the selected epoch differs by run (epoch-3 runs have mECE about 0.015 lower), giving C-A +0.0001 [-0.0065, +0.0067]. At matched epochs, the per-epoch mean mECE is higher for every triplet arm than for A at each of epochs 1-5 (C-A +0.0019, +0.0027, +0.0027, +0.0023, +0.0017; B-A +0.0011 to +0.0019; B'-A +0.0011 to +0.0024; arm s.d. 0.0004-0.0014); the 768 px report found the same sign (TM-ASL +0.0024, image-bootstrap CI). A paired test of this was NOT TESTED.

### H9 Effect budget and the published entry — verdict: the objective accounts for at most about 4% of the Stage-1-to-Stage-2 gain

| Step | mAP (internal val, 17,270 images) | Source |
|---|---|---|
| Published Stage 1 (384 px, training-time) | 0.385 (mF1 0.395, mAUC 0.875, mECE 0.134) | `ISBI2026.pdf` Table 1; repo `history.json` of `Model_20260119_062652` gives 0.3849 (`docs/result/2026-10-08-stage1-original-baseline.md`) |
| Same Stage-1 checkpoint re-evaluated at 512 px | 0.3688 (head 0.7051 / medium 0.3765 / tail 0.2028) | `eval_result.stage1.json`, job 47073, same report |
| Published Stage 2 (MoE) | 0.405 (mF1 0.402, mAUC 0.882, mECE 0.130) | `ISBI2026.pdf` Table 1 |
| ASL-only Stage 2 at 512 px (A) | 0.4381 +/- 0.0042 | 6 seeds, `history.json` |
| + triplet, all anchors (B) / random gate (B') / entropy gate (C) | 0.4408 / 0.4395 / 0.4394 | same |
| 6-seed same-arm ensemble at 512 px: A / B / B' / C | 0.4472 / 0.4511 / 0.4486 / 0.4505 | `preds` |
| 768 px single model, SWA, no TTA, 2 seeds: ASL / MoE / TM | 0.4584 / 0.4573 / 0.4591 (means of the two seeds) | logs / result report |
| 768 px, flip TTA, SWA, 2-seed ensemble: ASL / MoE / TM | 0.4625 / 0.4668 / 0.4659 | `docs/result/2026-10-11-moe-matched-eval.md` |
| TM greedy trio (selected on val for TM) / same members for MoE | 0.4722 / 0.4697 | same report |
| Published official test (Stage 2) | mAP 0.4599, mF1 0.4504, mAUC 0.8827, mECE N/A | `ISBI2026.pdf` Table 3 |

Computed steps: Stage-1 at 512 px to A is +0.0693; A to B +0.0027 (3.9% of that step), A to C +0.0014 (2.0%), C-B -0.0014. A at 512 px to ASL at 768 px (SWA, drop-path 0.1, 4-epoch recipe, different code path) is +0.0203 (0.4584 - 0.4381), and ASL, MoE and TM are within 0.002 of each other at 768 px. Notes: the A-to-ASL-768 step bundles resolution, drop-path, SWA, batch 4 x accum 12 and v3 vs v5; the old-recipe 0.414 +/- 0.002 5-seed entropy result and the memory-note decomposition (fixes + recipe +0.025, 768 px +0.017, drop-path + SWA +0.003-0.005, flip +0.005, ensemble +0.007) come from the project brief and memory, not re-verified here.

**Relation to the previous publication (read directly from `docs/paper/isbi_submit/figures/ISBI2026.pdf`, pages 3-4, and `docs/pdf/Entropy.pdf` p.9 rendered as an image).** The paper's Table 1 (internal validation, "103,000 samples for model training and 17,000 for internal validation") lists, in the column order mAP / mF1 / mAUC / mECE: Stage 1 0.385 / 0.395 / 0.875 / 0.134; Stage 2 0.405 / 0.402 / 0.882 / 0.130. Table 2 (Stage 1 on the development set): 0.505 / 0.415 / 0.874 / 0.776. Table 3 (test set, Stage 2): 0.4599 / 0.4504 / 0.8827 / N/A. Stated setup: ConvNeXt-B, 384 x 384, Adam lr 4e-5, weight decay 0.01; Stage 1 trained end-to-end for 10 epochs; Stage 2 adds sparse MoE (4 experts, top-2, aux weight 5e-3) for 2 epochs with "most backbone parameters frozen" (only the ConvNeXt blocks of stages 4+5 and the decoder active); a single Tesla P40; the dataset paragraph speaks of "36 long-tailed classes" (this repo uses 30 target labels, so mAP comparability of the label set is UNKNOWN).
- The ASL-only 512 px control (0.4381) is about +0.033 above the published internal Stage-2 value (0.405) and +0.053 above the published Stage 1 (0.385). It is **not** a match, and the two are **not** like-for-like: resolution (512 vs 384 stated), trainable parameters (all vs stages 4+5 + decoder), optimiser and LR (AdamW 1e-4 with warm-up, cosine, EMA 0.999 vs Adam 4e-5), epochs, fixed class-weight order, and possibly the label set.
- The data therefore **contradict** "ASL-only already reproduces the earlier entry's level" if that is read as equality (it exceeds it by about 0.033). They **say nothing** about what an ASL-only Stage 2 would score under the published frozen-stage 384 px recipe (no such run exists; NOT TESTED). They **do support** the weaker statement that, within the present recipe, ASL-only reaches the same level as the triplet and entropy variants, so the improvement over 0.405 comes from the recipe and not from the objective.
- Provenance of the published 0.405 is ambiguous: the project slide `Entropy.pdf` p.9 shows Stage 2 at 384 px = 0.392 / 0.373 / 0.881 and at 512 px = 0.405 / 0.395 / 0.884 (Stage 1 0.385 / 0.395 / 0.875 / 0.134 in both tables), whereas the paper says 384 x 384 and lists 0.405 / 0.402 / 0.882 / 0.130. The resolution behind 0.405 is UNKNOWN.

### H10 Code-path differences between compared systems — verdict: none found that masks or fakes a difference; one reproducibility caveat (INFERRED)

| Aspect | 512 px arms (all v3) | 768 px: TM (v3) vs ASL/MoE (v5) |
|---|---|---|
| Recipe args | identical except `--triplet-lambda`, `--anchor-rule` (logs' `args:` lines) | identical except TM adds `--memory-size 2048 --metric-embed query --head-classes 5 --triplet-lambda 0.1`; ASL/MoE add `--no-triplet --arch {convnext2,moe}`; all: `--batch-size 4 --accum-steps 12 --lr 1e-4 --epochs 4 --patience 2 --sched-epochs 8 --drop-path-rate 0.1 --swa-last-k 3 --class-weight-order shard`, same Stage-1 checkpoint |
| ASL control | v3 with lambda 0: bank and mining still run, gradient exactly zero (`loss = cls + 0 x triplet`) | v5 `--no-triplet`: no embedding, bank or mining |
| Sampler | `sample_triplets_v13`, anchor rule per arm | v5 TM path calls the same sampler without `anchor_rule/generator/stats` (default "mean", same gate) |
| Data / aug / loss / optimiser / EMA / SWA / eval | same code (diff of `train_2_v3.py` vs `train_2_v5.py` read to L1304 of v3; v5 additionally returns an all-zero prior tensor and has Markov/MoE branches) | same |
| In-worker sample shuffle | `wds...shuffle(400)` without seed (L1083-1088) | same |

The v3/v5 diff shows no change to augmentation, loss, optimiser, schedule, EMA, SWA or evaluation lines (remaining lines after L1304 of v3 not read). RNG-stream equality between arms was not verified. Reproducibility (INFERRED from code reading, empirically unverified, assuming the jobs used the venv's webdataset 1.0.2): `train_2_v3.py` L1054-1057 seeds the shard order from `--seed`, but webdataset's `_shuffle` (installed 1.0.2) builds `random.Random(int((os.getpid() + time.time()) * 1e9))` when no `rng`/`seed` is passed, and `FluidInterface.shuffle(size, **kw)` forwards nothing; `seed_worker` (L1391-1395) seeds only the global `random`. The comment at L1068-1074 ("reproducible via seed_worker") and the plan's line "arms with the same seed see the same data order" (`2026-10-08-entropy-ablation-isbi-submit-plan.md` L45) therefore need checking. This agrees with H1 and H8 (weak seed pairing, run noise as large as seed noise) but does not prove it. It applies equally to every arm, so it cannot fake a difference; it removes the pairing advantage.

### H11 Other observations
- ESTABLISHED: the "7.6% of steps yield a usable triplet" figure does not describe these runs. Step hit rates (logs): B 99.4%; C 96.5, 96.3, 95.6, 94.4, 92.9% (epochs 1-5); B' 94.1, 93.6, 92.6, 90.0, 86.4%; TM at 768 px with batch 4: 79.5-82.3%. The per-anchor fraction (48-58%) is set by the 48.3% of images that have a non-head label.
- ESTABLISHED: Normal-only images (39.9% of validation, 40.2% of training are Normal) and any head-only images (51.7% of validation) can never be anchors with a positive; they only serve as negatives. The `--head-classes 5` set (Normal, pleural effusion, cardiomegaly, Support Devices, aortic elongation) differs from the 3-class head of `class_groups.py` (Normal, pleural effusion, cardiomegaly).
- ESTABLISHED: `--class-weight-order shard` appears in the logs I opened (C s42, C s86, C-rerun s86 with weights printed, and all six 768 px runs). NOT CHECKED individually for the other 18 runs.
- ESTABLISHED by code: unreadable images are dropped silently (`decode_and_transform` returns None, `.select(lambda x: x is not None)`, L402-404, L1087); the count of 24 was not verified here and is immaterial (0.02%).
- ESTABLISHED: the gate lets lateral views through more often (mean entropy 1.892 vs 1.512), while the test set is frontal only.
- Trap for later analyses: `eval_result` mECE and `best_metric` mix selected epochs (best epoch 1, 2 or 3), so per-epoch comparisons must use the logs.

## 3. Decisive diagnostics, not run

| # | Measurement | Input | Runtime | CPU? | Confirms / refutes |
|---|---|---|---|---|---|
| D1 | Embedding geometry and hinge-active fraction. Forward 2,048 "bank" + 512 anchor validation images; compute the unit-sphere distance distributions, d_ap (closest positive by the sampler's rule) and d_an (farthest negative), the fraction with d_an - d_ap < 0.5, and the same with labels permuted among images (null geometry). Hook `model.head.last_h.mean(1)` then normalise. | C `train/checkpoint_Triplet_3/seedsweep/seed_42/Model_run/model_best.pth`; A `.../ablation512/asl/seed_42/Model_run/model_best.pth`; B `.../ablation512/nogate/seed_42/Model_run/model_best.pth`; t = 0 reference `train/checkpoint3/Model_20260119_062652/model_best.pth` (512 px) | about 5 min on one A6000/Ada GPU (extrapolated from the >= 22 min the 17,270-image eval jobs 51773/51774 had used when last seen; not measured) | No (131.5M-parameter ConvNeXt at 512 px; expected hours on CPU, not measured) | Active fraction < 5% in A, B and C and also under permuted labels would confirm H5 "satisfied by order statistics"; an active fraction >= 20% would refute it |
| D2 | Gradient-norm ratio |grad of 0.1 x triplet| / |grad of ASL| at the last ConvNeXt stage and decoder, over 200 training batches at the Stage-1 init and at the epoch-2 checkpoint | Stage-1 checkpoint above, plus the epoch-2 EMA of a C run | about 15 min GPU | No | Ratio < 0.05 confirms "no-op"; > 0.3 means the loss is active and the null lies elsewhere |
| D3 | Bank staleness: re-embed the 2,048 bank images with the current weights at steps 0, 500, 2,000 of a fine-tune and measure cosine drift to the stored embeddings | training run with logging (needs a code change by the main session) | about 10 min GPU per probe | No | Drift comparable to the positive-negative gap would confirm that stale entries add noise |
| D4 | Null-geometry order-statistics simulation: hinge-active fraction vs spread of pairwise cosine, with candidate counts taken from H4 (positives about 2,048 x q_pos, negatives about 2,048 x q_neg) | numpy only | minutes | Yes | Shows which D1 result would mean "inactive without any label structure" |
| D5 | Data-order / sibling simulation: model the 11 sorted shards, `split_by_worker`, 400-sample buffer, batch 8, bank 2,048; count anchors that have a same-study or same-patient sibling in the bank (reads only ImageID, StudyID, PatientID of train-fold images from the PadChest CSV, never labels or dev/test) | `data/train_fold_1.csv`, PadChest CSV (allowed columns) | minutes | Yes | A high sibling rate (> 20%) would show trivially close positives; plus a 10-second check that `.shuffle(400)` is not reproducible across processes with the same global seed |
| D6 (training) | Hard/semi-hard mining + lambda 1.0, and a Bernoulli-entropy gate arm, paired with A over >= 17 seeds for a 0.003 effect | needs code and 7-12 h per run | days | No | Tests whether any triplet configuration moves mAP above the 0.0043-0.0076 MDE; not cheap, run only after D1/D2 |

## 4. What the paper may and may not say

May say:
- At 512 px, with the recipe and Stage-1 checkpoint fixed, 6 seeds paired by seed, adding the entropy-gated triplet + memory-bank objective to ASL gave no measurable mAP gain over ASL alone (C-A +0.0014 [-0.0018, +0.0045]), over the same objective with every image as an anchor (C-B -0.0014 [-0.0048, +0.0021]) or over a random gate of equal size (C-B' 0.0000 [-0.0049, +0.0049]). The experiment could detect paired effects of about 0.004-0.008 mAP (80% power); the data exclude an entropy-specific gain above about 0.002 (C-B) to 0.005 (C-B').
- At 768 px (2 seeds, v3 vs v5 code paths), Triplet Memory, a matched sparse MoE and ASL alone are indistinguishable (flip-TTA paired TM-MoE -0.0008 [-0.0065, +0.0043]).
- The entropy gate (softmax over the 30 logits) retains the images the model is least sure are Normal-only: 55% of its anchors yield a valid triplet vs 48% for random anchors, with 58% higher mean per-image ASL loss on the validation proxy. It does not select multi-finding or tail images.
- The triplet term is small in the training loss (0.002-0.005 against margin 0.5; 0.5-2.4% of the loss), and the arms differ only in later epochs and slightly in calibration (higher per-epoch mECE by 0.002-0.003 for the triplet arms; formal test not run).
- Most of the improvement over the earlier Stage 2 comes from the recipe, not the objective: the ASL-only 512 px control already reaches 0.4381 +/- 0.0042.

May not say:
- That entropy gating (or the triplet term) improves mAP, mF1, mAUC or tail AP, or beats MoE or ASL, or that the 0.439 to 0.472 progression is due to entropy.
- That "no effect" is proven: only effects above about 0.002-0.005 are excluded. The tail-group differences (B-A +0.0150, C-A +0.0065) are within noise.
- That the gate picks "uncertain, multi-finding" or tail images; that same-seed arms saw identical data order; that only about 7.6% of steps yield a triplet (hit rate is 93-99% at 512 px, 80% at 768 px).
- That the ASL-only control reproduces, matches or equals the earlier entry. Cite the published numbers exactly (Stage 1 mAP 0.385 / mF1 0.395 / mAUC 0.875 / mECE 0.134; Stage 2 0.405 / 0.402 / 0.882 / 0.130 on internal validation; test 0.4599 / 0.4504 / 0.8827 as mAP / mF1 / mAUC) and state that the comparison is not like-for-like (384 px stated, Adam 4e-5, stages 4+5 and decoder only, 2 epochs, MoE; a possible 30 vs 36 label difference; the slide `Entropy.pdf` p.9 attributes 0.405 to 512 px). The data show the 512 px ASL control is about 0.033 above the published internal Stage-2 value; they do not show a match.
- Any leaderboard claim from internal validation numbers (the validation split is not patient-disjoint, per memory note `data-split-patient-leakage`).

## 5. Caveats
- Evidence status: all 26 prediction dumps are present; every prediction-based number above uses them (the Spearman comparison across arms in H3 used only 4 of the 6 B' runs; the headline gate analysis used the six C runs). The coordinator's note that `analysis/out/entropy_ablation.json` may be rewritten was respected (not an input).
- n = 6 seeds at 512 px, n = 2 at 768 px; the 768 px arms compare v3 (TM) with v5 (ASL, MoE), same recipe, different code path.
- Entropy, anchors and bank content are proxied by validation images at the selected EMA weights; training-time quantities may differ (gate anchors/step 4.27 simulated vs 4.58 logged at epoch 2).
- Per-epoch triplets/step of the six original C runs are proxied by the two reruns.
- The unseeded-shuffle finding comes from reading installed library code; not run.
- Statistical tests are numerous; p-values near 0.04-0.06 (Page L, ensemble B-A, epoch-5 C-B) are not corrected and should be read as descriptive.
- Pre-fix flags: all arms use `--metric-embed query` (log `v3:` line) and `--class-weight-order shard` (checked on the logs listed in H11); the 7.6%/13% utilisation discrepancy and the 384-vs-512 confound of `paper-revision-confounds` do not apply to the 512 px ablation, which is matched.

## Not measured / NOT TESTED
- H6 numerics: fp16 round-trip error vs distance gaps; ring-buffer wrap-around unit test; staleness drift.
- Data-order sibling effect (H6) and the shuffle reproducibility check (H10).
- Null-geometry order-statistics simulation (H5); hinge-active fraction and gradient ratio (need GPU).
- Formal paired test of the per-epoch mECE shift (H8); P(gate | lateral) (H3); `class-weight-order` flag in the remaining 18 `args:` lines (H11); the tail of the v3/v5 diff after L1304 (H10).
- No head/medium/tail evaluation for the 768 px arms (existing per-class files are for the 512 px ablation only).

## Notable
- First like-for-like evidence that the entropy gate is not even a better selector than random at equal anchor count (C-B' = 0.0000), with an explicit upper bound on the entropy-specific gain (C-B <= +0.0021).
- The logged triplet term is only 0.002-0.005 per step; "triplet utilisation 7.6%" in memory/paper does not apply to these recipes (93-99% hit rate).
- The softmax-entropy gate selects "not Normal-only", it does not select tail or multi-finding images; the gate prefers lateral views.
- Possible reproducibility issue: the in-worker `shuffle` is not seeded by `--seed` (INFERRED), which weakens the plan's seed-pairing assumption and may explain "non-deterministic beyond seed" in memory (v5 rerun).
- Published 0.405 has ambiguous provenance (384 px in the paper, 512 px on the slide) and the paper mentions 36 classes; the ASL-only 512 px control is +0.033 above it.

## Next steps
1. Run D1 and D2 (minutes of GPU) before any new training; run D4 and D5 on CPU now.
2. Have the main session verify and, if confirmed, seed the in-worker shuffle (not done here, no files edited).
3. If a positive result is still wanted, pre-register a design with >= 17 paired seeds for a 0.003 effect, or a configuration that D1/D2 show to be active (hard mining, higher lambda); report C-B and C-B' with intervals either way.
4. Write the formal results report `docs/result/<date>-entropy-ablation-512.md` (plan Step 4) and link it to this document.
