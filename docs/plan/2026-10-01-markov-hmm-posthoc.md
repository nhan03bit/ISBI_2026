# Post-prediction patient HMM (the rebuilt Markov layer) — 2026-10-01

**Type:** offline analysis on cached probabilities, no training. **Internal validation only, a longitudinal simulation; leaderboard effect 0 by construction.**
**Artifacts:** `analysis/out/markov_hmm/` (`run.log`, `results.json`, `oof_seed0.npz`, `model_H4_{s42,s1024,ens}.pt`). Canonical run: Slurm job 49445, COMPLETED, 6063 s, node havok.
**Code:** `analysis/markov_hmm.py`, `analysis/markov_hmm_check.py`, `analysis/patient_meta.py`.
**Links:** design and pre-specified rule [2026-10-01-markov-hmm-redesign-plan.md](2026-10-01-markov-hmm-redesign-plan.md); documentation plan [2026-10-01-markov-hmm-docs-plan.md](2026-10-01-markov-hmm-docs-plan.md); metadata EDA [2026-10-01-padchest-metadata-eda.md](2026-10-01-padchest-metadata-eda.md); A/B verdict [2026-10-01-markov-ab-verdict.md](2026-10-01-markov-ab-verdict.md) (training numbers: [../result/2026-10-01-markov-ab-training.md](../result/2026-10-01-markov-ab-training.md)); background [2026-09-30-patient-markov-research.md](2026-09-30-patient-markov-research.md); architecture [markov-layer-architecture.md](markov-layer-architecture.md). Paper: `docs/paper/markov/paper.tex`, §V-D.

## Question
Does a Markov model placed after the frozen classifier, using the patient's previous state and metadata as a prior, add to the image probabilities? How does it compare with feeding the same prior into the network (B-early), and where does the gain come from?

## Data and method
- **Emission.** Frozen ConvNeXt/ML-Decoder flip-TTA probabilities: s42 (job 49000, SWA, 768 px), s1024 (job 48999), ens (greedy 3-model). Calibrated with one shared temperature and per-class intercepts, fitted on held-out patients.
- **Prior.** Fitted on train-fold (patient, date) states only (74,782 states, 24,180 consecutive pairs, 50,602 first states; 1 ambiguous study excluded). Initial pi0(m) and transition pi(y_{t-1}, m, gap). m = sex, age (restricted cubic spline), acuity. Patient-grouped 5-fold CV chose l2 = 1e-4 and l2_cross = 1e-3.
- **Fusion.** z = calibrated logit + w (logit q − logit prevalence), w_h with an earlier state, w_0 without; weights on the grid {0, 0.25, ..., 1.5}. **Observed** mode: previous state = previous report's labels. **Filtered** mode: forward pass over earlier images only.
- **Arms.** E emission; H0 pooled chain, w = 1, no calibration (= B-late); H1 + calibration and fitted w_h; H2 + gap; H3 + metadata; **H4 + cross-class edges (primary)**; H5 + within-state MRF; H4-acq + acquisition fields; H4-F filtered mode (s42 only). Comparators B-early (v5 network fed the prior) and B-early0 (same network, zero prior), s42 and s1024 only.
- **Estimator.** Calibration and weights are cross-fitted on two patient halves. mAP is computed within each half and averaged (`cf_map`), then averaged over 3 split seeds. Delta [CI] = paired patient-clustered bootstrap (300 resamples) **on split seed 0**. Pooling halves fitted with different calibrations mixes scales; during this work that artefact was −0.006 on no-prior rows from calibration alone, which is why the per-half estimator is used. Absolute values therefore differ from the pooled estimator of the A/B report; compare only within one report.
- **Subsets.** full 17,270; has_prior 5,759; **primary = has_prior & frontal 4,061**; no_prior 11,511; has_prior & study_not_in_train 2,814; patient_unseen 3,966; has_prior & history val-only 248 (the only subset where filtered-mode history emissions are all out-of-sample).
- **Built-in checks (all passed, run.log):** unpenalised logistic = unsmoothed counts (max |diff| 1.19e-05 over 29 classes); filtered mode with one-hot beliefs = observed mode; H0 reproduces B-late (has_prior 0.5299, full 0.4752 on the ensemble).

## Findings

### 1. Ladder (Δ vs E, mean [95% CI], bootstrap on split seed 0)
Absolute E mAP: s42 full 0.4652 / primary 0.5020 / has_prior 0.4882 / no_prior 0.4543; s1024 0.4635 / 0.4937 / 0.4833 / 0.4551; ens 0.4737 / 0.5100 / 0.4977 / 0.4635. E mECE 0.1352 (s42, s1024), 0.1337 (ens). H4 mECE 0.0068 / 0.0069 / 0.0068.

**s42**

| Arm | full | primary | has_prior | no_prior |
|---|---|---|---|---|
| H0 | +0.0090 [-0.0033, +0.0242] | +0.0354 [+0.0150, +0.0583] | +0.0463 [+0.0236, +0.0681] | +0.0000 |
| H1 | +0.0222 [+0.0135, +0.0353] | +0.0477 [+0.0298, +0.0664] | +0.0541 [+0.0389, +0.0708] | +0.0000 |
| H2 | 0.4849 (abs; no CI) | 0.5480 (abs; no CI) | 0.5380 (abs) | 0.4543 (abs) |
| H3 | +0.0160 [+0.0093, +0.0239] | +0.0378 [+0.0226, +0.0516] | +0.0429 [+0.0312, +0.0545] | -0.0009 [-0.0018, -0.0001] |
| **H4** | +0.0152 [+0.0089, +0.0226] | **+0.0375 [+0.0228, +0.0513]** | +0.0423 [+0.0324, +0.0530] | -0.0009 [-0.0018, -0.0001] |
| H5 | 0.4812 (abs) | 0.5411 (abs) | 0.5305 (abs) | 0.4540 (abs) |
| H4-acq | +0.0174 [+0.0111, +0.0247] | +0.0377 [+0.0230, +0.0515] | +0.0434 [+0.0321, +0.0553] | -0.0001 [-0.0026, +0.0023] |
| H4-F | +0.0032 [-0.0016, +0.0083] | +0.0092 [-0.0013, +0.0181] | +0.0134 [+0.0071, +0.0195] | -0.0009 [-0.0018, -0.0001] |
| B-early | +0.0191 [+0.0129, +0.0257] | +0.0339 [+0.0152, +0.0554] | +0.0452 [+0.0338, +0.0565] | +0.0014 [-0.0045, +0.0130] |
| B-early0 | -0.0027 [-0.0092, +0.0047] | -0.0071 [-0.0197, +0.0013] | -0.0045 [-0.0182, +0.0033] | +0.0014 [-0.0045, +0.0130] |

**s1024**

| Arm | full | primary | has_prior | no_prior |
|---|---|---|---|---|
| H0 | +0.0115 [-0.0003, +0.0274] | +0.0425 [+0.0222, +0.0669] | +0.0563 [+0.0386, +0.0754] | +0.0000 |
| H1 | +0.0259 [+0.0165, +0.0396] | +0.0527 [+0.0346, +0.0752] | +0.0629 [+0.0464, +0.0804] | +0.0000 |
| H2 | 0.4837 (abs) | 0.5458 (abs) | 0.5377 (abs) | 0.4551 (abs) |
| H3 | +0.0167 [+0.0101, +0.0250] | +0.0435 [+0.0290, +0.0612] | +0.0480 [+0.0375, +0.0602] | -0.0005 [-0.0015, +0.0005] |
| **H4** | +0.0158 [+0.0090, +0.0245] | **+0.0424 [+0.0279, +0.0592]** | +0.0465 [+0.0359, +0.0587] | -0.0005 [-0.0015, +0.0005] |
| H5 | 0.4802 (abs) | 0.5396 (abs) | 0.5299 (abs) | 0.4549 (abs) |
| H4-acq | +0.0177 [+0.0117, +0.0249] | +0.0431 [+0.0287, +0.0599] | +0.0477 [+0.0378, +0.0576] | +0.0011 [-0.0025, +0.0045] |
| B-early | +0.0185 [+0.0106, +0.0246] | +0.0436 [+0.0254, +0.0632] | +0.0495 [+0.0391, +0.0630] | +0.0002 [-0.0058, +0.0062] |
| B-early0 | +0.0004 [-0.0045, +0.0049] | -0.0000 [-0.0074, +0.0091] | -0.0013 [-0.0070, +0.0044] | +0.0002 [-0.0058, +0.0062] |

**ens** (no B-early, no H4-F): H4 +0.0129 [+0.0068, +0.0222] full, +0.0364 [+0.0209, +0.0517] primary, +0.0417 [+0.0300, +0.0555] has_prior, -0.0009 [-0.0020, +0.0001] no_prior. H1 +0.0202 / +0.0451 / +0.0526 / 0.0000. H3 +0.0148 / +0.0393 / +0.0447 / -0.0000. H4-acq +0.0140 / +0.0361 / +0.0428 / +0.0005. H0 +0.0030 [-0.0083, +0.0167] / +0.0305 / +0.0418 / 0.

Reading notes.
- H2 and H5 have no bootstrap CI in the run; they are listed as absolute 3-split mean mAP. Derived differences vs E (3-split means, not comparable to the bootstrap-mean Δ at the 0.001 level): H2 primary +0.0460 (s42), +0.0521 (s1024); H5 primary +0.0391, +0.0459 (H5 = H4 exactly, see §6).
- Δ in the table is the bootstrap mean on split seed 0, so it differs slightly from the difference of 3-split means (e.g. H4 s42 primary: 0.5411 − 0.5020 = +0.0391 derived vs +0.0375 bootstrap).
- **Simplest is best.** H1 (calibration + w_h, no gap, no metadata, no cross edges) is numerically the highest on every has_prior subset, at all three emissions (primary: H1 0.5511 vs H4 0.5411 at s42, 0.5473 vs 0.5396 at s1024, 0.5550 vs 0.5486 at ens; derived differences −0.0100, −0.0077, −0.0064). Each added term (gap, metadata, cross edges) lowers mAP a little. No paired H1 − H4 bootstrap was run, so "not better" is the claim, not "significantly worse". H1 also leaves no_prior rows exactly unchanged (w_0 = 0).
- Calibration alone (H0 → H1) adds +0.0127 / +0.0090 / +0.0125 mAP on primary at the three emissions (derived from 3-split means; s42 0.5384 → 0.5511). mECE falls from 0.1293 to 0.0074.

### 2. Transition-model held-out diagnostics (run.log §[1]; patient-grouped 5-fold CV on train pairs)

| Nested prior | macro log-loss | macro AUROC |
|---|---|---|
| table (pooled a, b) | 0.15130 | 0.6625 |
| own (alpha, beta) | 0.15133 | 0.6611 |
| + gap | 0.14523 | 0.7647 |
| + meta | 0.13831 | 0.8028 |
| + cross (H4 prior) | 0.13697 | 0.8128 |
| + cross + acq (H4-acq prior) | 0.13462 | 0.8356 |
| initial prior, intercept | 0.14266 | n/a |
| initial prior, + meta | 0.12560 | n/a |

The prior itself improves monotonically as terms are added (gap and metadata matter most), yet the fused mAP does not follow (§1). The extra prior information is largely carried by the image already, which agrees with the EDA verdict that metadata are redundant with the image ([EDA](2026-10-01-padchest-metadata-eda.md)).

### 3. Decision rule (pre-specified, [redesign plan](2026-10-01-markov-hmm-redesign-plan.md))
- **Win: PASS.** H4 beats E on primary with CI excluding 0 at both seeds: s42 +0.0375 [+0.0228, +0.0513], s1024 +0.0424 [+0.0279, +0.0592].
- **Matches B-early: PASS.** Paired Δ(H4 − B-early, matched seed): primary +0.0021 [-0.0155, +0.0197] (s42), -0.0022 [-0.0204, +0.0158] (s1024); full -0.0037 [-0.0110, +0.0049], -0.0022 [-0.0101, +0.0064]. All CIs include 0. (has_prior: -0.0031 [-0.0142, +0.0069], -0.0032 [-0.0158, +0.0069]; no_prior -0.0022 [-0.0144, +0.0033], -0.0007 [-0.0059, +0.0042].)
- **Guard 1, H4 not worse than E on no_prior: marginal.** s1024 -0.0005 [-0.0015, +0.0005] passes. At s42 the Δ is -0.0009 with CI [-0.0018, -0.0001], which excludes 0 by 0.0001 (ens -0.0009 [-0.0020, +0.0001]). Strictly read, the guard fails at s42; in magnitude it is under 0.001 mAP. The cause is H3 and H4 using w_0 = 0.25 in some folds (the metadata-only initial prior); H1 and H2 leave no_prior unchanged. H4-acq is neutral (-0.0001 [-0.0026, +0.0023]).
- **Guard 2, H4 not worse than E on new-finding stratum AP ("new vs absent", 28 classes): not clearly met.** s42 0.3678 → 0.3621 (-0.0057), s1024 0.3647 → 0.3615 (-0.0032), ens 0.3765 → 0.3746 (-0.0019). results.json stores no CI for the stratum values, so significance is unknown; the sign is negative at all three emissions. "persistent vs resolved" (26 classes): 0.7505 → 0.7462, 0.7526 → 0.7515, 0.7512 → 0.7493 (also slightly down). Contrast with B-early in the A/B verdict, whose new-finding stratum was flat to up (+0.0057 at s42). H4-F moves the new-finding stratum up (0.3678 → 0.3689).
- Outcome: both headline criteria pass; both guards are borderline-to-negative by 0.001 to 0.006. Do not describe H4 as strictly guard-clean. H1, which is not the primary arm, has no guard issue on no_prior (stratum values not printed for H1).
- Sub-populations for H4: has_prior & study_not_in_train +0.0491 [+0.0321, +0.0683] (s42), +0.0547 [+0.0337, +0.0765] (s1024), +0.0485 (ens); history val-only +0.0319 [+0.0035, +0.0600], +0.0238 [+0.0005, +0.0529], +0.0320 [+0.0077, +0.0579]; **patient_unseen -0.0053 [-0.0151, -0.0006]**, -0.0057 [-0.0153, -0.0001], -0.0043 [-0.0124, +0.0004] (a small loss on patients with no earlier state in the prior, of the same kind as Guard 1 but larger; the s42 and s1024 CIs end just below 0). Head/medium/tail dAP on has_prior (s42): +0.0130 / +0.0438 / +0.0465; s1024 +0.0132 / +0.0443 / +0.0659; ens +0.0120 / +0.0425 / +0.0498. Medium and tail gain most, in line with persistent findings.

### 4. H4 vs B-early and exploratory comparisons
- H4 vs B-early: indistinguishable on primary and full val at both seeds (§3). The post-hoc model needs no retraining and matches the network that was fine-tuned with the prior.
- **EXPLORATORY** H1 − B-early (not pre-specified): primary +0.0124 [-0.0076, +0.0348] (s42), +0.0081 [-0.0126, +0.0285] (s1024); full +0.0036 [-0.0061, +0.0172], +0.0082 [-0.0025, +0.0202]; has_prior +0.0088 [-0.0066, +0.0246], +0.0131 [-0.0032, +0.0287]. All include 0; point estimates favour H1.
- **EXPLORATORY** H4-F − B-early (s42): primary -0.0265 [-0.0467, -0.0063]; full -0.0157 [-0.0233, -0.0084]; has_prior -0.0326 [-0.0449, -0.0202]. The image-only history is clearly below the in-network prior that sees the report.
- B-early0 (zero prior, the only condition available at test): full -0.0027 [-0.0092, +0.0047] (s42), +0.0004 [-0.0045, +0.0049] (s1024); no usable leaderboard gain.

### 5. Filtered mode
H4-F (s42): primary +0.0092 [-0.0013, +0.0181] vs observed H4 +0.0375; has_prior +0.0134 [+0.0071, +0.0195] vs +0.0423; full +0.0032 [-0.0016, +0.0083] vs +0.0152. On the history val-only subset (the only one with out-of-sample history emissions): +0.0043 [-0.0098, +0.0187] vs H4 +0.0319 [+0.0035, +0.0600]. Derived: filtered mode keeps about a quarter of the observed gain on primary (0.0092 / 0.0375 = 0.25). Most of the gain therefore needs the previous **report**, not the previous **image**. That fits a reporting habit (a radiologist copying forward the earlier report) as well as real persistence of findings; the data cannot separate the two. Weights chosen for H4-F are lower (w_h 0.25 to 0.5), consistent with a noisier belief. The filtered history of train-fold images uses in-sample emissions (train-fold mAP 0.713 against about 0.465 on val), so any filtered-mode gain is, if anything, optimistic.

### 6. Ablations
- **H5 (MRF):** w_J chosen 0 in all folds at all three emissions, so H5 equals H4 to four decimals in every subset. Strongest learned couplings J (run.log): Normal~pleural effusion -1.51, Support Devices~central venous catheter +1.46, Normal~cardiomegaly -1.40, Nodule~Normal -1.37, Normal~Support Devices -1.31, Normal~aortic elongation -1.31. These are real label correlations that the ML-Decoder head already encodes.
- **H4-acq:** H4-acq ≈ H4 (primary +0.0377 vs +0.0375, +0.0431 vs +0.0424, +0.0361 vs +0.0364; no_prior -0.0001 vs -0.0009 at s42). The held-out prior log-loss and AUROC improve (0.1346 and 0.8356 vs 0.1370 and 0.8128) without a mAP gain; the full-val Δ is higher by 0.0019 to 0.0022 at s42/s1024 (derived), within the CI width.
- **Learned cross-class temporal edges B[k → c] (H4, largest |B|):** sternotomy → Normal -0.86, Normal → pleural effusion -0.85, Normal → alveolar pattern -0.70, central venous catheter → Support Devices -0.66, pleural effusion → Normal -0.65, pleural thickening → hypoexpansion +0.59, Normal → sternotomy -0.58, Normal → atelectasis -0.54. Mostly "a normal or implant study predicts the next study is not X"; plausible and not surprising.

### 7. Reproducibility note
Job 49445 is canonical. An earlier observed-only run (job 49424, node storm, `analysis/out/markov_hmm_obs/`) gives identical H0 and H1. The logistic-prior arms differ by at most 0.0014 mAP on the main subsets, because L-BFGS float differences between CPU nodes occasionally change a tied grid weight (H2 at all emissions, H4 and H5 at s1024, H4-acq at s42). Conclusions are unchanged. 49445 is the only run with filtered mode and the ens H0 = B-late check.

## Recommendation
- Treat the post-hoc HMM as a longitudinal-deployment tool only. If prior reports are available, use H1-style calibrated late fusion (calibration, a fitted w_h) first; the extra terms of H4 have not paid for their complexity. Report H4 as the pre-specified primary, but say H1 is numerically better (exploratory).
- Do not spend GPU on B-early; it adds nothing to post-hoc fusion.
- For the leaderboard nothing changes: no history, no metadata, so effect 0.

## Caveats
- Labels are report-derived, so the "previous state" is a previous report. Reporting habits may inflate the prior's value (consistent with the filtered-mode result).
- Validation is not patient-disjoint (77% of val images share a patient with train, 53% their StudyID); the prior is fitted on train and applied to patients it has seen.
- Filtered-mode train-image emissions are in-sample (train mAP 0.713 vs about 0.465 val).
- One validation fold; bootstrap on split seed 0 only (300 resamples); three splits average the point estimates but not the CIs. H2 and H5 have no CI.
- Guards: Guard 1 is marginally failed at s42, Guard 2 has no CI and a negative sign (§3).
- Leaderboard effect is 0 by construction. Dev and test images must never be joined to the PadChest CSV.
- The Δ [CI] columns are bootstrap means on split seed 0, not differences of the 3-split means.
- Paper confounds (384 vs 512 resolution, softmax entropy, 7.6% vs 13% triplet use) do not enter this offline analysis.
- Row order of the cached probabilities is verified in the A/B work.

## Not measured
Leaderboard effect; stratum-AP bootstrap CIs; H1 − H4 paired CI; filtered mode at s1024 or ens; B-early at ens; a patient-disjoint re-split; filtered mode with out-of-sample train emissions (e.g. cross-fitted); a real prospective setting.

## Notable
- First pre-specified win for any Markov construct in this project: H4 +0.0375 / +0.0424 primary at two seeds, matching B-early (CIs include 0). Only valid in simulation, with leaderboard effect 0.
- Contradicts the assumption that "more prior information = more mAP": the simplest calibrated chain (H1) is best; metadata, gap, cross edges, MRF and acquisition add nothing.
- About three quarters of the gain needs the previous report (filtered mode +0.0092 [-0.0013, +0.0181] primary).
- The H4 guards are borderline (no_prior -0.0009 at s42 with CI [-0.0018, -0.0001]; new-finding stratum -0.002 to -0.006).
- Absolute values here use the per-half estimator and are not comparable to the pooled numbers in the A/B report.

## Next steps
1. Optional: a paired H1 − H4 bootstrap and stratum-AP CIs from `oof_seed0.npz` (analysis only).
2. Cross-fitted train emissions to make the filtered mode honest.
3. Update the memory note `posthoc-patient-hmm` with the canonical numbers (orchestrator).
