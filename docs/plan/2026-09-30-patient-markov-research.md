# Patient-history Markov chain vs. label-graph Markov layer (A / B-late / B-early) — 2026-09-30

**Directory:** `/home/psytp7/ISBI_2026/analysis/out/patient_markov/` (`run.log`, `results.json`, `literature_notes.md`)
**Plan:** [`2026-09-30-patient-markov-research-plan.md`](2026-09-30-patient-markov-research-plan.md). This is a findings document, so it lives in `docs/plan/`. Training-run results stay in `docs/result/`.
**Script / recipe:** `analysis/patient_markov_check.py`. CPU only. Post-hoc fusion on cached flip-TTA probabilities; no training or eval job.
**Slurm job(s):** 49370 COMPLETED (elapsed 00:03:47, 16G, started 2026-09-30 19:11:45). An earlier attempt, 49369, FAILED (exit 1:0) after 00:02:53; the cause is in the main-session notes at the end. No `.out` in `/home/psytp7/logs/`. `run.log` is the tee'd stdout; its mtime (19:15:30) matches the end of 49370.

All numbers are **internal val** (`data/val_fold_1.csv`, n = 17,270 scored images). Nothing here is leaderboard or test. The val fold is not patient-disjoint and is about 30% lateral (see `docs/plan/2026-09-26-markov-applicability.md`).

## Question
The user asked which is the best approach to integrating PadChest patients' data into a Markov chain: a Markov layer as a fine-tune layer for ConvNeXt, or applying patients' data to a Markov chain before feeding ConvNeXt. Three options:
- **A.** Label-graph Markov layer after the ML-Decoder logits. Tested; negative.
- **B-late.** Patient-history Markov chain fused with the ConvNeXt output as an HMM (ConvNeXt = emission, chain = prior). Measured here.
- **B-early.** The patient's prior-study labels fed into ConvNeXt as an input (token or channel), then retrained. Not run; assessed from the literature notes only.

"Before feeding ConvNeXt" is read as B-early. The measured variant is the "after" one, B-late.

**Short answer:**
- **Leaderboard (Task 1):** none helps.
  - A is measured negative.
  - B has zero applicability by construction: there is no usable history, and using any on evaluation images is forbidden.
  - B-early would also cost a retrain for no test-time upside.
- **Clinical longitudinal study:** B-late is the best of the three.
  - It has a positive internal signal whose CI excludes zero.
  - It costs CPU minutes and needs no retraining.
  - It reproduces the ConvNeXt baseline exactly when no prior exists.
  - B-late > B-early is provisional, because B-early was not run.

## Hard constraints (Task 1)
- **B has zero leaderboard applicability.** CXR-LT 2026 de-duplicated PadChest-GR test patients and studies against training and withheld patient IDs and hidden labels. The rules require "no attempts to re-identify individual patients" and forbid "external annotations overlapping with the CXR-LT evaluation sets" (arXiv:2604.15555, pp. 4–5, §2.7; quotes from the literature notes).
- **The PadChest CSV carries `Report` and `Labels` for every image, including the PadChest-GR evaluation images.** Joining dev/test images to it would be direct label leakage and a data-use violation. This analysis stays within those limits:
  - the script reads only `ImageID, StudyID, PatientID, StudyDate_DICOM, Projection`;
  - labels come only from `label_list` in `data/{train,val}_fold_1.csv`;
  - no dev/test image is joined;
  - `run.log` has aggregates only (no patient or study IDs).
- **B is meaningful only for a clinical-deployment or longitudinal study.**
- **Housekeeping risk:** `data/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv` is untracked (`??`) and not gitignored (`git check-ignore` returns nothing). It contains evaluation-image Reports and Labels, and redistribution is prohibited. Do not commit it.

## Data and method (verified against the code)
- **Labels:** from `label_list` of `data/{train,val}_fold_1.csv`, in shard/logit order via `label_info.pt`.
- **Built-in asserts:**
  - the rebuilt val row order reproduces `labels.npy` exactly;
  - the ensemble baseline is 0.4722;
  - w = 0 reproduces the baseline;
  - every prior is strictly earlier;
  - each image date equals its study date.
- **Emission `p`:** the cached flip-TTA greedy-3 ensemble (`push__img768_dp01_seed42` SWA, `push__img1024_dp01_seed86` SWA, `res_sweep__img640_seed86` best). The logged `args:` of the 640 px member (job 47094) and the push-wave / v4 reports for the 768 and 1024 members all show `--metric-embed query` and `--class-weight-order shard`, so every member is post-fix.
- **Patient metadata:** the study-date state is the OR of all labels of the studies on the same (patient, date), so a same-day multi-study date is one state.
- **Chain:**
  - per class, a_c = P(1|1) and b_c = P(1|0), Laplace-smoothed as (k+1)/(n+2);
  - fitted on consecutive (patient, date) states of train-fold studies (a study counts as "train" if any of its images is in train);
  - pooled chain plus a gap-binned variant (≤30 d / 31–365 d / >365 d); a bin falls back to the pooled a, b when it has fewer than 50 pairs.
- **Prior:** each val image uses the patient's most recent strictly earlier date, across train and val studies. The state comes from the prior study's report-derived labels, treated as observed. This is a one-step Bayes update, not full forward filtering of a hidden state.
- **Fusion:** logit q = logit p + w·(logit π(prior) − logit prevalence).
  - w = 1 is the Bayes update.
  - "CV-w" is a 2-fold patient-grouped choice of one global w from {0, 0.25, …, 1.5}, scored on the held-out half.
  - Rows without a prior are unchanged.
- **Scoring:** macro mAP over the classes with at least one positive in each subset, so class counts differ across rows.
- **Bootstrap:** 300 patient-clustered resamples, 95% percentile CI.

## Results

### [1] Study-level leak (val images whose StudyID is also in train)
- Val CSV rows: 17,274.
- StudyID also in train: 9,189 (0.532).
- Identical `label_list`: 1.0000.
- Train studies with more than one distinct `label_list`: 0.

| Subset (greedy-3, flip) | n | classes | mAP |
|---|---|---|---|
| all | 17,270 | 30 | 0.4722 |
| study_in_train | 9,186 | 30 | 0.4515 |
| study_not_in_train | 8,084 | 30 | 0.4918 |
| frontal & study_in_train | 4,667 | 29 | 0.4990 |
| frontal & study_not_in_train | 7,306 | 30 | 0.4914 |
| patient_seen & study_not_in_train | 4,118 | 30 | 0.5105 |
| patient_unseen | 3,966 | 28 | 0.4743 |

| Bootstrap difference | Δ mAP [95% CI] |
|---|---|
| study_in_train − study_not_in_train (all views) | −0.0402 [−0.0776, −0.0019] |
| frontal & study_in_train − frontal & study_not_in_train | +0.0082 [−0.0289, +0.0455] |

### [2] Patient chain (train-fold consecutive pairs)
- 83,209 studies; 1 excluded because its StudyID spans more than one patient or date.
- Inconsistent labels within a study: 0. Images without a date: 0.
- 24,180 consecutive (patient, date) pairs from 12,845 patients; 1,024 same-day multi-study dates; median gap 133 d.

Top 10 rows by prevalence, plus four extreme rows (marked †):

| class | group | prev | n_prev+ | a = P(1\|1) | b = P(1\|0) | a/prev |
|---|---|---|---|---|---|---|
| Normal | head | 0.4018 | 6430 | 0.6169 | 0.1042 | 1.5 |
| pleural effusion | head | 0.1211 | 4581 | 0.6352 | 0.1090 | 5.2 |
| cardiomegaly | head | 0.1065 | 2780 | 0.5266 | 0.0631 | 4.9 |
| Support Devices | medium | 0.0833 | 4379 | 0.7759 | 0.0747 | 9.3 |
| aortic elongation | medium | 0.0815 | 1943 | 0.4406 | 0.0493 | 5.4 |
| Nodule | medium | 0.0760 | 1958 | 0.4444 | 0.0418 | 5.8 |
| interstitial pattern | medium | 0.0667 | 2334 | 0.4259 | 0.0674 | 6.4 |
| alveolar pattern | medium | 0.0659 | 3129 | 0.5292 | 0.0705 | 8.0 |
| atelectasis | medium | 0.0654 | 2215 | 0.3861 | 0.0736 | 5.9 |
| hyperinflated lung | medium | 0.0499 | 871 | 0.2875 | 0.0276 | 5.8 |
| sternotomy † | medium | 0.0216 | 712 | 0.8207 | 0.0073 | 38.0 |
| Pneumothorax † | tail | 0.0041 | 132 | 0.4254 | 0.0038 | 104.6 |
| pneumoperitoneo † | tail | 0.0005 | 19 | 0.0952 | 0.0010 | 189.2 |
| Hydropneumothorax † | tail | 0.0004 | 11 | 0.0769 | 0.0006 | 209.1 |

| Group | mean a | mean b | mean prevalence |
|---|---|---|---|
| head (3 classes) | 0.593 | 0.0921 | 0.2098 |
| medium (20) | 0.457 | 0.0340 | 0.0428 |
| tail (7) | 0.309 | 0.0023 | 0.0030 |

By gap ("pooled" is the pooled a over the same classes):

| Gap bin | pairs | classes with own a | mean a (own) | mean a (pooled) |
|---|---|---|---|---|
| ≤30 d | 9,123 | 25 | 0.456 | 0.471 |
| 31–365 d | 7,136 | 24 | 0.438 | 0.467 |
| >365 d | 7,921 | 24 | 0.402 | 0.467 |

The a/prev ratios of the rarest tail classes are dominated by tiny denominators. For Hydropneumothorax, a = 0.0769 = 1/13 is the Laplace floor: none of 11 positives recurred in the next state.

### [3] History-only predictor
- 5,759 val images (0.333) have a strictly earlier study; 11,511 have none (derived).
- Of the 5,759: prior in train 5,292 (91.9%, derived), frontal 4,061, patient unseen in train 80. Median gap 168 d.
- On the has-prior subset, the history-only predictor (2 distinct scores per class, so tie-heavy) scores **mAP 0.2623**, against **0.4978** for the greedy-3 ensemble on the same rows (30 classes).
- For scale, the mean train prevalence over the 30 classes is about 0.050 (derived from the group means above). This is only a rough chance level.

### [4] HMM late fusion
"Point Δ" is w=1 minus base, computed from the unrounded values in `results.json` (derived). The "Δ [CI]" column is the bootstrap **mean** as printed in `run.log`; in several rows it sits slightly above the point difference.

| Subset | n | classes | base | w=1 | Δ [CI] (bootstrap mean) | point Δ (derived) | w=1 gap-binned | CV-w (w chosen) |
|---|---|---|---|---|---|---|---|---|
| has_prior | 5,759 | 30 | 0.4978 | 0.5299 | +0.0364 [+0.0095, +0.0621] | +0.0321 | 0.5306 | 0.5545 ([0.5, 0.5]) |
| has_prior & frontal | 4,061 | 30 | 0.4989 | 0.5305 | +0.0320 [+0.0116, +0.0605] | +0.0317 | 0.5316 | 0.5394 ([0.5, 0.25]) |
| has_prior & prior_in_train | 5,292 | 30 | 0.4954 | 0.5284 | +0.0359 [+0.0104, +0.0643] | +0.0331 | 0.5292 | 0.5528 ([0.5, 0.5]) |
| has_prior & patient_unseen | 80 | 21 | 0.6157 | 0.6376 | +0.0224 [−0.0497, +0.0960] | +0.0220 | 0.6398 | 0.6058 ([0.0, 0.5]) |
| has_prior & study_not_in_train | 2,814 | 30 | 0.4957 | 0.5501 | +0.0547 [+0.0312, +0.0799] | +0.0543 | 0.5518 | 0.5611 ([0.5, 0.5]) |
| has_prior & frontal & study_not_in_train | 2,558 | 30 | 0.4879 | 0.5353 | +0.0484 [+0.0232, +0.0837] | +0.0474 | 0.5387 | 0.5466 ([0.5, 0.5]) |

Per-class ΔAP at w = 1 on has_prior:

| Group | mean ΔAP | classes | improved |
|---|---|---|---|
| head | +0.0024 | 3 | 2 |
| medium | +0.0367 | 20 | 15 |
| tail | +0.0317 | 7 | 6 |

Full val (no-prior rows unchanged): 0.4722 → 0.4752 (**+0.0030**). Leaderboard: **0** by construction.

### Reading the results

**1. CV-chosen w = 0.5 beats w = 1: confirmed as a result; the cause is not identifiable from these data.**
- In the five subsets with n ≥ 2,558, 9 of 10 fold choices are 0.5, one is 0.25, and none is 1.
- On has_prior, CV-w gains +0.0567 over base, against +0.0321 for w = 1 (derived). The Bayes update gives away about 43% of the achievable gain.
- Two explanations both predict w < 1, and the data cannot separate them:
  - **Uncalibrated probabilities.** ASL-trained probabilities are not calibrated posteriors, and the Bayes update needs p(s|x) calibrated against training prevalence. The single-model v3 SWA mECE is 0.1354 (v4 report); the ensemble's mECE was not measured.
  - **Shared evidence.** A persistent device, sternotomy or hernia is visible in the current image too, so adding the prior double-counts it.
- The run stores only the chosen w, with no score-vs-w curve and no per-class w. The per-class pattern (gain in medium/tail, about +0.002 in head) fits both explanations.
- The data do rule out time decay as the reason: gap-binning the chain adds only +0.0006 on has_prior at w = 1, against a +0.0246 gap between w = 1 and CV-w (derived).
- A recalibrate-then-refit-w step would separate the two (Next steps).
- The CV-w gain depends on the view mix: +0.0405 on has_prior & frontal (0.5394 − 0.4989, derived) against +0.0567 on all views, and the frontal fold choices disagree (0.5 vs 0.25). The frontal figure is the closer proxy for the test population.
- CV-w has no CI. It is chosen and scored within the same val fold (different halves), but only one scalar is tuned.

**2. Report-derived labels may inflate the simulated gain: plausible, but untestable here.**
- Labels come from reports, and radiologists read the prior report. The persistence a_c therefore mixes true disease persistence with reporting habit, and the data cannot separate the two.
- The likely direction is upward relative to independent ground truth. The leaderboard uses radiologist labels.
- Two observations bound the concern. Labels are not copied wholesale: mean a within the ≤30 d bin is only 0.456 (25 classes). The highest a/prev are persistent states (sternotomy 38.0, Hernia 40.3, central venous catheter 15.3, fracture 15.9). That matches physical persistence, though devices and old findings are also copied between reports.
- Treat the has-prior gain as an optimistic bound for an independent-label setting. The magnitude is unknown.

**3. The cleanest subset carries essentially no weight.**
- has_prior & patient_unseen: n = 80, 21 classes, Δ +0.0224 [−0.0497, +0.0960] (half-width about 0.07).
- CV-w falls below base there (0.6058 vs 0.6157; w chosen [0.0, 0.5]).
- The subset is 1.4% of has-prior rows and 2.0% of the 3,966 unseen-patient val images (derived). "Unseen" patients almost never have a prior in this fold, because multi-study patients mostly also appear in train.
- Its point gain is similar in size to the overall one: consistent with it, but not evidence for it.
- The claim that matters for deployment, that the image model has never seen the patient, is not established.

**4. The study-level leak: largely inconclusive, and view-confounded in the all-views comparison.**
- 53.2% of val CSV rows share a StudyID with train, with 100% identical labels. Labels belong to the study, so a sibling image carries the same label.
- On all views, study_in_train scores *lower*: −0.0402 [−0.0776, −0.0019]. A leak would give the opposite sign.
- The view mix explains it. study_in_train is 50.8% frontal (4,667/9,186) against 90.4% for study_not_in_train (7,306/8,084) (derived). Lateral mAP is lower (0.4058 vs 0.4968 frontal in the 2026-09-26 audit).
- Among frontal images, Δ is +0.0082 [−0.0289, +0.0455]. No leak benefit is detectable, but with a CI half-width of about 0.037, effects of a few hundredths are not excluded.
- The log cannot say whether a frontal val image's sibling in train is frontal or lateral.
- **Consequence for B-late.** The chain's a, b are fitted on "train" studies, which include the studies of val rows that also appear in train. For those 53% of val rows, the target label is part of the chain fit. This is negligible for pooled parameters but not strictly zero for the rarest classes.
  - The has_prior & study_not_in_train rows (n = 2,814) avoid this, and their gain is *larger* (+0.0543 point). The leak does not drive the B-late gain.
  - The complement (has-prior & study_in_train, n = 2,945, derived) was not scored, and the two CIs overlap, so no real difference between them can be claimed.

**5. Gain concentrates in medium/tail: confirmed. Gap-binning adds nothing demonstrable.**
- The per-class dAPs average to +0.0321, equal to the has_prior point Δ (derived check).
- Contribution by group (derived): medium about 76%, tail about 23%, head about 1%. 23 of 30 classes improved.
- These dAPs have no CIs, and tail APs rest on very few positives. The two rarest tail classes have train prevalence 0.0004–0.0005, about 2–3 expected positives among 5,759 rows (derived from prevalence, not counted).
- Head has only 3 classes, with low a/prev (1.5–5.2), so the prior has little to add there.
- Gap-binned vs pooled at w = 1, in table order: +0.0006, +0.0011, +0.0008, +0.0022, +0.0017, +0.0033 (derived). All positive but tiny, with no CI, from heavily overlapping rows.
- Persistence decays only slowly with gap: own a falls below pooled a by about 0.015, 0.029 and 0.065 across the three bins (derived).

**6. Full val vs has-prior subset vs leaderboard.**
- Has-prior: +0.0321 point at w = 1 (bootstrap mean +0.0364, CI excludes 0), covering 33.3% of val.
- Full val: +0.0030 at w = 1. Linear dilution would predict about +0.011 (0.0321 × 0.333, derived); the observed gain is about 28% of that.
- Hypothesis, not tested: the prior shifts has-prior rows relative to the unshifted no-prior rows within the pooled per-class ranking. A per-subset AP cannot see that cross-population calibration effect.
- This full-val number uses w = 1, which CV disfavours; full val at w = 0.5 was not computed.
- Leaderboard effect: 0.

## A vs B-late vs B-early

| | **A. Label-graph Markov layer** | **B-late. Patient chain as HMM prior** | **B-early. Prior labels as ConvNeXt input** |
|---|---|---|---|
| Usable at leaderboard test time | Yes (needs only the image) | **No.** No patient history at test time; IDs withheld | **No.** Needs the prior at inference |
| Allowed by challenge rules | Yes | Not on dev/test images: re-identification, and external annotations overlapping the eval sets (the CSV holds their Report and Labels). Internal train/val simulation only | Same as B-late; training would also need patient-linked priors |
| Internal evidence | Offline smoothing: −0.0006 to −0.0053 mAP on 0.4722 (α 0.02–0.2). Trainable v4 (jobs 49063–49065 vs 49000): SWA −0.0049 / −0.0033 / −0.0031; single seed, within noise but consistently negative; gates near 0 (mean \|g\| 0.011–0.013) | has-prior (n = 5,759): +0.0321 point at w = 1 (bootstrap +0.0364 [+0.0095, +0.0621]); CV-w +0.0567 (w = 0.5); frontal CV-w +0.0405. Full val +0.0030. Leaderboard 0 | **None (not run).** The literature notes have no head-to-head early-vs-late numbers and no quantitative gains for either |
| Compute cost | Offline: CPU seconds. Trainable: a full Stage-2 run (v4 jobs ran 2026-09-26 20:00 → 2026-09-27 06:14) | CPU only; job 49370 took 3 min 47 s. No retraining | Highest: a prior-linked data pipeline (priors in the shards), then a full Stage-1/2 retrain per recipe |
| Robustness / shortcut risk | No patient information, so no copy risk; the layer is barely used | The prior enters through 2 parameters per class and one scalar w, with bounded logit shifts (derived, at w = 1: sternotomy +5.3 / −1.1; Normal +0.87 / −1.75). It cannot "copy" like a network, but a wrong prior still shifts scores. w = 1 over-trusts it (CV prefers 0.5). Behaviour on new or resolved findings is **unmeasured**. About half of positive priors are not followed by a positive (mean a 0.457 medium, 0.593 head) | Highest. Zhu et al. (arXiv:2306.08749), who pre-fill reports from prior CXR + prior report + current CXR: "when the label results of current and previous report are the same, 88.96% percent of the generated results match them", and "when the labels of current and previous report are different, there is an 84.42% probability of generated results being incorrect". That is report generation, not classification, so transfer is an inference, but it is direct evidence that input-level conditioning copies the prior and fails on changes |
| Behaviour when no prior exists | Always applies, with no or slightly negative effect | **Exact identity.** No-prior rows (66.7% of val, 11,511 derived) are unchanged, and w = 0 reproduces the baseline (asserted) | Needs a no-prior token plus prior-dropout in training. At test time everything is no-prior, so at best it matches the baseline |

Literature context (notes only):
- Santeramo et al. (arXiv:1807.06144) model exam sequences with a time-modulated LSTM over per-exam features: a late, sequence-after-CNN design.
- BioViL-T (Bannur et al., arXiv:2301.04558) feeds the prior image and report into the encoder: an early design.
- Both need priors at inference. Most longitudinal models target progression labels (improved / unchanged / worse), not static presence.

## Recommendation

**For Task 1 / the leaderboard: none of the three.**
- A is measured negative.
- B-late is 0 by construction, and B on dev/test images would be prohibited.
- B-early has no test-time upside and costs a retrain.
- Do not spend GPU time on any Markov variant for this track. The +0.0030 full-val figure is a simulation artefact that does not exist at test time.

**For a clinical longitudinal study: B-late first.**
- It is the only variant here with a positive signal whose CI excludes 0.
- It is cheap, needs no retraining, is inspectable, and reduces to the ConvNeXt baseline when there is no prior.
- Start with a global w near 0.5, not 1. Recalibrate `p` and refit w before trusting the Bayes w = 1.
- Consider B-early only if B-late leaves headroom. It needs prior-dropout and an evaluation focused on changed findings, which is why the literature targets progression labels.
- B-late > B-early is **provisional**. B-early was not run; the ranking rests on cost, graceful degradation and the Zhu et al. copying statistics from a neighbouring task, not on an in-project measurement.

## Caveats
- **Simulation, not deployment.** The val fold is not patient-disjoint, and 91.9% of has-prior rows have their prior in train (derived): the image model has seen the patient's earlier study. The patient-unseen subset (n = 80) is too small to test the patient-disjoint case.
- **Report-derived labels.** The simulated gain is likely optimistic (point 2).
- **Observed prior.** The prior is the observed label state from the prior study's report, not a hidden state filtered through the ConvNeXt. Without prior reports, the prior would be noisier.
- **Bootstrap.** 300 resamples. The Δ [CI] column is the bootstrap mean, which differs slightly from the point difference. w is not re-selected inside the bootstrap, and CV-w has no CI.
- **Subset mAPs are not comparable across rows**, because the class sets differ (30 / 29 / 28 / 21).
- **Single configuration.** One ensemble, one val fold (fold_1), one CV split seed (0), a w-grid step of 0.25. The greedy-3 ensemble was selected on this same internal val, so the baseline is already optimistic for every arm.
- **Chain fit vs the study-level leak:** see point 4. Negligible for pooled parameters.
- **Validity vs earlier work.** All three ensemble members are post-fix (`--metric-embed query`, `--class-weight-order shard`), so the class-weight-order and triplet-wiring bugs do not affect these numbers. The A row comes from earlier reports. The paper-revision confounds (resolution, entropy type, mining type) do not touch B-late.

## Not measured
- B-early, in any form.
- Behaviour on changed findings (new or resolved between studies). Only aggregate mAP was computed, so the gain could sit entirely on unchanged findings while hurting changes.
- Ensemble calibration (mECE), and any recalibration of `p` before fusion.
- A score-vs-w curve, and per-class w.
- A patient-disjoint re-split (frontal, at least one finding), and radiologist ground truth independent of the prior report.
- Full val at w = 0.5 or CV-w.
- The has-prior & study_in_train complement (n = 2,945, derived), and a paired test of its gain against study_not_in_train.
- Per-class CIs on the head/medium/tail ΔAP.
- Two most-recent priors (a higher-order chain), and gap effects beyond three bins.
- Prior labels taken from ConvNeXt predictions instead of reports.

## Notable
- **First internal signal in the campaign from a Markov-type idea whose CI excludes 0.** Patient-history late fusion on the has-prior subset gives +0.0321 point (bootstrap mean +0.0364, CI [+0.0095, +0.0621]), and CV-w gives +0.0405 on frontal has-prior. The earlier "drop" verdict for (B) still holds for Task 1, but because of the test design, not because the signal is absent. The paper text (§V-D) should not imply the idea is uninformative; it is unusable on the leaderboard.
- **New finding not in the earlier audit.** 53.2% of val CSV rows (9,189 of 17,274) share a StudyID with train, with 100% identical labels. The earlier audit reported only patient overlap (77.0% of val images). The frontal-only mAP comparison is inconclusive (+0.0082 [−0.0289, +0.0455]). This belongs in §V-E.
- **Untracked PadChest CSV**, with evaluation-image Reports and Labels, is not gitignored. Do not commit it.

## Next steps (suggestions only; nothing was launched)
1. Add `data/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv` (and `data/*filtered*.csv` if appropriate) to `.gitignore` before any commit.
2. CPU-only follow-up on cached probabilities: fit per-class temperature or Platt calibration of `p` on one patient-grouped half, then refit w on the other. If the optimum moves toward 1, the cause is calibration; if it stays near 0.5, the evidence is correlated.
3. Score the changed-finding subset (y_t ≠ y_{t−1}) and the has-prior & study_in_train complement. Both need only the cached probabilities and the existing metadata join.
4. Report full-val at w = 0.5, so the +0.0030 dilution at w = 1 is not misread.
5. If a clinical track is pursued, build a patient-disjoint, frontal-only split before any B-early training. B-early is a GPU job and needs user approval.
6. For the paper: one sentence in §V-D pointing to this simulation as out of scope for the leaderboard, with the caveats above, and the study-level overlap number in §V-E.

## Sources (all from the verified literature notes, `analysis/out/patient_markov/literature_notes.md`)
- Zhu et al., arXiv:2306.08749.
- Santeramo et al., arXiv:1807.06144.
- Bannur et al. (BioViL-T), arXiv:2301.04558.
- CXR-LT 2026 challenge paper, arXiv:2604.15555.
- In-repo: `analysis/patient_markov_check.py`, `analysis/out/patient_markov/run.log`, `analysis/out/patient_markov/results.json`, `docs/result/2026-09-27-v4-markov-layer.md`, `docs/plan/2026-09-26-markov-applicability.md`, `docs/paper/markov/paper.tex` (§V-D, §V-E).

---

## Main-session notes (added 2026-09-30)
- **Delivery.** The `cxrlt-research-analyst` subagent returned this report as text. The main session saved it here and changed only paths: the moved markdown note, and the literature-notes file now copied into the run directory.
- **Job 49369 failure cause.** The first run stopped at an assertion that every StudyID maps to a single patient and date. Exactly one of the 83,210 fold studies has a StudyID shared by two PatientIDs (a PadChest metadata error). None spans two dates. Job 49370 excludes that study and logs the exclusion (section [2]: "excluded … 1").
- **Spot checks of derived numbers (main session):**
  - point Δ has_prior = 0.5299 − 0.4978 = +0.0321;
  - prior-in-train share = 5,292/5,759 = 91.9%;
  - sternotomy logit shifts: logit(0.8207) − logit(0.0216) = +5.33 and logit(0.0073) − logit(0.0216) = −1.10;
  - Normal: +0.87 and −1.75.

  All match.
- **Filtered-CSV path.** Since this run, `CXRLT_2026_training_filtered.csv` has moved from `train/` to `data/`, and `evaluate/class_groups.py` (read by the script for train prevalence) now points there. The file content is unchanged: its SHA-256 equals the git-LFS oid committed for `train/` (`caa1ff31…4189459`, 14,583,116 bytes). The results therefore stand.
