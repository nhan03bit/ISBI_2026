# Claims check after the v5 reproducibility rerun — 2026-10-05

Type: findings / claims check (not a training-result report). INTERNAL val only; nothing here is leaderboard or test.
Numbers and artifacts: [docs/result/2026-10-05-v5-rerun-repro.md](../result/2026-10-05-v5-rerun-repro.md) (jobs 49855/49856 training, 49857-49860 evals; analysis in `analysis/out/v5_rerun/`).
Related: [Markov A/B verdict](2026-10-01-markov-ab-verdict.md), [0.4722 provenance](2026-10-02-047-provenance-and-paper-corrections.md), [post-hoc HMM](2026-10-01-markov-hmm-posthoc.md), [v4 result](../result/2026-09-27-v4-markov-result.md), [applicability check](2026-09-26-markov-applicability.md), [metadata EDA](2026-10-01-padchest-metadata-eda.md).

## Questions
1. Does the rerun reproduce v5? (answered in the result report; summary under Findings)
2. Is it correct that "every training and eval job converges to ~0.47 mAP" and that patient metadata / the Markov prior lifts the entropy-triplet model "from 0.45 to 0.47"?
3. Is the core novelty "transition probabilities between different related labels solve label co-occurrence (one X-ray with several diseases)" supported?
4. What does the seen-vs-unseen-patient structure of the internal val imply?

## Data and method
- 4 training runs (orig and rerun, seeds 42 and 1024, identical `args:` lines) and 2-seed SWA flip-TTA ensembles evaluated with the prior WITH (previous report available) and ZERO (withheld, the test condition) on the full 17,270-row val fold, plus my offline analysis of the dumped probabilities: per-class AP, patient-clustered paired bootstrap (300 resamples), subsets (has-prior, frontal, patient-unseen).
- Row alignment verified (identical `keys.txt` md5 across `probs_ab_flip`, `probs_v5_orig_recheck_flip`, `probs_v5_rerun_flip`; equal to the reconstructed `val_key_order.npy`; v3 s42 dump in `probs_w01_flip` equals the `probs_ab_flip` copy, max |d| = 0.0).
- Code read for Q3: `analysis/patient_prior.py`, `train/prior_conditioning.py`, `analysis/out/patient_markov/priors.pt` (keys `a`, `b`, `prevalence`, `train`, `val`, ...), and the existing label-graph / HMM results in the linked reports.

## Findings

### Q1. Reproducibility (short form; tables in the result report)
- Same args, same seeds: single SWA 0.4731 (s42) and 0.4764 (s1024) vs original 0.4807 and 0.4786. Ensemble flip: WITH 0.4797 vs 0.4867 (gap -0.0070, CI [-0.0143, -0.0011]); ZERO 0.4611 vs 0.4693 (gap -0.0081, CI [-0.0144, -0.0006]). These gaps are 3-4x the v3 seed s.d. (about 0.002).
- The gap is the base model, not the prior. The prior's own contribution (WITH - ZERO, same models) is +0.0175 and +0.0186 on full val and +0.0329 / +0.0317 on has-prior frontal rows, so it reproduces.
- The gap is almost entirely Hydropneumothorax (7 val positives): ensemble AP 0.466 (orig) vs 0.285 (rerun); that one class is 0.0060 of the 0.0070 WITH gap. Excluding it (exploratory, chosen post hoc), orig vs rerun differ by -0.0010 (WITH) and -0.0011 (ZERO), inside the seed s.d., and WITH - ZERO is +0.0190 / +0.0192.
- Paired comparison with the matched no-prior v3 2-seed flip ensemble (0.4659), patient-clustered 95% CI: rerun WITH +0.0138 [+0.0042, +0.0209]; orig WITH +0.0208 [+0.0132, +0.0285]; rerun ZERO -0.0048 [-0.0129, +0.0004]; orig ZERO +0.0033 [-0.0033, +0.0074]. On 29 classes: WITH +0.0161 / +0.0171 (CIs exclude 0), ZERO -0.0031 [-0.0064, -0.0001] / -0.0019 [-0.0056, +0.0010].

### Q2. What reaches "~0.47", and what is responsible
Internal val mAP, flip TTA unless stated. "Prior" below means the same-label patient-history prior built from the previous report's labels.

| Configuration | mAP | History / metadata used? | What produces the level |
|---|---|---|---|
| v3 single, SWA, no TTA (768 px) | 0.4600 / 0.4582 (s42 / s1024) | no | recipe (768 px, dp 0.1, SWA) |
| v3 single, flip | 0.4650 / 0.4626 | no | + flip TTA (+0.005) |
| **v3 matched 2-seed ensemble, flip** | **0.4659** | no | + 2-model averaging; **the correct like-for-like baseline for every 2-seed v5 ensemble** |
| v3 fixed 4-SWA average, flip (`tta_w1_flip.json`) | 0.4699 | no | 4 models, no selection |
| v3 greedy-3 ensemble, flip | 0.4722 | no | mixed 768/1024/640 px, selected on the scored fold (optimistic) |
| v5 ZERO ensemble (orig / rerun / 4-model) | 0.4693 / 0.4611 / 0.4667 | no (prior zeroed) | the v3 recipe with an unused conditioner; not different from v3 (see Q2 numbers above) |
| v5 WITH ensemble (orig / rerun / 4-model) | 0.4867 / 0.4797 / 0.4847 | **yes, previous report** | the prior, on has-prior rows |

Statements that follow from the data:
- "Every job converges to ~0.47" is wrong. This batch alone spans 0.4560 (single rerun ZERO flip) to 0.4867 (orig WITH ensemble); the WITH and ZERO arms differ by 0.017-0.019 by design.
- **The ~0.47 level is reached without any patient information.** v3 greedy-3 is 0.4722 and the fixed 4-model v3 average is 0.4699; the v5 ZERO ensembles are 0.4611-0.4693. So 0.47 is the cost of ensembling, TTA and resolution, not of metadata or the prior.
- **"From 0.45 to 0.47" has no matching pair of numbers.** No no-prior 768 px v3 model scores 0.45; the like-for-like single-model baseline is 0.4600 / 0.4582 (SWA) or 0.4585 / 0.4564 (best epoch). The ensemble baseline is 0.4659. The v5-with-history lift over these is +0.013 to +0.021 (single, no TTA; both runs) and +0.014 to +0.021 (2-seed ensemble), i.e. 0.466 to about 0.48-0.49, not 0.45 to 0.47. If 0.45 comes from an older 512 px or pre-fix Stage-2 number, that comparison mixes resolution and recipe changes and must not be attributed to the prior.
- **Where the lift comes from, per configuration.**
  - History-available prior: +0.0175 to +0.0186 full val over the same models without the prior (+0.032 to +0.033 on has-prior frontal rows; 33% of val images have a prior); it is concentrated in medium (20/20 up, +0.019) and head (3/3 up, +0.007) classes. Tail not reproducible (7 classes, driven by Hydropneumothorax). Zero effect on no-prior rows by construction.
  - No-history prior (ZERO): -0.0048 to +0.0033 vs v3 matched ensemble, all CIs include 0 on 30 classes; on 29 classes two of three ZERO ensembles are below v3 with CIs that exclude or touch 0. No evidence of a benefit. Against the greedy-3 v3 ensemble the ZERO ensembles are lower (-0.003 to -0.011 on full val; -0.010 to -0.012 excluding Hydropneumothorax, CIs exclude 0).
  - v3 ensemble: ensembling/TTA, not the Markov component (also stated in the 0.4722 provenance note; no gate ablation exists).
- **"Metadata" is the wrong word for what v5 uses.** v5 reads only the previous study's label set (via a per-class chain), not age, sex, projection or scanner fields. The metadata EDA and the HMM ladder found those fields add nothing to the image model (H3, H4-acq no better than H1 on primary; Markov paper abstract: stacking them lowers mAP by 0.003 to 0.016).
- The prior-labelled gain is a simulation of a longitudinal setting. Filtered mode (history inferred from earlier images) keeps about a quarter of it (primary +0.0092 [-0.0013, +0.0181] vs +0.0375). The previous report's label is what carries the gain, and report copy-forward cannot be ruled out from report-derived labels.

### Q3. What the prior actually models, and the evidence on cross-label transitions
**What is in the code.**
- `analysis/patient_prior.py`: `fit_chain` fits, per class c, a 2-state chain on consecutive (patient, date) states from the train fold: `a_c = P(c now | c at the previous visit)` and `b_c = P(c now | c absent at the previous visit)` (30 values each; 24,180 consecutive pairs in `priors.pt`). The prior vector is `r_c = logit(a_c or b_c) - logit(prevalence_c)` plus a has-prior flag (31 numbers). There is no 30 x 30 matrix over different labels anywhere in this prior; each class sees only its own earlier state.
- `train/prior_conditioning.py`: label query c is offset by `r_c * u_c + W r` (u: 30 x 768 per-class, W: Linear(31 -> 768)). `W r` is shared across queries, so the network could learn some cross-label use in principle, but nothing in the prior or the loss asks for it, and no cross-label structure was measured in the trained model.
- So v5 models **same-label persistence across a patient's visits** (temporal recurrence/resolution of each finding), not transitions between different labels, and not co-occurrence within one study.

**Evidence on cross-label / co-occurrence modelling in this project (all internal val; all negative or null).**

| Experiment | Result |
|---|---|
| Label-graph random-walk smoothing, fixed co-occurrence P, post hoc (`analysis/markov_check.py`, greedy-3 base 0.4722) | alpha 0.02 / 0.05 / 0.10 / 0.20: -0.0006 / -0.0012 / -0.0024 / -0.0053; hurts at every strength, mostly head classes |
| Normal-gating (challenge winner's exclusivity prior) | -0.0103 (alpha 0.25), -0.0239 (0.5), -0.0459 (1.0) |
| v4 trainable label-graph Markov layer in the network (mk1, mk2, mk1fix; job 49063-49065) | SWA 0.4551 / 0.4566 / 0.4569 vs matched v3 0.4600 (-0.0031 to -0.0049, single seed); gates stayed small (mean 0.011-0.013); learnable vs frozen P no difference |
| v4 A-strong (lr multiplier 100, job 49377) | SWA 0.4575 vs 0.4600; primary-subset tail -0.048; full-val dAP -0.0034 [-0.0115, +0.0044] |
| Post-hoc HMM H4 (adds cross-class temporal edges: label k at t-1 -> label c at t) | adds held-out prior information (log-loss 0.13831 -> 0.13697, AUROC 0.8028 -> 0.8128) but no mAP: H4 primary +0.0375 vs H3 +0.0378 (s42); the simplest chain H1 is numerically best on every has-prior subset (primary 0.5511 vs H4 0.5411 at s42) |
| H5 (within-study label MRF, coupling J between labels) | w_J chosen 0 in every fold at all three emissions, so H5 equals H4 to 4 decimals; the learned couplings are real (Normal~pleural effusion -1.51, Support Devices~central venous catheter +1.46) but the ML-Decoder head already encodes them |
| Orig-vs-rerun consistency of the same-label prior | stable (+0.018-0.019 full val) |

**Verdict on the claim.** The data do not support "transition probabilities between related labels solve co-occurrence"; on this model, metric and data they refute its usefulness: every cross-label construct tested is null or negative, and the one construct that works (v5 / B-late / H1) uses only same-label transitions across visits. The likely reason is the one already in the Markov paper: ML-Decoder's query self-attention already mixes labels per image, and AP is a per-class ranking metric that a global label-to-label matrix cannot improve. This does not show that co-occurrence cannot be modelled; it shows these constructs add nothing to this decoder. The draft `docs/paper/isbi/paper.tex` already states this correctly ("Same-label transitions across visits do help ... Label-to-label transition priors ... do not improve it"); the proposed claim would contradict the draft and the data.

Nuance for the draft: its statement that "cross-label terms inside a patient-level model are never selected" is exact only for the within-study MRF (H5, w_J = 0). The cross-class temporal edges of H4 are fitted and used, they just do not beat H1.

### Q4. Seen vs unseen patients
- 77.0% of val images share a patient with train (53% their StudyID, 30% are lateral); 33.3% have an earlier study (5,759 of 17,270) and use the prior. The test set is frontal and patient-disjoint and has no history.
- On the leaderboard-like subset (frontal & patient-unseen, n 3,572) the prior's contribution is nil: WITH - ZERO +0.0011 [-0.0016, +0.0029] (orig) and +0.0014 [-0.0009, +0.0030] (rerun). The WITH ensembles score 0.4703 and 0.4761 there vs 0.4808 for the no-prior v3 matched ensemble (differences -0.0105 [-0.0372, +0.0022] and -0.0046 [-0.0215, +0.0075]). ZERO is -0.0116 / -0.0060 vs v3 on the same subset, CIs include 0 but all point estimates are negative.
- The gain requires the previous report, which does not exist for a deployed single-image case or for the CXR-LT test patients. The patient-seen/seen-study structure inflates everything on internal val (the model has also seen those patients' images in training).
- Overall metrics on the full fold mix seen and unseen patients; the better proxy for the leaderboard is the frontal & patient-unseen subset.

## What a paper can and cannot claim
**Can claim (with the scope stated):**
1. When the patient's previous report labels are available, a per-class same-label chain prior, injected into the label queries or fused post hoc, raises internal-val mAP. Matched-baseline numbers: single SWA +0.013 to +0.021 over 0.4600 / 0.4582 (four runs: 0.4731, 0.4764, 0.4786, 0.4807; mean 0.4772, s.d. 0.0032); 2-seed flip ensemble +0.014 to +0.021 over 0.4659 (0.4797 rerun, 0.4867 original; 4-model 0.4847). Report these as two independent trainings, with the spread, not 0.4807 / 0.4867 alone.
2. The prior's own contribution reproduces: +0.0175 / +0.0186 full val, +0.032 to +0.033 on has-prior frontal rows, all CIs exclude 0, concentrated in head and medium classes.
3. The gain is zero without history (ZERO within -0.005 to +0.003 of v3, CIs include 0 on 30 classes) and zero by construction on the test set; the claim is for deployments where prior reports exist.
4. Negative results for cross-label Markov constructs (smoothing, Normal-gating, v4 layer, MRF) are a legitimate finding, as the draft states.
5. The result is a longitudinal simulation on a split that shares patients with training.

**Cannot claim:**
- Any leaderboard or test gain from the patient prior (history absent; effect 0).
- That the prior or metadata "lifts the model from 0.45 to 0.47" or that "all jobs converge to ~0.47": the 0.47 level is reached by v3 ensembling with no history (0.4699 fixed, 0.4722 greedy-3), and the with-history models are at 0.48-0.49.
- That cross-label transition probabilities resolve label co-occurrence: the prior is same-label; all cross-label tests are null or negative.
- That patient metadata helps: v5 does not use demographics or acquisition fields, and where they were tested they added nothing.
- That v5 helps without history, or on unseen patients (frontal & patient-unseen: +0.001, CI includes 0).
- A tail-class benefit: seven tail classes with 7 to 140 val positives, Hydropneumothorax (7) swings macro mAP by up to 0.006, orig vs rerun tail effect flips sign.
- Single-run 0.4807 / 0.4867 as a typical result.
- Any statement on entropy gating's share of the v3 baseline (no `--triplet-lambda 0` control exists).

## Recommendation
- Reframe the contribution as: a same-label patient-history Markov prior, useful where previous reports exist; with cross-label co-occurrence priors reported as negative. Use the 4-run mean and ensemble numbers with the explicit v3 matched baseline (0.4659 ensemble; 0.4600 / 0.4582 single).
- Report the 29-class (or head+medium) mAP beside the 30-class mAP wherever run-to-run or small-sample comparisons are made, and state that the two rarest classes (7 and 9 val positives) are noise-dominated.
- In the draft's abstract and conclusion, add the rerun numbers (0.4731 / 0.4764 single; 0.4797 ensemble) next to the originals if the 0.4807 / 0.4867 are quoted.

## Caveats
- Single val fold, two seeds per arm, 300 bootstrap resamples; the 4-model ensemble mixes orig and rerun members.
- The Hydropneumothorax-excluded analysis was chosen after seeing the per-class table (exploratory).
- v3 greedy-3 is selected on the scored fold and mixes resolutions, so the ZERO-vs-greedy-3 comparison favours greedy-3.
- Whether the 0.45 in the user's framing refers to a specific run was not determinable from the repo; I could not find a no-prior 768 px v3 number of 0.45.
- Report-derived labels: a previous-report prior can reflect copy-forward habits as well as true persistence.
- Paper confounds (384 vs 512 px, softmax entropy, 7.6% vs 13% triplet use) do not affect these 768 px matched comparisons.

## Not measured
- Leaderboard effect of anything here; a patient-disjoint re-split; v5 on a held-out set with history; cross-label structure inside the trained v5 network (what `W r` learned); the cause of non-reproducible training at fixed seeds; a `--triplet-lambda 0` v3 control; per-class eval via `eval_perclass.sh` for the v5 checkpoints; mAUC / mF1 / mECE of the v3 matched ensemble.

## Notable
- The original v5 headline (0.4807 / 0.4867) did not reproduce: reruns 0.4731 / 0.4764 single, 0.4797 WITH and 0.4611 ZERO ensemble. The gap is 0.001 once the 7-positive class Hydropneumothorax is excluded.
- The ZERO result on 2026-10-01 (+0.0033 vs v3) was favourable noise; the rerun is -0.0048.
- Hydropneumothorax is a noise source for every full-val mAP comparison in the campaign.
- The user's proposed novelty (cross-label transition probabilities) is contradicted by the project's own data; the working component is same-label temporal persistence.

## Next steps
1. Quote the 4-run mean/ensemble and spread in the paper draft, add the rerun to the Markov abstract/conclusion, keep the "leaderboard effect 0" disclosure.
2. Add a `--triplet-lambda 0` v3 control if any entropy-gating claim is wanted (still missing).
3. Optional: evaluate on frontal & patient-unseen as the primary leaderboard proxy in all future tables.
4. Orchestrator: update the memory notes (campaign note: v5 0.4807 not reproduced, rerun 0.4731-0.4764; the Hydropneumothorax noise caveat).
