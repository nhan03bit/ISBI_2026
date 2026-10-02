# PadChest metadata vs the 30 Task 1 labels: EDA — 2026-10-01

*Offline analysis (no training run). Step 2 of [`2026-10-01-markov-hmm-redesign-plan.md`](2026-10-01-markov-hmm-redesign-plan.md). Related: [`2026-09-30-patient-markov-research.md`](2026-09-30-patient-markov-research.md), [`2026-09-26-markov-applicability.md`](2026-09-26-markov-applicability.md). Paper section: `docs/paper/markov/paper.tex`, `sec:metaeda`.*

**Script:** `analysis/metadata_eda.py` (loader `analysis/patient_meta.py`)
**Slurm jobs:** 49419 (sections [1]-[4], COMPLETED 00:30:09), 49425 (corrected [5], COMPLETED 00:49:49)
**Artifacts:** `analysis/out/metadata_eda/{run.log, run_incremental.log, eda.json, assoc.csv, figures/}`; paper figures in `docs/paper/markov/figures/`
**Superseded:** the [5] block at the end of `run.log` pooled cross-fitted predictions from halves with different per-class transforms (mixed scales inside one class ranking). It is invalid and is not used here.

## Question
Which PadChest patient, study and acquisition metadata are associated with the 30 CXR-LT labels, and do they add anything on top of the image model? The answer decides what the transition prior of the post-prediction patient HMM may use.

## Data and method
(Verified against the script docstring and `run.log`.)
- **Data:** train fold only for the screening: 103,305 images, 50,604 patients, 75,819 studies, 74,782 (patient, date) states (1 ambiguous study excluded). Labels come from the fold CSV `label_list` only; the loader never reads the PadChest `Labels`/`Report` columns. Metadata come from a column whitelist, for fold images only. No leaderboard image is joined.
- **Units:** study for patient/study fields (sex, age, pediatric, n_prior_states, gap_days, study_year, study_acuity); image for view/acquisition fields.
- **[1] univariate screen:** Cramer's V, mutual information, chi-square p with Benjamini-Hochberg FDR, log-OR (Haldane) of the extreme level with a patient-clustered Poisson bootstrap CI (200), and single-feature AUROC for numeric fields. Classes with q_BH < 0.05 are counted as significant, but p-values ignore clustering of images within patients, so this report ranks by effect size.
- **[2] multivariate:** per-class metadata-only logistic regression (L2 1e-4), patient-grouped 5-fold CV on train images. Groups: *patient* = sex + age spline; *temporal* = has-prior, log gap, log study rank; *view* = acuity flags + lateral flag; *acquisition* = DICOM scanner/exposure/resolution fields.
- **[3] transition modulation:** persistence a = P(1|1) and onset b = P(1|0) by stratum of the later state, on 24,180 consecutive train (patient, date) pairs (mean over classes with >= 30 prior positives in the stratum).
- **[5] incremental value (corrected, `run_incremental.log`):** on val, per-class stacking a_c logit p + b_c + gamma_c phi, cross-fitted on two patient halves; mAP is computed within each half and averaged; paired patient-clustered bootstrap (300). Emissions: **s42** = job 49000 SWA (768 px, seed 42, flip TTA); **ens** = greedy-3 ensemble. Both are post-fix runs (`--metric-embed query`, `--class-weight-order shard`). The "recalibration only" reference equals the emission exactly for s42 (delta 0.0000), which validates the estimator; for ens it is -0.0003 (all) and -0.0005 (frontal).

## Findings

### [1] Strongest associations
By field (max Cramer's V over classes; sig classes = q_BH < 0.05):

| Field | Unit | Max V | Mean V | Sig classes |
|---|---|---|---|---|
| study_acuity | study | 0.528 | 0.104 | 27/30 |
| projection | image | 0.504 | 0.102 | 28/30 |
| age | study | 0.447 | 0.104 | 28/30 |
| gap_days | study | 0.447 | 0.103 | 28/30 |
| SpatialResolution | image | 0.406 | 0.099 | 30/30 |
| RelativeXRayExposure | image | 0.304 | 0.076 | 29/30 |
| Rows | image | 0.293 | 0.067 | 28/30 |
| n_prior_states | study | 0.272 | 0.056 | 20/30 |
| sex | study | 0.133 | 0.036 | 24/30 |
| study_year | study | 0.122 | 0.035 | 24/30 |
| pediatric | study | 0.058 | 0.011 | 14/30 |

(Other exposure/window/manufacturer fields: max V 0.24-0.29; full list in `run.log`.)

By class (strongest field, log-OR of the extreme level with 95% CI):
- **Support Devices:** study_acuity V 0.528 (PA/L log-OR -3.34 [-3.44, -3.25]); projection V 0.504 (AP_horizontal +2.95).
- **Central venous catheter:** study_acuity V 0.495 (PA/L -4.21 [-4.35, -4.09]); projection V 0.494 (AP_horizontal +3.52).
- **Normal:** age V 0.447 (oldest band, 76-105 y, -1.79 [-1.84, -1.73]); gap_days V 0.286 (gap <= 3 d, -2.78).
- **Alveolar pattern:** study_acuity V 0.333; gap <= 3 d +2.25 [+2.15, +2.34].
- **Pleural effusion:** gap_days V 0.269 (gap <= 3 d, +1.59); study_acuity V 0.223.
- **Cardiomegaly:** age V 0.271 (youngest band, <= 42 y, -2.39).
- **Aortic elongation:** age V 0.264 (<= 42 y, -4.32 [-4.89, -3.94]).
- **Aortic atheromatosis:** age V 0.134 (<= 42 y, -6.31); single-feature age AUROC 0.816.
- Single-feature AUROC for age: aortic atheromatosis 0.816, aortic elongation 0.776, cardiomegaly 0.755, Hernia 0.753, Normal 0.237. Number of prior states: central venous catheter 0.760, Support Devices 0.722. Gap in days: central venous catheter 0.161, Support Devices 0.180 (short gaps go with devices).
- **Tail classes** have weak, mostly acquisition-driven associations (V 0.02-0.10; e.g. Pneumothorax with SpatialResolution 0.15, V 0.101; Hydropneumothorax V 0.029). These rest on few positives and wide CIs.

Sex is a weak field (max V 0.133); the largest sex log-ORs are Emphysema (M, +1.08) and sternotomy (M, +0.77).

### [2] Metadata alone, held-out (macro over 30 classes; prevalence baseline AP 0.0502, AUROC 0.5)

| Feature set | Macro AUROC | Macro AP |
|---|---|---|
| all four groups | 0.760 | 0.142 |
| only patient (sex, age) | 0.654 | 0.088 |
| only temporal | 0.589 | 0.090 |
| only view | 0.589 | 0.082 |
| only acquisition | 0.659 | 0.091 |
| all - patient | 0.691 | 0.116 |
| all - temporal | 0.752 | 0.130 |
| all - view | 0.753 | 0.139 |
| all - acquisition | 0.729 | 0.133 |

Derived drop-out cost relative to "all" (AUROC / AP): patient -0.069 / -0.026; acquisition -0.031 / -0.009; temporal -0.008 / -0.012; view -0.007 / -0.003. Removing the patient group costs the most on both metrics, even though sex and age alone are not the best single group by AUROC (acquisition alone is marginally higher, 0.659 vs 0.654). Per-class AUROC (`eda.json`, figure `metadata_only_auroc.pdf`): central venous catheter 0.929, Support Devices 0.903, aortic atheromatosis 0.869, Subcutaneous Emphysema 0.856, Pneumothorax 0.847; lowest azygos lobe 0.609, hypoexpansion 0.637, Mass 0.642.

### [3] Transition modulation (24,180 pairs; mean persistence a over classes with enough positives)

| Stratum of later state | Pairs | Classes | Mean a | Mean b |
|---|---|---|---|---|
| sex F / M | 10,990 / 13,184 | 26 / 28 | 0.468 / 0.459 | 0.030 / 0.035 |
| age <50 | 4,436 | 17 | 0.503 | 0.022 |
| age 50-69 | 9,014 | 26 | 0.454 | 0.030 |
| age 70-79 | 6,758 | 25 | 0.468 | 0.038 |
| age >=80 | 3,970 | 24 | 0.454 | 0.045 |
| acuity PA/L | 16,669 | 28 | 0.439 | 0.030 |
| acuity AP | 1,427 | 15 | 0.468 | 0.045 |
| acuity AP_horizontal | 6,084 | 26 | 0.388 | 0.046 |
| gap <=30 d | 9,123 | 26 | 0.450 | 0.038 |
| gap 31-365 d | 7,136 | 24 | 0.438 | 0.030 |
| gap >365 d | 7,921 | 24 | 0.402 | 0.034 |

(Pooled a is 0.464-0.534 depending on the class set, in brackets in `run.log`; the class sets differ per stratum, so the means are not directly comparable across rows.) Sex barely moves persistence; age moves onset b (0.022 to 0.045) more than persistence; gap shortens persistence over long gaps; acuity changes b and, per class, a.

Per-class persistence by acuity of the later study (`eda.json`, figure `persistence_by_acuity.pdf`; PA/L -> AP -> AP_horizontal):
- Support Devices 0.578 -> 0.794 -> 0.834; central venous catheter 0.269 -> 0.753 -> 0.812; alveolar pattern 0.266 -> 0.639 -> 0.680.
- Normal 0.676 -> 0.237 -> 0.258 (a Normal study that is followed by an AP study is much less often Normal again).
- Pleural effusion is nearly flat (0.626 -> 0.607 -> 0.653); sternotomy is high everywhere (0.77-0.84).
- Per-cell pair counts are not stored in `eda.json`; AP cells are small (1,427 pairs in total), so treat the spread across the AP column as noisy.

### [5] Incremental value over the image model (val; delta mAP vs emission, 95% patient-clustered CI)

| Emission | Subset | Base mAP | + patient | + view | + acquisition | + all three |
|---|---|---|---|---|---|---|
| s42 | all | 0.4687 | -0.0026 [-0.0047, -0.0004] | -0.0036 [-0.0075, -0.0008] | -0.0048 [-0.0079, -0.0018] | -0.0095 [-0.0138, -0.0056] |
| s42 | frontal | 0.4940 | -0.0028 [-0.0050, -0.0010] | -0.0032 [-0.0081, +0.0022] | -0.0063 [-0.0120, -0.0018] | -0.0106 [-0.0174, -0.0049] |
| ens | all | 0.4785 | -0.0049 [-0.0083, -0.0023] | -0.0047 [-0.0089, -0.0019] | -0.0069 [-0.0110, -0.0034] | -0.0123 [-0.0181, -0.0076] |
| ens | frontal | 0.5062 | -0.0060 [-0.0153, -0.0012] | -0.0044 [-0.0087, -0.0010] | -0.0091 [-0.0181, -0.0031] | -0.0158 [-0.0264, -0.0067] |

Base mAP is the within-half average, so it differs from the full-val mAP quoted elsewhere (s42: 0.4687 here against 0.4650 pooled in the superseded block). Against the recalibration-only reference (columns `d vs recal-only` in `run_incremental.log`) the deltas differ by at most 0.0008 from those above. Every CI excludes 0 except s42 frontal + view (-0.0032 [-0.0081, +0.0022]). No cell is positive.

## What this means (reading)
1. **Metadata are informative about findings.** Metadata alone give held-out macro AUROC 0.760 against 0.5 and macro AP 0.142 against a prevalence of 0.050.
2. **The patient group (sex, age) carries the most unique information** (largest cost when removed), mainly age: age-related findings (aortic atheromatosis, elongation, cardiomegaly, Normal) have steep prevalence curves.
3. **Acquisition fields rank high, but they are care-setting and scanner proxies** (SpatialResolution, exposure, Rows, Manufacturer separate portable AP-supine units from fixed PA/L rooms), not patient biology. They should not be presented as clinical risk factors. The EDA's own acuity and gap fields (AP_horizontal, short gaps) track the same inpatient/ICU context and are the strongest class-level associations (devices, catheters, alveolar pattern, effusion).
4. **Metadata add nothing on top of the image model; they lower mAP.** Stacking metadata on the image logit changes mAP by -0.003 (s42, patient) to -0.012 (ens, all groups) over the whole val fold, with CIs excluding 0 (one frontal cell does not). Interpretation (not tested here): the image already shows devices, age-related aortic change and AP-supine geometry, so a per-class metadata term mostly re-adds this signal as noise with extra per-class parameters fitted on half of val.
5. **Persistence varies with acuity and gap**, and the variation is class-specific (devices and catheters persist more after AP-supine studies, Normal persists less). This is the one place where metadata carry something the single image cannot: how the previous study's state carries over.

### Consequences for the HMM design
- Do not use metadata as a per-image prior on the emission side. Expect H3's "metadata in the initial prior" (w_0 > 0) to be neutral or negative on no-prior rows; the plan's guard (H4 not worse than E on no-prior rows) should catch it.
- Put metadata in the **transition** (acuity, gap; optionally age for onset), which is where persistence and onset actually change. Sex can be dropped (max V 0.133, persistence 0.468 vs 0.459).
- H4-acq (acquisition fields) is not supported by [5]: acquisition is the worst single group (-0.005 to -0.009). Keep it as an ablation only, flagged as a care-setting proxy.
- Age: use a spline (the age curves in `age_prevalence.pdf` are monotone but non-linear).
- Leaderboard effect is **0 by construction** (see caveats).

## Caveats
- **Labels are report-derived** (PadChest report text mined), so associations include reporting habits, e.g. device mentions that follow acuity.
- **The internal val fold is not patient-disjoint** (77% of val images share a patient with train; 53% share a study; see project memory `data-split-patient-leakage`). [5] is cross-fitted on patient halves of val, so it is patient-clean within val, but the emission itself was trained on train patients who overlap with val.
- **No metadata or patient history exists for the leaderboard test images, and they must never be joined to the PadChest CSV** (it holds their labels). The leaderboard effect of anything in this report is 0 by construction; every number here is an internal-val or train-fold number.
- **p-values are naive with respect to clustering** (images/studies of one patient are correlated). Sig-class counts are therefore upper bounds; rank by effect size. Log-OR CIs use a patient-clustered bootstrap and are the more reliable uncertainty.
- **Tail-class results rest on few positives:** the [2] AUROCs for Hydropneumothorax (0.781, AP 0.001), pneumoperitoneo (0.841, AP 0.002), Subcutaneous Emphysema (0.856), Pneumothorax (0.847), Mass (0.642), azygos lobe (0.609) are unstable and their ranking should not be interpreted.
- **[5] estimator:** the stacking model is simple (per-class linear in logit p and phi, fit on about half of val); a richer or more regularised model could do less badly, but there is no reason to expect a gain over the image. The earlier pooled estimate in `run.log` (about -0.023 to -0.032) is superseded and invalid.
- Emission runs are post-fix (`--metric-embed query`, `--class-weight-order shard`). The EDA itself is independent of the class-weight and metric-embedding bugs; only [5] depends on an emission.
- Univariate numbers are associations in observational data, not causal effects.

## Not measured
- Metadata effect inside the post-hoc HMM transition (job 49445 still running at the time of writing; not reported here or in the paper).
- Interactions beyond per-class linear terms (for example age x acuity), or non-linear (tree) metadata models.
- Per-cell sample sizes for the persistence-by-acuity table, and bootstrap CIs for [3].
- Head/medium/tail breakdown of [2] and [5].

## Notable
- Metadata (including acquisition fields) lower image-model mAP when stacked (-0.003 to -0.012, CIs exclude 0 in 15 of 16 cells; the exception is s42 frontal + view). This supports routing metadata only through the transition, and closes the "metadata as per-image prior" idea.
- Acquisition/DICOM fields are a care-setting proxy and carry real signal (acquisition-only AUROC 0.659 is as high as sex+age alone); relevant to any leakage-style arguments about PadChest.

## Next steps
- Wait for job 49445 (post-hoc HMM) and read H3 against H4 to see whether acuity/gap in the transition helps on the has-prior frontal subset.
- If H4-acq is run, report it as an ablation with the care-setting caveat.
- Write the HMM report to `docs/plan/` and link it here.
