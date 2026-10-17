# Edge cases where Entropy + Triplet Memory might beat ASL — 2026-10-17

**Status.** Sections 1-2 (the subset list, the metric, the tests, the decision rule) were written to disk at 16:27 BST on 2026-10-17 (file mtime), before any contrast in this document was computed. The first run of `analysis/entropy_edge_cases.py` started after that. Section 3 onward is filled in after the run. Nothing in sections 1-2 was edited afterwards except where marked "AMENDED". Sections 3.8-3.10 (ensemble-level view, resolution, positive directions) and the Holm-variant bullet in 3.1 are post hoc additions made after the main run and are labelled as such.

**Companion result document:** [docs/result/2026-10-17-entropy-best-config.md](../result/2026-10-17-entropy-best-config.md) (best Entropy configuration and the fair 5-seed ensemble control). Earlier context: [2026-10-12-why-entropy-shows-no-gain.md](2026-10-12-why-entropy-shows-no-gain.md), [2026-10-16-tail-map-breakdown.md](2026-10-16-tail-map-breakdown.md), [2026-10-16-beat-asl-plan.md](2026-10-16-beat-asl-plan.md).

**No training, no GPU, no Slurm job.** Everything below reads cached flip-TTA probability dumps and `labels.npy`; PadChest metadata is read only for `ImageID`, `PatientID` and `Projection` of internal train/val-fold images.

## Question
Entropy + Triplet Memory (arm C: `train/train_2_v5.py`, λ 0.1, `--anchor-rule mean`, memory 2048) ties ASL (`train/train_2_v0.py`) on every aggregate: Table 1 at 768 px, area, 5 seeds, C − A = +0.0001 [−0.0042, +0.0045], 2 wins / 3 losses; tail −0.003. Is there a slice of the validation set, or a class, where C beats A, and that the macro average hides?

## 1. Pre-registration: subsets and metric

### 1.1 Image-level subsets and class-level subsets (the family F1, ten tests)
Validation = the 17,270 rows of `analysis/out/probs_unified_flip/keys.txt` / `labels.npy` (every dump used here shares that row order). "Unseen patient" = the val image's `PatientID` has no image in `data/train_fold_1.csv`. Frontal = `Projection` in {PA, AP, AP_horizontal}; lateral = `Projection` == L.

| ID | Subset | Definition | Images (audit, counts only) |
|---|---|---|---|
| S1 | frontal | Projection PA / AP / AP_horizontal | 11,973 |
| S2 | frontal & unseen patient | S1 and unseen patient | 3,572 |
| S3 | lateral | Projection L | 5,234 |
| S4 | AP_horizontal | Projection AP_horizontal (bedside / supine) | 1,664 |
| S5 | abnormal only | at least 1 positive among the 29 non-Normal classes (drops the 6,891 Normal-only images); Normal excluded from the macro | 10,379 |
| S6 | at least 3 findings | at least 3 positives among the 29 non-Normal classes (co-occurrence); Normal excluded from the macro | 2,195 |
| S7 | tail group, all views | the 7 tail classes (train prevalence < 1%, `evaluate/class_groups.py`), all images | 17,270 |
| S8 | tail group, frontal | the same 7 classes, S1 images | 11,973 |
| S9 | rare10, all views | the 10 rarest classes by train prevalence | 17,270 |
| S10 | extreme tail, all views | the 6 classes with train prevalence < 0.5% | 17,270 |

Neither frontal nor lateral: 62 COSTAL and 1 EXCLUDE image (they sit only in the all-view rows). Normal co-occurs with another label on 26 images; they are kept in S5 (they have a non-Normal label) and S6 where they qualify.

### 1.2 Per-class family F2 (seven tests)
AP of each of the 7 tail classes on all views (val positives): Hydropneumothorax 7, pneumoperitoneo 9, Subcutaneous Emphysema 29, Mass 49, azygos lobe 58, Pneumothorax 71, calcified densities 140. Frontal per-class values are printed as context only and are not tested.

### 1.3 Metric M_S
- S1-S6: macro mean of per-class AP over the classes that have at least 1 positive and at least 1 negative among the subset's images (the convention of `entropy_ablation.subset_map` and the earlier frontal & unseen number). AP is computed on the subset's images only. Normal is dropped from the classes of S5 and S6.
- S7-S10: mean of per-class AP over the named classes, AP computed on all images (S7, S9, S10) or on frontal images (S8). A class with no positive in the subset is skipped (this only matters in a bootstrap resample).
- AP is the tie-grouped AP of `analysis/entropy_ablation.FastAP` (equals torchmetrics at unit weights; asserted in the script).

### 1.4 Contrasts
**Primary (768 px).** D_S = mean over the 5 seeds {42, 123, 456, 789, 1024} of M_S(C_seed) − M_S(A_seed). C = `analysis/out/probs_unified_flip_area/c__seed_<S>__model_swa@768_flip_area.npy`, A = `analysis/out/probs_v0_flip_area/v0_768__seed_<S>__model_swa@768_flip_area.npy`: flip-TTA, SWA weights, INTER_AREA eval. Pairing by seed index is nominal: the sample shuffle of these 10 runs was unseeded, so a seed does not mean the same data order.

**Secondary (512 px ablation).** Same estimand over the 6 seeds {42, 86, 123, 456, 789, 1024} for C − A and C − B (A = ASL only, B = triplet + bank with every image an anchor, C = entropy gate). Single-scale 512 px, no TTA, `model_best.pth`, lanczos eval, from `preds.<tag>.npz` under `train/checkpoint_Triplet_3/{ablation512/{asl,nogate},seedsweep}/seed_<S>/Model_run/` (the C arm is the six `seedsweep` runs, as in `analysis/entropy_ablation.py`). Computed on the same subsets.

### 1.5 Inference and decision rule (fixed now)
For each subset and contrast:
1. Seed level: mean, sample sd, paired t 95% interval, paired-t two-sided p (p_t), exact sign-flip two-sided p (reported; with 5 seeds its minimum is 0.0625, with 6 it is 0.03125), wins.
2. Image level: paired image bootstrap, B = 2000 resamples of the 17,270 images (one resample shared by every run and every subset), percentile 95% interval of the seed-mean D_S, and p_img = min(1, 2 · min((#{D* ≤ 0} + 1) / (B + 1), (#{D* ≥ 0} + 1) / (B + 1))).
3. Decision statistic p_dec = max(p_t, p_img). A real gain has to be repeatable across training runs and not an accident of which validation images were drawn; max is the intersection-union test of those two.
4. Holm step-down adjustment of p_dec at α = 0.05, separately within F1 (m = 10) and within F2 (m = 7). A subset "survives correction" if its Holm-adjusted p_dec is below 0.05.
5. Everything is labelled exploratory. Two-sided tests; a surviving subset with a negative sign would mean ASL beats Entropy there.
6. The resolvable effect of each subset: the 80%-power minimum detectable effect of the paired t-test, (t_0.975,df + t_0.80,df) · sd / √n, and the half-width of the image interval.
7. Sensitivity (reported, not in the decision): patient-cluster bootstrap in place of the image bootstrap; Welch unpaired t instead of paired t at 768 px; the same 768 px contrast under lanczos (point estimate and seed interval).

### 1.6 "Positive direction" criterion (fixed now)
A subset is a candidate positive direction if the point estimate is above 0 in all three contrasts: 768 px C − A, 512 px C − A, 512 px C − B. It is "supported" only if in addition its 768 px Holm-adjusted p_dec is below 0.05. Otherwise it is hypothesis-generating. Subsets are not re-cut after seeing the results; any new slice would be reported as post hoc.

### 1.7 What was known before this run (so the reader can discount)
- All-view aggregates, tail group and frontal group of C − A at 768 px (lanczos) were already published in [2026-10-16-tail-map-breakdown.md](2026-10-16-tail-map-breakdown.md): macro +0.0001 (area), tail −0.0030 all views and −0.0016 frontal, rare10 −0.0022 and −0.0014. S1, S7, S8, S9 and S10 are therefore re-measurements (area instead of lanczos for the tail rows).
- The 512 px ablation already reported, on frontal & unseen images, C − B = +0.0071 [−0.0009, +0.0151], p = 0.0625, 5/6 wins, and C − A = +0.0052 [−0.0036, +0.0141], p = 0.1875, 4/6 wins (`analysis/out/entropy_ablation.json`, contrasts `gate - nogate` and `gate - asl`, key `subsets.frontal_unseen`). S2 was chosen because of it, so the 512 px S2 numbers are not an independent test.
- S5 and S6 come from the gate analysis: the softmax-entropy gate mostly separates not-Normal-only images from Normal-only ones and does not select multi-finding or tail images ([2026-10-12-why-entropy-shows-no-gain.md](2026-10-12-why-entropy-shows-no-gain.md) H3). S3 and S4 come from the same analysis (the gate lets lateral views through more often) and from the bedside views having the most findings per image.

## 2. Data and method (as run; written after the run)

**AMENDED (status line, wording only).** The status paragraph at the top says sections 1-2 were written before the run. Only section 1 (1.1-1.7) is pre-registered; this section 2 and section 3 are post-run text. No number, subset, test or rule in section 1 was changed.

**Run.** `analysis/entropy_edge_cases.py` (helpers in `analysis/edge_common.py`), Slurm job 53227, CPU only (8 CPUs, 32 GB), COMPLETED, exit 0:0, elapsed 00:49:34, ended 2026-10-17 17:23:13. Outputs: `analysis/out/entropy_edge_cases.json` and `analysis/out/entropy_edge_cases.log`. The log header reads `boot 2000 workers 8 seed 20261017` (bootstrap RNG seed 20261017). Post hoc additions (3.8): `analysis/entropy_edge_cases_ens.py`, Slurm job 53235, COMPLETED 0:0, 00:09:27; output `analysis/out/entropy_edge_cases_ens.json`. The first smoke runs (24 resamples) were used only to debug the scripts; their numbers appear nowhere.

**Inputs.** 38 cached runs, all on the same 17,270 val rows (`keys.txt` / `labels.npy`; label equality with each dump's `labels.npy` is asserted in the loader):
- 768 px, flip-TTA, SWA, INTER_AREA: C (`probs_unified_flip_area/c__seed_<S>__model_swa@768_flip_area.npy`) and A (`probs_v0_flip_area/v0_768__seed_<S>__...`), seeds 42/123/456/789/1024. The lanczos versions of the same dumps feed the sensitivity table. That accounts for 5 x 2 x 2 = 20 runs.
- 512 px ablation, single-scale, `model_best.pth`, lanczos, `preds.*.npz`: A, B, C at seeds 42/86/123/456/789/1024 = 18 runs.
- No training, no GPU. Metadata: `ImageID`, `PatientID`, `Projection` only (internal train/val folds).

**Image counts (log, matches the audit in 1.1).** all 17,270; frontal 11,973; frontal & unseen patient 3,572; lateral 5,234; AP_horizontal 1,664; abnormal 10,379; >=3 findings 2,195.

**Pipeline check.** 64 comparisons of this script's per-subset values against the values already stored in `analysis/out/entropy_ablation.json` (512 px, e.g. B seed 42, S1): worst absolute difference 6.03e-08. The AP implementation is therefore the one used for the earlier ablation.

**Statistics, exactly as in 1.5.** Per subset and contrast: seed-level paired t (mean, 95% interval, p_t, exact sign-flip p, wins = number of seeds with C - A > 0); paired image bootstrap with B = 2000 resamples of the 17,270 images, one resample shared by every run and every subset (percentile interval of the seed-mean difference, p_img); p_dec = max(p_t, p_img); Holm step-down within F1 (m = 10) and within F2 (m = 7), computed separately for each of the three contrasts (768 C - A, 512 C - A, 512 C - B); 80%-power MDE of the paired t-test. Sensitivity: patient-cluster bootstrap (768 px only, B = 2000), Welch t (768 px only), lanczos 768 px contrast. The patient-cluster bootstrap and the Welch test were not run at 512 px.

**Cost of the three bootstraps (log).** image 768 815 s, patient-cluster 768 753 s, image 512 1289 s; total 2906 s.

**Deviation from 1.x.** None that changes a test. The 512 px C - A and C - B contrasts were each Holm-adjusted as their own families of 10 + 7; the "ALL" row is printed as a reference and is not in either family.

## 3. Findings

All numbers are read from `analysis/out/entropy_edge_cases.log`. "Wins" = seeds with C > A (or C > B). "Seed CI" = paired-t 95% interval over seeds (5 at 768 px, 6 at 512 px). "Image CI" = paired image-bootstrap percentile interval. p_dec = max(p_t, p_img); Holm = Holm-adjusted p_dec within the family. Reference, all views, 30 classes at 768 px: C - A = +0.0001, seed CI [-0.0042, +0.0045], 2/5 wins, image CI [-0.0032, +0.0028], p_dec 0.931, MDE 0.0059.

### 3.1 Headline (decision rule 1.5)

- **No subset and no class survives Holm at 768 px (F1: all ten Holm-adjusted p_dec = 1.000; F2: all seven = 1.000).** The smallest 768 px p_dec values are calcified densities 0.218, S6 (>=3 findings) 0.250, S2 0.378, S3 0.381.
- **Nothing survives Holm in the secondary 512 px contrasts either.** Smallest Holm-adjusted p_dec (unadjusted p_dec in brackets): 512 C - A, F1 1.000 (S6, 0.194), F2 0.227 (calcified densities, 0.032); 512 C - B, F1 0.850 (S2, 0.085), F2 0.424 (azygos lobe, 0.061).
- **Only one candidate positive direction under 1.6: calcified densities** (positive in all three contrasts: +0.0063, +0.0097, +0.0070). It is **hypothesis-generating, not supported**: its 768 px Holm-adjusted p_dec is 1.000 (p_dec 0.218). S4 AP_horizontal is the only subset negative in all three contrasts (-0.0020, -0.0016, -0.0013), also not significant.
- Counting all 51 tested contrasts (17 per contrast family x 3), only four have p_t < 0.05 before any correction: S2 at 768 px (negative, see 3.3), Pneumothorax 512 C - A (+0.0142, p_t 0.006), calcified densities 512 C - A (+0.0097, 0.032) and Subcutaneous Emphysema 512 C - B (+0.0219, 0.013). About 2.5 would be expected by chance at 0.05 with no true effect. Only one of the four (calcified densities, 512 C - A, p_dec 0.032) also has p_img < 0.05 (0.007), and it does not survive Holm (0.227).
- **The conclusion does not depend on the choice of p (checked after the run).** Holm applied to the seed-level paired-t p alone gives, at 768 px, a smallest adjusted p of 0.321 in F1 (S2, negative direction) and 1.000 in F2; applied to the image-bootstrap p alone it gives 1.000 in both families. At 512 px the looser definitions would let two class-level C - A results through, neither of which is a 768 px result: Pneumothorax by the seed-level p alone (+0.0142, 6/6 wins, Holm 0.044; at 768 px +0.0023, 2/5 wins) and calcified densities by the image-level p alone (+0.0097, Holm 0.049; at 768 px +0.0063, Holm 1.000). Under the exact sign-flip p (minimum 0.0625 with 5 seeds, 0.03125 with 6) no Holm-adjusted value is below 0.219 in any family or contrast.

### 3.2 F1: image- and class-group subsets, 768 px area, C - A (primary)

| ID | Subset | M(A) | M(C) | C - A | Seed CI (wins) | Image CI | p_t | p_img | p_dec | Holm | MDE80 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | frontal | 0.4899 | 0.4896 | -0.0003 | [-0.0074, +0.0067] (4/5) | [-0.0050, +0.0049] | 0.898 | 0.958 | 0.958 | 1.000 | 0.0095 |
| S2 | frontal & unseen patient | 0.4738 | 0.4696 | **-0.0042** | [-0.0079, -0.0006] (0/5) | [-0.0164, +0.0027] | 0.032 | 0.378 | 0.378 | 1.000 | 0.0049 |
| S3 | lateral | 0.4007 | 0.4033 | +0.0025 | [-0.0046, +0.0097] (3/5) | [-0.0012, +0.0055] | 0.381 | 0.195 | 0.381 | 1.000 | 0.0096 |
| S4 | AP_horizontal | 0.4016 | 0.3995 | -0.0020 | [-0.0087, +0.0046] (2/5) | [-0.0078, +0.0025] | 0.449 | 0.367 | 0.449 | 1.000 | 0.0089 |
| S5 | abnormal only | 0.4753 | 0.4755 | +0.0002 | [-0.0043, +0.0048] (3/5) | [-0.0031, +0.0030] | 0.905 | 0.864 | 0.905 | 1.000 | 0.0061 |
| S6 | >=3 findings | 0.5718 | 0.5755 | +0.0036 | [-0.0006, +0.0078] (4/5) | [-0.0020, +0.0116] | 0.077 | 0.250 | 0.250 | 1.000 | 0.0057 |
| S7 | tail group, all views | 0.3404 | 0.3413 | +0.0009 | [-0.0185, +0.0203] (2/5) | [-0.0131, +0.0120] | 0.905 | 0.872 | 0.905 | 1.000 | 0.0260 |
| S8 | tail group, frontal | 0.3903 | 0.3894 | -0.0010 | [-0.0306, +0.0287] (3/5) | [-0.0205, +0.0220] | 0.933 | 0.987 | 0.987 | 1.000 | 0.0397 |
| S9 | rare10, all views | 0.3614 | 0.3616 | +0.0002 | [-0.0125, +0.0129] (3/5) | [-0.0099, +0.0082] | 0.966 | 0.967 | 0.967 | 1.000 | 0.0170 |
| S10 | extreme tail, all views | 0.3746 | 0.3745 | -0.0000 | [-0.0216, +0.0216] (3/5) | [-0.0156, +0.0132] | 0.999 | 0.986 | 0.999 | 1.000 | 0.0289 |

Notes. (i) S1: 4/5 wins but a mean of -0.0003, because one seed (456) is -0.0097 while the other four are between +0.0004 and +0.0059. (ii) S6 is the largest positive 768 px subset estimate (+0.0036, 4/5 wins, seed CI just touches zero) but its image CI includes zero and in the 512 px C - B contrast it is -0.0033, so it fails 1.6. (iii) Tail groups S7-S10 are ~0 at 768 px (+0.0009, -0.0010, +0.0002, -0.0000) with MDE 0.017-0.040: a tail effect smaller than about 0.03 cannot be resolved with 5 seeds on this val set.

### 3.3 F1 detail: S2, the slice that was chosen because of the 512 px result

S2 (frontal & unseen patient) was selected in 1.7 because the 512 px ablation favoured C there. The two 512 px numbers are not independent evidence: C - A +0.0052 [-0.0036, +0.0141] (4/6 wins), C - B +0.0071 [-0.0009, +0.0151] (5/6 wins, p_dec 0.085, Holm 0.850).

At 768 px, on the same slice, **the sign reverses**: C - A = -0.0042 area (0/5 wins, per-seed -0.0067, -0.0031, -0.0072, -0.000003, -0.0042), -0.0065 lanczos [-0.0128, -0.0002] (0/5 wins), paired-t p = 0.032, sign-flip p = 0.0625 (the minimum possible with 5 seeds), Welch p 0.051. The image bootstrap interval [-0.0164, +0.0027] and the patient-cluster interval [-0.0166, +0.0027] both contain zero, so p_dec = 0.378 and Holm = 1.000. It is therefore not a significant ASL advantage either. The honest reading: the 512 px S2 advantage for the entropy-gated arm did not replicate at 768 px (if anything C is behind A there), and since S2 was picked from the 512 result, that 512 number carries selection bias. S2 also has the smallest MDE of the family (0.0049) because the seed spread is small, so the estimate is relatively tight and the reversal is not a precision problem.

### 3.4 F2: per tail class, 768 px area, C - A (primary)

| Class (val positives) | M(A) | M(C) | C - A | Seed CI (wins) | Image CI | p_t | p_img | p_dec | Holm | MDE80 |
|---|---|---|---|---|---|---|---|---|---|---|
| Hydropneumothorax (7) | 0.2863 | 0.2921 | +0.0058 | [-0.0866, +0.0983] (2/5) | [-0.0698, +0.0720] | 0.869 | 0.794 | 0.869 | 1.000 | 0.1238 |
| pneumoperitoneo (9) | 0.3257 | 0.3203 | -0.0054 | [-0.0760, +0.0653] (1/5) | [-0.0527, +0.0223] | 0.843 | 0.958 | 0.958 | 1.000 | 0.0946 |
| Subcutaneous Emphysema (29) | 0.6188 | 0.6208 | +0.0020 | [-0.0145, +0.0185] (3/5) | [-0.0191, +0.0232] | 0.755 | 0.896 | 0.896 | 1.000 | 0.0221 |
| Mass (49) | 0.0768 | 0.0729 | -0.0039 | [-0.0186, +0.0109] (2/5) | [-0.0276, +0.0074] | 0.508 | 0.641 | 0.641 | 1.000 | 0.0198 |
| azygos lobe (58) | 0.5646 | 0.5636 | -0.0010 | [-0.0183, +0.0162] (2/5) | [-0.0113, +0.0087] | 0.878 | 0.873 | 0.878 | 1.000 | 0.0231 |
| Pneumothorax (71) | 0.3752 | 0.3775 | +0.0023 | [-0.0318, +0.0365] (2/5) | [-0.0154, +0.0204] | 0.858 | 0.791 | 0.858 | 1.000 | 0.0457 |
| calcified densities (140) | 0.1353 | 0.1417 | +0.0063 | [-0.0038, +0.0165] (3/5) | [-0.0033, +0.0141] | 0.159 | 0.218 | 0.218 | 1.000 | 0.0136 |

For the two rarest classes (7 and 9 positives) the MDE is 0.09-0.12 AP and the image intervals are about +-0.07 / +-0.04: these classes are unresolvable. The frontal-only per-class context (not tested) gives the same picture: Hydropneumothorax +0.0061 (4/5), pneumoperitoneo -0.0048, Subcutaneous Emphysema -0.0021, Mass -0.0096, azygos lobe -0.0036, Pneumothorax +0.0014, calcified densities +0.0059 (3/5).

### 3.5 Secondary: 512 px ablation, point estimates and Holm (C - A and C - B)

C - A uses the ASL-only arm, C - B the triplet + bank arm with every image an anchor (no gate). 6 seeds, no TTA, lanczos. Cells: difference (wins); p_dec / Holm in brackets.

| ID | Subset | 512 C - A | 512 C - B |
|---|---|---|---|
| ALL | all views (ref) | +0.0013 (4/6); [0.323 / -] | -0.0014 (2/6); [0.388 / -] |
| S1 | frontal | +0.0018 (4/6); [0.488 / 1.000] | -0.0009 (2/6); [0.590 / 1.000] |
| S2 | frontal & unseen | +0.0052 (4/6); [0.188 / 1.000] | +0.0071 (5/6); [0.085 / 0.850] |
| S3 | lateral | -0.0001 (4/6); [0.964 / 1.000] | -0.0004 (3/6); [0.944 / 1.000] |
| S4 | AP_horizontal | -0.0016 (3/6); [0.558 / 1.000] | -0.0013 (1/6); [0.578 / 1.000] |
| S5 | abnormal only | +0.0016 (5/6); [0.225 / 1.000] | -0.0014 (2/6); [0.416 / 1.000] |
| S6 | >=3 findings | +0.0057 (4/6); [0.194 / 1.000] | -0.0033 (3/6); [0.518 / 1.000] |
| S7 | tail, all views | +0.0065 (5/6); [0.244 / 1.000] | -0.0086 (1/6); [0.174 / 1.000] |
| S8 | tail, frontal | +0.0083 (5/6); [0.434 / 1.000] | -0.0078 (1/6); [0.271 / 1.000] |
| S9 | rare10 | +0.0047 (5/6); [0.285 / 1.000] | -0.0052 (1/6); [0.265 / 1.000] |
| S10 | extreme tail | +0.0060 (5/6); [0.344 / 1.000] | -0.0111 (1/6); [0.114 / 1.000] |
| C:Hydropneumothorax | | +0.0025 (3/6); [0.838 / 1.000] | -0.0472 (2/6); [0.212 / 0.750] |
| C:pneumoperitoneo | | +0.0045 (3/6); [0.830 / 1.000] | -0.0457 (1/6); [0.131 / 0.750] |
| C:Subcutaneous Emphysema | | -0.0029 (3/6); [0.931 / 1.000] | +0.0219 (6/6); [0.125 / 0.750] |
| C:Mass | | +0.0028 (4/6); [0.833 / 1.000] | -0.0081 (2/6); [0.216 / 0.750] |
| C:azygos lobe | | +0.0146 (5/6); [0.065 / 0.390] | +0.0163 (4/6); [0.061 / 0.424] |
| C:Pneumothorax | | +0.0142 (6/6); [0.160 / 0.800] | -0.0040 (2/6); [0.703 / 0.750] |
| C:calcified densities | | +0.0097 (5/6); [0.032 / 0.227] | +0.0070 (5/6); [0.130 / 0.750] |

Observations.
- At 512 px the tail rows (S7-S10) are positive for C - A (+0.0047 to +0.0083, 5/6 wins) and negative for C - B (-0.0052 to -0.0111, 1/6 wins), with neither significant. From the printed M columns, the no-gate arm B is above A on the tail group (S7: A 0.2854, B 0.3004, C 0.2919; arithmetic from the log's M values). The 512 px tail "gain" of C over A is thus no larger than the gain of the ungated triplet arm; the entropy gate does not add to it. At 768 px, where C and A are compared with a stronger recipe, the tail-group difference is ~0.
- 512 px Subcutaneous Emphysema C - B = +0.0219 (6/6 wins, p_t 0.013, image CI [-0.0049, +0.0481], p_dec 0.125, Holm 0.750) **does not replicate at 768 px**: C - A there is +0.0020 (3/5 wins, p_dec 0.896). The 512 C - A for the same class is -0.0029. The 512 C - B number reflects B being weak on this class (0.5219) rather than C being strong (0.5438 vs A 0.5467).
- 512 px azygos lobe (+0.0146 / +0.0163) and Pneumothorax C - A (+0.0142, 6/6 wins) are the largest 512 px class estimates, but at 768 px they are -0.0010 and +0.0023. Pneumothorax is also -0.0040 against B at 512 px. Neither meets 1.6.
- 512 px MDE values at class level are 0.011-0.115, so even the "large" class effects are about the size of what the design can just detect.

### 3.6 Candidate positive directions under 1.6 (sign-consistency block)

Point estimate above 0 in all three of 768 C - A / 512 C - A / 512 C - B:

| Subset / class | 768 area C - A | 512 C - A | 512 C - B | 768 Holm | Status |
|---|---|---|---|---|---|
| calcified densities (140 val positives) | +0.0063 | +0.0097 | +0.0070 | 1.000 (p_dec 0.218) | candidate; hypothesis-generating, not supported |

Closest others: S2 (-0.0042 / +0.0052 / +0.0071) fails on 768 px; azygos lobe (-0.0010 / +0.0146 / +0.0163) fails on 768 px; Subcutaneous Emphysema (+0.0020 / -0.0029 / +0.0219) fails on 512 C - A. All other subsets have a negative sign in at least one contrast. S4 AP_horizontal is negative in all three (-0.0020 / -0.0016 / -0.0013).

Calcified densities caveats: the 768 px effect is driven mainly by seed 1024 (per-seed, seeds 42/123/456/789/1024: C - A +0.0098, +0.0088, -0.0021, -0.0018, +0.0169; 3/5 wins, seed CI [-0.0038, +0.0165]); the 768 px Welch p is 0.063 and lanczos gives +0.0053 (4/5 wins, seed CI [-0.0017, +0.0124]); at 512 px C - A has p_dec 0.032 but Holm 0.227. It is a low-AP class (second lowest of the seven after Mass; 0.135 / 0.142 at 768 px, 0.088 / 0.097 at 512 px), so a 0.006-0.010 gain is a small absolute change. 140 val positives is the largest of the tail classes but the val split is not patient-disjoint (see below).

### 3.7 Sensitivity checks (768 px C - A)

- **Patient-cluster bootstrap vs image bootstrap: agree.** Cluster and image intervals are nearly identical for every row (e.g. S2 [-0.0166, +0.0027] vs [-0.0164, +0.0027]; S6 [-0.0023, +0.0106] vs [-0.0020, +0.0116]; calcified densities [-0.0033, +0.0143] vs [-0.0033, +0.0141]). Zero is inside every cluster interval, including the ones where the seed CI excludes zero (S2). Clustering by patient does not change any conclusion; the leakage of patients across train/val is a separate concern and is not addressed by this bootstrap.
- **Welch (unpaired) vs paired t: agree.** Welch p is within a few hundredths of paired-t p in most rows. S2 is the only borderline row: Welch 0.051 vs paired 0.032; S6 0.154 vs 0.077; calcified densities 0.063 vs 0.159 (Welch is lower here, still above 0.05). No Welch p is below 0.05.
- **Lanczos vs area: mostly agree in sign and size; three differences.** S2 is more negative under lanczos (-0.0065, seed CI [-0.0128, -0.0002], 0/5) than area (-0.0042); the S7/S9/S10 tail groups go from ~0 (area) to -0.0030 / -0.0022 / -0.0044 (lanczos, seed CIs +-0.014 to +-0.024, include zero); Hydropneumothorax flips from +0.0058 to -0.0252 (CI +-0.08 to +-0.11, i.e. noise from 7 positives). S6 (+0.0036 / +0.0036) and calcified densities (+0.0063 / +0.0053) are stable across the two interpolation modes.
- The resize choice (train LINEAR vs eval LANCZOS4 vs INTER_AREA) moves tail estimates by about the size of the effects being searched for, so no tail subset conclusion rests on one interpolation mode.

### 3.8 Ensemble-level view (post hoc, added after the run; `analysis/entropy_edge_cases_ens.py`)
Averaging all seeds of an arm removes most of the training noise, so the difference of two ensembles is the cleanest estimate of the systematic gap that the validation images allow. Cells: ensemble(C) - ensemble(A or B) [paired image-bootstrap 95% interval, B = 2000], and, for 512 px, the Holm-adjusted image-bootstrap p within the family in brackets. Only the image-bootstrap p exists for one pair of ensembles, so it carries no training-seed uncertainty and is not the decision statistic of 1.5. 768 px is area eval with flip TTA; 512 px is no TTA, lanczos.

| ID | Subset | 768 area, 5 seeds each | 512, 6 seeds each: C - A | 512, 6 seeds each: C - B |
|---|---|---|---|---|
| ALL | all views, 30 classes (reference) | -0.0011 [-0.0081, +0.0038] | +0.0032 [+0.0009, +0.0055] | -0.0006 [-0.0048, +0.0026] |
| S1 | frontal | -0.0035 [-0.0149, +0.0100] | +0.0038 [+0.0006, +0.0068] (0.136) | -0.0020 [-0.0097, +0.0032] (1.000) |
| S2 | frontal & unseen patient | -0.0023 [-0.0157, +0.0056] | +0.0113 [+0.0015, +0.0333] (0.081) | +0.0113 [+0.0004, +0.0326] (0.350) |
| S3 | lateral | +0.0011 [-0.0054, +0.0086] | -0.0017 [-0.0085, +0.0035] (1.000) | -0.0035 [-0.0101, +0.0020] (1.000) |
| S4 | AP_horizontal | +0.0000 [-0.0084, +0.0069] | -0.0021 [-0.0106, +0.0058] (1.000) | -0.0046 [-0.0177, +0.0031] (1.000) |
| S5 | abnormal only (>=1 non-Normal label) | -0.0010 [-0.0074, +0.0031] | +0.0038 [+0.0014, +0.0062] (0.030) | +0.0003 [-0.0031, +0.0030] (1.000) |
| S6 | >=3 findings | +0.0055 [-0.0025, +0.0253] | +0.0047 [-0.0025, +0.0187] (0.732) | -0.0022 [-0.0180, +0.0122] (1.000) |
| S7 | tail group, all views | -0.0042 [-0.0338, +0.0169] | +0.0112 [+0.0020, +0.0207] (0.136) | -0.0053 [-0.0230, +0.0078] (1.000) |
| S8 | tail group, frontal | -0.0144 [-0.0619, +0.0433] | +0.0136 [+0.0011, +0.0264] (0.170) | -0.0127 [-0.0453, +0.0091] (1.000) |
| S9 | rare10, all views | -0.0032 [-0.0241, +0.0117] | +0.0082 [+0.0015, +0.0149] (0.136) | -0.0029 [-0.0151, +0.0063] (1.000) |
| S10 | extreme tail, all views | -0.0061 [-0.0412, +0.0180] | +0.0108 [+0.0003, +0.0217] (0.170) | -0.0071 [-0.0275, +0.0079] (1.000) |
| C:Hydropneumothorax | Hydropneumothorax (7 val+) | -0.0416 [-0.2337, +0.0644] | -0.0123 [-0.0404, +0.0053] (0.672) | -0.0243 [-0.0938, +0.0149] (1.000) |
| C:pneumoperitoneo | pneumoperitoneo (9 val+) | +0.0085 [-0.0772, +0.0958] | +0.0146 [-0.0001, +0.0464] (0.380) | -0.0354 [-0.1257, +0.0064] (1.000) |
| C:Subcutaneous Emphysema | Subcutaneous Emphysema (29 val+) | +0.0066 [-0.0180, +0.0344] | +0.0217 [-0.0187, +0.0721] (0.718) | +0.0115 [-0.0331, +0.0623] (1.000) |
| C:Mass | Mass (49 val+) | -0.0038 [-0.0337, +0.0087] | +0.0035 [-0.0116, +0.0221] (0.718) | -0.0077 [-0.0236, +0.0018] (0.720) |
| C:azygos lobe | azygos lobe (58 val+) | -0.0046 [-0.0199, +0.0100] | +0.0217 [+0.0050, +0.0423] (0.030) | +0.0215 [+0.0037, +0.0435] (0.140) |
| C:Pneumothorax | Pneumothorax (71 val+) | -0.0017 [-0.0241, +0.0204] | +0.0157 [-0.0068, +0.0396] (0.672) | -0.0085 [-0.0294, +0.0122] (1.000) |
| C:calcified densities | calcified densities (140 val+) | +0.0073 [-0.0016, +0.0169] | +0.0133 [+0.0041, +0.0235] (0.014) | +0.0056 [-0.0035, +0.0146] (1.000) |

- **768 px: nothing.** No interval excludes zero in any of the 17 rows (smallest Holm-adjusted p 0.819, calcified densities +0.0073 [-0.0016, +0.0169]); the same under lanczos (largest positive ensemble gaps: S6 +0.0034, calcified densities +0.0048). The tail groups are slightly negative (S7 -0.0042, S8 -0.0144, S10 -0.0061). The five-seed Entropy ensemble does not beat the five-seed ASL ensemble on any subset.
- **512 px C - A: several image-level hits that do not carry to 768 px.** The six-seed ensemble gap is +0.0032 [+0.0009, +0.0055] on all views, with intervals excluding zero on 9 of the 17 rows (frontal, frontal & unseen, abnormal-only, the four tail groups, azygos lobe, calcified densities) and Holm below 0.05 on three (abnormal-only 0.030, azygos lobe 0.030, calcified densities 0.014). Against that, the same subsets are -0.0010 (abnormal-only), -0.0046 (azygos lobe) and +0.0073 [-0.0016, +0.0169] (calcified densities) at 768 px, and the tail groups are -0.0032 to -0.0144. These intervals ignore which training seeds were drawn: the seed-level test of the same all-view gap (6 seeds) has p_t 0.323 and the 2026-10-12 permutation test of the six-seed ensemble gap p 0.268.
- **512 px C - B: the gate adds nothing to the ungated triplet.** All views -0.0006; the tail groups are negative (-0.0029 to -0.0127). Only frontal & unseen (+0.0113 [+0.0004, +0.0326], Holm 0.350) and azygos lobe (+0.0215, Holm 0.140) have intervals excluding zero, and neither survives Holm. Frontal & unseen is the slice that was picked from this result in 1.7.

### 3.9 What the validation set can resolve (768 px, 5 seeds)
The seed-level 80%-power minimum detectable effect (MDE) next to the half-width of the image interval. Image half-width is the floor that no number of extra training seeds can lower, because every inference-only or retraining analysis scores the same 17,270 images.

| ID | Subset | Seed sd of C - A | MDE 80% (5 seeds) | Image half-width | Binding limit |
|---|---|---|---|---|---|
| S1 | frontal | 0.0057 | 0.0095 | 0.0050 | seeds |
| S2 | frontal & unseen patient | 0.0029 | 0.0049 | 0.0096 | images |
| S3 | lateral | 0.0058 | 0.0096 | 0.0034 | seeds |
| S4 | AP_horizontal | 0.0054 | 0.0089 | 0.0051 | seeds |
| S5 | abnormal only (>=1 non-Normal label) | 0.0037 | 0.0061 | 0.0030 | seeds |
| S6 | >=3 findings | 0.0034 | 0.0057 | 0.0068 | images |
| S7 | tail group, all views | 0.0157 | 0.0260 | 0.0126 | seeds |
| S8 | tail group, frontal | 0.0239 | 0.0397 | 0.0212 | seeds |
| S9 | rare10, all views | 0.0102 | 0.0170 | 0.0090 | seeds |
| S10 | extreme tail, all views | 0.0174 | 0.0289 | 0.0144 | seeds |
| C:Hydropneumothorax | Hydropneumothorax (7 val+) | 0.0745 | 0.1238 | 0.0709 | seeds |
| C:pneumoperitoneo | pneumoperitoneo (9 val+) | 0.0569 | 0.0946 | 0.0375 | seeds |
| C:Subcutaneous Emphysema | Subcutaneous Emphysema (29 val+) | 0.0133 | 0.0221 | 0.0212 | both |
| C:Mass | Mass (49 val+) | 0.0119 | 0.0198 | 0.0175 | both |
| C:azygos lobe | azygos lobe (58 val+) | 0.0139 | 0.0231 | 0.0100 | seeds |
| C:Pneumothorax | Pneumothorax (71 val+) | 0.0275 | 0.0457 | 0.0179 | seeds |
| C:calcified densities | calcified densities (140 val+) | 0.0082 | 0.0136 | 0.0087 | seeds |

"Binding limit" = images if the image half-width exceeds the MDE, seeds if the MDE is more than 1.5 times the half-width, otherwise both.

- **Images bind for S2 (+-0.0096, 3,572 images) and S6 (+-0.0068, 2,195 images).** More seeds cannot make an effect below about 0.007 (S6) or 0.010 (S2) resolvable on this split. The S6 point estimate (+0.0036) is half the S6 floor.
- **Seeds bind for the large aggregate subsets** (all views +-0.0030, S5 +-0.0030, S3 +-0.0034, S4 +-0.0051, S1 +-0.0050): there 5 seeds resolve effects of 0.006-0.010 and more seeds would help down to the half-width.
- **Tail groups and single tail classes** resolve effects of 0.017-0.040 (groups) and 0.014-0.124 (classes) only. The two rarest classes (7 and 9 validation positives) are unresolvable.
- Seed pairs needed for 80% power at the observed point estimate and the observed seed sd (non-central t, seed noise only; the image floor above still applies): calcified densities (+0.0063, sd 0.0082) 16 pairs, 11 more than exist, 22 trainings of about 12.5 h; S6 (+0.0036, sd 0.0034) 10 pairs, 5 more, 10 trainings; S2 on ASL's side (-0.0042, sd 0.0029) 6 pairs, 1 more.

### 3.10 Positive directions (closing)
**Which.** By the criterion of 1.6 (point estimate above 0 in 768 px C - A, 512 px C - A and 512 px C - B) one class qualifies: **calcified densities** (+0.0063, +0.0097, +0.0070; table in 3.6). It is hypothesis-generating, not supported: its 768 px Holm-adjusted p_dec is 1.000. The only subset negative in all three contrasts is AP_horizontal (S4), also not significant. The slices that looked favourable at 512 px reverse (S2, -0.0042 at 768 px) or vanish (tail groups, azygos lobe, Subcutaneous Emphysema) at 768 px.

**How large an effect can the data resolve there.** For calcified densities (140 validation positives) 5 seeds detect 0.0136 (MDE) and the image half-width is +-0.0087; the observed +0.0063 is below both, and a true effect of that size could not be confirmed on this validation set even with unlimited seeds, because the image interval would still span zero. The ensemble-level contrasts say the same: +0.0073 [-0.0016, +0.0169] (768 px area), +0.0048 [-0.0050, +0.0140] (768 px lanczos), +0.0133 [+0.0041, +0.0235] (512 px), +0.0056 [-0.0035, +0.0146] against the ungated arm (512 px).

**What would test it properly.**
- **Inference only: nothing further.** Every inference-only analysis (per-seed, ensemble, patient-cluster bootstrap, lanczos or area) draws on the same 17,270 images and the same 10 + 18 trained models. The extra analyses in 3.7 and 3.8 are already the inference-only experiments, and none moves the conclusion.
- **Seeds (not cheap):** 11 more paired 768 px seeds of A and C (about 22 trainings) would fix the seed-level power for calcified densities, but the image floor of +-0.0087 would remain.
- **Data (not available):** an evaluation set with about four times the calcified-densities positives (roughly 560) would halve the image interval; neither the internal split nor the 3,020-image leaderboard set offers it, and the labels of the leaderboard set are hidden.
- **The best-posed alternative, S6 (at least 3 findings).** It has the smallest seed spread (sd 0.0034) and a mechanism (the triplet positives share labels, so co-occurrence is what the loss sees), so about 5 more seed pairs would suffice seed-wise. But its image floor is +-0.0068 on 2,195 images, so only an effect of 0.007 or more could be confirmed; the estimate is +0.0036 at 768 px, +0.0057 at 512 px against ASL and -0.0033 at 512 px against the ungated arm. It is not worth GPU time.

## 4. Recommendation (what the paper may say)

In this exploratory search, **there is no tail, minority-class or subset edge case where Entropy + Triplet Memory beats ASL.** Out of 17 pre-registered 768 px tests (10 subsets, 7 tail classes), none has a Holm-adjusted p_dec below 0.05 (all are 1.000); the same holds for the 512 px C - A and C - B contrasts. The aggregate tie (C - A = +0.0001 at 768 px) is matched by ties on frontal, lateral, abnormal, multi-finding and tail-group slices, with tail-group estimates of +0.0009 to -0.0010 at 768 px. One candidate, calcified densities (positive sign in all three contrasts, +0.0063 at 768 px), and one near-miss, S6 (>=3 findings, +0.0036), are hypothesis-generating only. The slice that favoured C at 512 px (frontal & unseen patient) reverses at 768 px (-0.0042, 0/5 wins), and the 512 px Subcutaneous Emphysema C - B gain (+0.0219, 6/6) is +0.0020 at 768 px.

Wording to keep: these are exploratory subset and per-class analyses, 17 tests per contrast and 51 in total, run after the all-view result was known; the validation set is not patient-disjoint from training (77% of val images share a patient with train), so absolute subset numbers are optimistic; tail classes have 7 to 140 validation positives, so the resolvable effect (MDE at 80% power) is 0.017 to 0.040 for the tail groups and 0.014 to 0.124 for single classes; and seed pairing at 768 px is nominal (unseeded shuffle). The paper should not claim a niche where the entropy gate or the memory bank helps. If a sentence is wanted: "A pre-specified search of ten validation subsets and seven tail classes found no slice in which Entropy + Triplet Memory outperformed ASL after multiplicity correction; the smallest unadjusted decision p was 0.218 at 768 px for calcified densities."

## 5. Caveats
- **Exploratory and multiple.** 17 tests per contrast, 51 in total, run after the all-view result was known; the subset list was fixed before any contrast was computed (section 1), but S2, S5, S6 and the tail rows were suggested by earlier results (1.7).
- **Not patient-disjoint, 30% lateral.** 77% of the 17,270 validation images share a patient with train and 53% share a study (identical labels). Absolute subset numbers are optimistic; the leaderboard set is 3,020 frontal, patient-deduplicated images. S2 (frontal & unseen patient, 3,572 images; no positives for the two rarest tail classes) is the closest proxy and the most image-limited (section 3.9).
- **Nominal pairing at 768 px.** The sample shuffle of the 10 runs was unseeded, so a seed does not mean the same data order; the paired t is a one-sample test on five differences and the Welch test (reported) gives the same picture.
- **Two sources of noise, two intervals.** The seed interval has 4 degrees of freedom; the image interval ignores training noise. The decision statistic needs both; a gain visible in only one (the 512 px class results) is not reported as a finding.
- **Different recipes at the two resolutions.** 512 px: no TTA, `model_best.pth`, lanczos, older recipe, 6 seeds, arms A / B / C; 768 px: flip-TTA SWA, area, 5 seeds, arms A / C. The sign pattern across them is a robustness check, not a pooled estimate.
- **Pre-fix flags.** None apply: `--class-weight-order shard` is in the `args:` line of all 28 training logs (18 at 512 px, 10 at 768 px, all read in this task) and `--metric-embed query` in all 23 logs of the triplet arms (the 512 px ASL arm passes it with lambda 0; the 768 px ASL runs have no metric embedding). Easy mining, the softmax entropy gate and the 7.6%-vs-13% utilisation discrepancy remain open paper confounds; they apply to every triplet arm here.

## 6. Not measured
- Test-set (leaderboard) behaviour of any subset; patient-disjoint validation (not available).
- The no-gate arm B at 768 px with 5 seeds: `unified768/b` has seed 42 only (a dump exists); seeds 123 and 456 stopped after 2 and 1 epochs without SWA. C - B at 768 px therefore does not exist as a paired test.
- Calibration (mECE) on any subset; frontal per-class tail tests (printed as context only); a patient-cluster bootstrap and a Welch test at 512 px.
- Whether a gain of 0.003-0.005 on S1, S3, S4 or S5 exists: the 768 px MDE for those subsets is 0.006-0.010.

## 7. Notable
- **The one slice that favoured Entropy at 512 px (frontal & unseen patient) reverses at 768 px**: C - A = -0.0042 (0/5 wins, seed-level p 0.032 before correction, ASL ahead; lanczos -0.0065). It does not survive the decision rule (image interval [-0.0164, +0.0027]), so it is not an ASL advantage either, but it contradicts the working hypothesis that Entropy helps on patients the model has not seen.
- At 512 px the tail groups favour C over ASL (+0.0047 to +0.0083, 5/6 wins) but C is below the ungated triplet arm B (-0.0052 to -0.0111, 1/6 wins): the tail "gain" is the effect of having a triplet term, not of the entropy gate. At 768 px it is about zero (+0.0009 to -0.0010 for S7-S10).
- The 512 px ensemble-level gains (all views +0.0032, abnormal-only +0.0038, tail +0.0112, azygos lobe +0.0217, calcified densities +0.0133) are absent at 768 px (-0.0011, -0.0010, -0.0042, -0.0046, +0.0073 n.s.).
- 4 of 51 contrasts have p_t below 0.05 before correction (expected by chance: 2.5); one of them is in ASL's favour.

## 8. Next steps
1. Report the 17-test table as an appendix result with the sentence in section 4; do not claim a niche. No GPU time is justified by this search.
2. If a confirmatory test is wanted anyway, S6 is the cheapest in seed terms (about 5 more paired 768 px seeds) but cannot be confirmed below +-0.0068 on this split; calcified densities needs about 11 more pairs and a larger evaluation set.
3. Finishing the B arm (no gate) at 768 px for seeds 123 and 456 would give the missing paired C - B contrast at the paper's resolution (training, not inference).
4. Companion result document (best Entropy configuration and the fair 5-seed ensemble control): [docs/result/2026-10-17-entropy-best-config.md](../result/2026-10-17-entropy-best-config.md). Reproduce: `python -I analysis/entropy_edge_cases.py` (about 50 min on 8 CPUs), `analysis/entropy_edge_cases_ens.py` (about 10 min); outputs `analysis/out/entropy_edge_cases.json` and `analysis/out/entropy_edge_cases_ens.json`.
