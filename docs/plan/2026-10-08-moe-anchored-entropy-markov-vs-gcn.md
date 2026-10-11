
# MoE-anchored Triplet Memory and Markov Prior deltas, and what they say against CheXGCN — 2026-10-08

Type: findings / comparison note (not a training-run report).
Builds on (provenance not repeated here): [2026-10-08-entropy-vs-moe-baseline.md](2026-10-08-entropy-vs-moe-baseline.md), [../result/2026-10-08-stage1-original-baseline.md](../result/2026-10-08-stage1-original-baseline.md), [2026-10-05-v5-rerun-claims-check.md](2026-10-05-v5-rerun-claims-check.md), [../result/2026-10-05-v5-rerun-repro.md](../result/2026-10-05-v5-rerun-repro.md).
All our numbers are internal validation (17,270 images, not patient-disjoint), not leaderboard. Naming follows the paper: "Triplet Memory" = entropy-gated triplet with cross-batch memory bank (code v3); "Triplet Memory + Markov Prior" = patient-history prior into the ML-Decoder queries (code v5).

## Question
Table III of `docs/paper/isbi/paper.tex` (`tab3`, lines 248-276) has no MoE row. Using the MoE Stage 2 as the reference: (1) how much does Triplet Memory improve over MoE, (2) how much does the Markov Prior add on top, and (3) what can the two deltas say against CheXGCN (Chen et al., IEEE JBHI 24(8), 2020)?

## Data and method
- MoE: the single published b12 number (mAP 0.405, mF1 0.402, mAUC 0.882, mECE 0.130; 384 px; 3 d.p.; no seeds, no s.d.; no checkpoint, history, config or log in the repo; see the two documents above). Rounding alone gives each MoE value an uncertainty of +/-0.0005.
- Table III rows: values re-checked against artifacts earlier today (supplied by the orchestrator). I additionally recomputed from the cached flip-TTA probability dumps (`analysis/out/probs_*`) the four metrics of every ensemble row below with `torchmetrics` exactly as in `evaluate/evaluate.py` (macro AP, macro AUROC, macro F1 at 0.5, `BinaryCalibrationError` 15 bins L1 on flattened probabilities). All reproduce Table III: Markov 4-model with 0.4847 / 0.9177 / 0.5025 / 0.1330, withheld 0.4667 / 0.9135 / 0.4845 / 0.1355; Triplet Memory fixed 4-model 0.4699 / 0.9154 / 0.4819 / 0.1385; greedy 3-model 0.4722 / 0.9144 / 0.4769 / 0.1337.
- Matched Triplet Memory baseline for the Markov increment: the 2-model flip average of `push__img768_dp01_seed42` and `push__img768_dp01_seed1024` SWA (`probs_w01_flip`, identical to the dumps in `probs_ab_flip`, max |d| = 0), recomputed: mAP 0.4659, mAUC 0.9126, mF1 0.4817, mECE 0.1352. (mF1 and mECE for this matched baseline are new; the earlier reports gave mAP/mAUC only.)
- Paired bootstrap: 2,000 resamples of the 17,270 images (image level, identical weights for both arms, seed 2026), percentile 95% CI, on all four metrics, using a weighted re-implementation of AP/AUC/F1/ECE that matches torchmetrics to <= 4e-7 on every arm at unit weights. Script `scratchpad/work/boot.py` (session scratchpad), raw output `scratchpad/work/res.json`; archive suggestion: copy to `analysis/moe_anchored_boot.py` and `analysis/out/moe_anchored_boot.json`. Ran on the login node, about 7.5 min, no Slurm.
- CheXGCN: read from the PDF in `docs/pdf/` (Tables II/III text, Section IV). No jobs launched, no existing file modified.

## Findings

### 1. MoE-anchored table (all four metrics, deltas = row minus MoE)
MoE = 0.405 / 0.882 / 0.402 / 0.130 (mAP / mAUC / mF1 / mECE; 384 px). mECE: negative delta is better. "Res" is the evaluation resolution; a dagger marks rows where no resolution-matched MoE exists.

| Row | Res (px) | mAP | mAUC | mF1 | mECE | d mAP | d mAUC | d mF1 | d mECE |
|---|---|---|---|---|---|---|---|---|---|
| Stage 1 (no entropy, no MoE), history.json ep 6 | 384 | 0.3849 | 0.8746 | 0.3946 | 0.1338 | -0.0201 | -0.0074 | -0.0074 | +0.0038 |
| Stage 1, same weights re-evaluated (job 47073) | 512 | 0.3688 | 0.8643 | 0.3689 | 0.1408 | -0.0362 | -0.0177 | -0.0331 | +0.0108 |
| Original impl., 5 seeds (pre-fix) | 512 | 0.414 | 0.889 | 0.407 | 0.143 | +0.0090 | +0.0070 | +0.0050 | +0.0130 |
| Triplet Memory best epoch, 6 seeds | 512 | 0.439 | 0.900 | 0.454 | 0.128 | +0.0340 | +0.0180 | +0.0520 | -0.0020 |
| Triplet Memory SWA s86 | 768 | 0.4566 | 0.9063 | 0.4676 | 0.1275 | +0.0516 | +0.0243 | +0.0656 | -0.0025 |
| + drop-path SWA s42 | 768 | 0.4600 | 0.9092 | 0.4685 | 0.1354 | +0.0550 | +0.0272 | +0.0665 | +0.0054 |
| + drop-path SWA s1024 | 768 | 0.4582 | 0.9104 | 0.4741 | 0.1354 | +0.0532 | +0.0284 | +0.0721 | +0.0054 |
| + drop-path SWA s86 | 1024 | 0.4619 | 0.9138 | 0.4679 | 0.1438 | +0.0569 | +0.0318 | +0.0659 | +0.0138 |
| Triplet Memory fixed 4-model avg, flip | 768-1024 | 0.4699 | 0.9154 | 0.4819 | 0.1385 | +0.0649 | +0.0334 | +0.0799 | +0.0085 |
| Triplet Memory greedy 3-model, flip (selected on val) | 640-1024 | 0.4722 | 0.9144 | 0.4769 | 0.1337 | +0.0672 | +0.0324 | +0.0749 | +0.0037 |
| Triplet Memory + Markov Prior, 4 models, flip, history supplied | 768 | 0.4847 | 0.9177 | 0.5025 | 0.1330 | +0.0797 | +0.0357 | +0.1005 | +0.0030 |
| Triplet Memory + Markov Prior, 4 models, flip, history withheld | 768 | 0.4667 | 0.9135 | 0.4845 | 0.1355 | +0.0617 | +0.0315 | +0.0825 | +0.0055 |

Sources: Stage 1 384 `train/checkpoint3/Model_20260119_062652/history.json`; Stage 1 512 `eval_result.stage1.json` (job 47073); other rows as supplied in the brief and re-verified for the ensemble rows (above). Deltas computed from the 4-decimal values; the Original impl. and best-epoch rows carry 3-decimal values from the paper.

How to read the deltas:
- **There is no like-for-like row at matched resolution except Stage 1 vs MoE at 384 px** (the only comparison where both sides are 384 px and both start from the same Stage-1 checkpoint): MoE adds +0.0201 mAP, +0.0074 mAUC, +0.0074 mF1 and improves mECE by 0.0038 over Stage 1. Every Triplet Memory / Markov row is at 512 px or higher, and no Stage-2 run exists at 384 px.
- **The Original impl. vs MoE gap (+0.009 mAP, +0.007 mAUC, +0.005 mF1, +0.013 mECE worse) is confounded by 384 vs 512 px.** The only measurement of what resolution does to this lineage is Stage 1 itself: the same weights lose 0.0161 mAP, 0.0103 mAUC, 0.0257 mF1 (and gain 0.0070 mECE) when scored at 512 instead of 384. That is a train/test mismatch for a 384-trained model, not the effect of training at 512, so it cannot be applied to the MoE row in either direction. In the other direction, Triplet Memory itself gains from resolution when trained there: seed 86, 0.4385 (512, best epoch) to 0.4560 (768) to 0.4619 (1024, with drop-path/SWA; paper line 278 and Table III), about +0.017 and +0.006 mAP. If MoE also gained a similar amount when trained and scored at 512 or more, the true Triplet Memory minus MoE gap is smaller than any delta in the table. If MoE lost something at 512 (as Stage 1 does without adaptation), it is larger. The sign and size are not measurable from current artifacts. Treat the MoE-anchored deltas as upper-bound-style indications for the higher-resolution rows.
- **Like-for-like chain at 512 px (all rows trained at 512, one Stage-1 start, no MoE needed):** Stage 1 @512 0.3688 -> Original impl. 0.414 (+0.045 mAP, +0.025 mAUC, +0.038 mF1, +0.002 mECE) -> Triplet Memory best epoch 0.439 (a further +0.025 mAP, +0.011 mAUC, +0.047 mF1, -0.015 mECE). These steps do not need the MoE number. The Original impl. -> Triplet Memory step bundles the class-weight-order and metric-embedding fixes and the new recipe (lr 1e-4, bank 2048, EMA, 8-epoch schedule); no triplet-off control exists (see the 2026-10-08 entropy-vs-MoE note).
- **Calibration.** The only Triplet Memory rows that beat MoE's mECE (0.130) are the 512-px best epoch (0.128) and SWA s86 768 (0.1275), by 0.002 to 0.003, which is within a few times the +/-0.0005 rounding of the MoE value and without any MoE s.d.; all other rows are worse by 0.004 to 0.014. The Markov rows are +0.003 (supplied) and +0.0055 (withheld) worse than MoE. The data do not support a calibration advantage over MoE.
- **MoE has no seeds and no s.d.** No significance statement about any MoE delta is possible. For scale, the Triplet Memory seed s.d. is 0.002 (6 seeds, 512 px; Original impl. 0.002 at 5 seeds), but the MoE run-to-run spread is unknown.
- **Pre-fix flag.** The Original impl. row predates both fixes (CSV-order class weights, legacy embedding); the Triplet Memory rows are post-fix. The Original impl. - MoE delta therefore describes the pre-fix method.

Decomposition of "MoE to Triplet Memory + Markov Prior (history)" = +0.0797 mAP, indicative only because of the resolution mismatch: MoE to Original impl. +0.009 (resolution-confounded, pre-fix); Original impl. to Triplet Memory best epoch +0.025 (fixes + recipe, at 512); to 768/1024-px SWA singles about +0.018 to +0.023 (resolution, drop-path, SWA, seeds mixed); flip TTA + 4-model averaging about +0.005 to +0.01; Markov Prior with history +0.0188 (below). The steps are approximate because consecutive rows differ in seed and recipe, as the paper itself says.

### 2. Markov increment, against the matched Triplet Memory baseline (not against MoE)
Baseline TM2 = Triplet Memory, 2 SWA models (seed 42 and seed 1024, 768 px), flip: **0.4659 / 0.9126 / 0.4817 / 0.1352** (mAP / mAUC / mF1 / mECE).
Markov arms are the 768 px SWA models of the original run (O) and the rerun (R), seeds 42 and 1024, flip; "supplied" = patient history given, "withheld" = prior zeroed (the test-set condition). Point values from torchmetrics; deltas Markov minus TM2 with paired image-level bootstrap 95% CI (2,000 resamples).

| Arm (models) | mAP | mAUC | mF1 | mECE |
|---|---|---|---|---|
| Markov 4-model, supplied | 0.4847 | 0.9177 | 0.5025 | 0.1330 |
| Markov 4-model, withheld | 0.4667 | 0.9135 | 0.4845 | 0.1355 |
| O 2-model, supplied | 0.4867 | 0.9170 | 0.4980 | 0.1332 |
| O 2-model, withheld | 0.4693 | 0.9131 | 0.4811 | 0.1358 |
| R 2-model, supplied | 0.4797 | 0.9169 | 0.4904 | 0.1328 |
| R 2-model, withheld | 0.4611 | 0.9128 | 0.4706 | 0.1352 |

| Delta vs TM2 | d mAP [95% CI] | d mAUC [95% CI] | d mF1 [95% CI] | d mECE [95% CI] (negative = better) |
|---|---|---|---|---|
| **4-model, supplied** | **+0.0188 [+0.0101, +0.0282]** | **+0.0051 [+0.0031, +0.0069]** | **+0.0208 [+0.0133, +0.0287]** | **-0.0022 [-0.0023, -0.0021]** |
| **4-model, withheld** | **+0.0008 [-0.0052, +0.0082]** | **+0.0009 [+0.0000, +0.0018]** | **+0.0028 [-0.0027, +0.0093]** | **+0.0003 [+0.0003, +0.0004]** |
| O 2-model, supplied | +0.0208 [+0.0130, +0.0295] | +0.0044 [+0.0019, +0.0066] | +0.0163 [+0.0083, +0.0243] | -0.0020 [-0.0021, -0.0019] |
| O 2-model, withheld | +0.0033 [-0.0033, +0.0112] | +0.0005 [-0.0005, +0.0015] | -0.0006 [-0.0060, +0.0059] | +0.0006 [+0.0005, +0.0007] |
| R 2-model, supplied | +0.0138 [+0.0047, +0.0217] | +0.0043 [+0.0022, +0.0064] | +0.0087 [-0.0059, +0.0223] | -0.0025 [-0.0026, -0.0024] |
| R 2-model, withheld | -0.0048 [-0.0128, +0.0017] | +0.0002 [-0.0008, +0.0012] | -0.0111 [-0.0248, +0.0010] | +0.0000 [-0.0000, +0.0001] |
| 4-model supplied minus 4-model withheld (same models, the prior's own contribution) | +0.0179 [+0.0087, +0.0261] | +0.0042 [+0.0022, +0.0060] | +0.0181 [+0.0120, +0.0240] | -0.0025 [-0.0026, -0.0025] |

Against other Triplet Memory references (4-model Markov vs 4-model or 3-model Triplet Memory, not matched in members or seeds):

| Markov 4-model vs | d mAP [95% CI] | d mAUC [95% CI] | d mF1 [95% CI] | d mECE |
|---|---|---|---|---|
| TM fixed 4-model flip (0.4699 / 0.9154 / 0.4819 / 0.1385), supplied | +0.0148 [+0.0052, +0.0249] | +0.0023 [-0.0001, +0.0046] | +0.0206 [+0.0093, +0.0343] | -0.0055 |
| same, withheld | -0.0032 [-0.0093, +0.0043] | -0.0019 [-0.0029, -0.0008] | +0.0026 [-0.0070, +0.0161] | -0.0030 |
| TM greedy 3-model flip (0.4722 / 0.9144 / 0.4769 / 0.1337), supplied | +0.0124 [+0.0024, +0.0236] | +0.0033 [+0.0009, +0.0057] | +0.0256 [+0.0122, +0.0407] | -0.0007 |
| same, withheld | -0.0055 [-0.0114, +0.0031] | -0.0009 [-0.0020, +0.0005] | +0.0076 [-0.0052, +0.0222] | +0.0018 |
| (reference) TM fixed 4-model minus TM2 | +0.0040 [-0.0014, +0.0098] | +0.0028 [+0.0016, +0.0039] | +0.0002 [-0.0124, +0.0099] | +0.0033 |

Reading:
- **With history supplied the Markov Prior adds about +0.019 mAP (CI excludes 0), +0.005 mAUC, +0.021 mF1 and slightly better mECE (-0.002) over the matched Triplet Memory baseline.** This is the longitudinal simulation (5,759 of 17,270 val images have an earlier state; Section "Evaluation" of the paper) and its leaderboard effect is 0 because no test image has history.
- **With history withheld (the only test-set condition) the increment is indistinguishable from 0 on mAP, mF1 (CIs span 0) and +0.0009 on mAUC (CI lower bound 0.0000).** The two ensembles disagree in sign (O +0.0033, R -0.0048 mAP); the withheld increment is not a stable positive. Against the larger 4-model and 3-model Triplet Memory references it is negative on mAUC (-0.0019, CI excludes 0, vs the fixed 4-model average). The Markov models were trained to rely on the prior, and zeroing it is a distribution shift at inference.
- **The 4-vs-2 model count is a mild confound**: the Markov arm averages 4 models (2 seeds x original/rerun), the matched baseline 2. The reference line (TM fixed 4-model minus TM2: +0.004 mAP, +0.003 mAUC, with different members) suggests that a 4-model Triplet Memory baseline would absorb roughly +0.003 to +0.004 of the +0.019. The 2-vs-2 rows (O: +0.021, R: +0.014 mAP) bracket the 4-model estimate and avoid the confound.
- Run-to-run variation of the Markov models is larger than the bootstrap CI conveys. The CI is the image-sampling uncertainty with the models fixed. The rerun reproduces the prior's effect at a smaller size (+0.014 vs +0.021, see `docs/result/2026-10-05-v5-rerun-repro.md`: rerun minus original -0.0070 [-0.0143, -0.0011] supplied), and v5 is not bit-reproducible. The image-level CIs above agree with the patient-clustered (300 resamples) CIs in the earlier report to within about 0.002 (e.g. O supplied +0.0208 [+0.0132, +0.0285], R supplied +0.0138 [+0.0042, +0.0209]), so the 77% patient overlap does not change the picture for these paired contrasts.
- mECE CIs are very narrow because ECE is an aggregate over 518,100 prediction entries and the paired difference is nearly deterministic; read them as image-sampling only.

### 3. CheXGCN comparison
CheXGCN (Chen, Li, Lu, Yu, Zhang, IEEE JBHI 24(8), 2020; read from `docs/pdf/Label_Co-Occurrence_Learning_...pdf`): 14 labels; ChestX-Ray14 (70/10/20 split; images resized 556 and cropped 512) and CheXpert; AUC only; two GCN layers (300 -> 1024 -> 1664) map GloVe label embeddings to the classifier weights; graph from co-occurrence counts thresholded at phi 0.3 and re-weighted with lambda (0.2 on ChestX-Ray14, 0.3 / 0.1 on the two CheXpert variants); the baseline is DenseNet-169 with an FC layer.

| Quantity | Metric | Value |
|---|---|---|
| CheXGCN minus DenseNet-169 FC, ChestX-Ray14 (14 labels) | mean AUC | +0.012 (0.814 -> 0.826) |
| CheXGCN minus DenseNet-169 FC, CheXpert U-Ones | mean AUC | +0.009 (0.818 -> 0.827) |
| CheXGCN minus DenseNet-169 FC, CheXpert U-Zeros | mean AUC | +0.009 (0.823 -> 0.832); the text says "0.7%", which does not match its own numbers (0.832 - 0.823 = 0.009) |
| Ours: Stage 1 -> MoE (384, like-for-like) | mAUC | +0.007 |
| Ours: MoE -> Original impl. (512, resolution-confounded, pre-fix) | mAUC | +0.007 |
| Ours: MoE -> Triplet Memory best epoch (512) | mAUC | +0.018 |
| Ours: MoE -> Triplet Memory fixed 4-model, flip (768-1024) | mAUC | +0.033 |
| Ours: Triplet Memory 2-model -> Markov Prior, history supplied | mAUC | +0.0051 [+0.0031, +0.0069] |
| Ours: Triplet Memory 2-model -> Markov Prior, history withheld | mAUC | +0.0009 [+0.0000, +0.0018] |
| Ours: label-to-label co-occurrence chain (route A, the nearest analogue of CheXGCN), post hoc on the greedy ensemble | mAP only | -0.0006 to -0.0459 (paper Table `tab:mk-graph`); trained label-graph layer (v4) mAP delta -0.003 to -0.011, one seed (2026-10-01 verdict); mAUC not computed |

Why the numbers are not directly comparable (each alone would block a ranking):
1. **Dataset and labels.** 14 coarse labels on ChestX-Ray14 / CheXpert (NIH / Stanford, US) vs 30 labels on PadChest / PadChest-GR (Spain) with a long tail: 7 tail and 6 extreme-tail classes, one with 38 training images. A macro average over 30 labels with rare-class AUCs from a handful of positives behaves differently.
2. **Metric family.** CheXGCN reports AUC only. Our primary metric is mAP; mAUC is the only shared family, and AUC is the less sensitive metric for long-tailed labels (it is insensitive to prevalence, and our mAUC saturates near 0.91-0.92, so the headroom differs from CheXGCN's 0.81-0.83).
3. **Baseline strength.** CheXGCN's reference is a vanilla DenseNet-169 + FC. Ours is the MoE Stage 2, itself a strong ConvNeXt + ML-Decoder system with per-class queries; ML-Decoder already conditions each class query on the others through query self-attention, which is a learned form of label interaction. The room for a co-occurrence prior is not the same.
4. **What the delta isolates.** CheXGCN's delta isolates a label graph (but with no graph-only ablation; it also swaps the classifier for GCN-generated weights). Our MoE-to-Triplet-Memory delta isolates nothing (resolution, recipe, fixes, SWA, TTA, ensembling all change; no triplet-off control). Our Markov delta isolates a **patient-history** prior, not a label co-occurrence graph, and is unavailable at test time; the true analogue of CheXGCN (route A) was negative here.
5. **Selection on evaluation curves.** CheXGCN selects phi and lambda by sweeping and reading average AUC (Fig. 6; the paper text does not say which split the curves use and I did not confirm that it is the test split; the brief states test). Our ensemble rows include a greedy forward selection on the same validation fold; the Markov and fixed-ensemble rows have none.
6. **Resolution and training scale.** CheXGCN: 512 px, DenseNet-169, two Titan XP GPUs. Ours: 384 (MoE) vs 512-1024 px, ConvNeXt, ensembles and TTA.
7. **Statistical treatment.** CheXGCN reports single runs without CIs, and its prose overstates some per-class gains (the Fibrosis and Hernia pairs "0.819 vs. 0.834" and "0.903 vs. 0.929" are printed in the CheXGCN-first order of the other pairs, which would be losses; I did not check them against Table II). We report seeds or paired CIs where available; MoE has neither.
8. **Splits.** CheXGCN follows the ChestX-Ray14 official split (patient-disjoint by construction); ours is 77% patient-overlapping with train.

Magnitude only, with all caveats: CheXGCN's +0.009 to +0.012 AUC is of the same order as our Markov Prior's +0.005 mAUC with history and larger than its +0.001 without; our MoE-to-Triplet-Memory mAUC delta (+0.018 to +0.033) is 1.5 to 3 times CheXGCN's, but it bundles resolution and ensembling. These are not like-for-like and must not be presented as "ours beats CheXGCN".

**The one sentence the paper can defensibly say** (cite the JBHI paper; needs a new `\bibitem`):
"For context, CheXGCN [x] reports a gain of 0.009 to 0.012 mean AUC over a DenseNet-169 classifier from a label co-occurrence graph on 14-label ChestX-Ray14 and CheXpert; in our 30-label long-tailed setting the patient-history Markov Prior adds 0.005 mAUC (95% CI 0.003 to 0.007) over the matched Triplet Memory ensemble when the previous report is available and 0.001 without it, and label-to-label co-occurrence chains did not help, so the two are not comparable and we do not rank the methods."

### 4. Recommendation for Table III
Table I (`tab1`, lines ~196-205) already has the MoE row (and the Stage-1 row) in order mAP, mF1, mAUC, mECE; Table III (order mAP, mAUC, mF1, mECE) has neither. Recommended:
1. **Add an MoE row to Table III**, plus the Stage 1 rows, as a labelled top block, with a dagger footnote. Do not add delta columns (the deltas are resolution-confounded; give them in prose only, and only against like-for-like rows).
2. **Add a matched Triplet Memory 2-model flip row** so the Markov Prior increment is read against its own baseline (the existing greedy-3 and fixed-4 rows are not matched in members).
3. Prose: state the Markov increment against the matched baseline with CIs, and the MoE comparison only as indicative.

LaTeX (do not paste without review; the paper was not edited):

```latex
% ---- new block at top of tab3 tabular (after the header \hline) ----
Stage 1 (no entropy, no MoE)~\cite{b12} & 384 & 0.385 & 0.875 & 0.395 & 0.134 \\
Stage 1, same weights re-evaluated & 512 & 0.369 & 0.864 & 0.369 & 0.141 \\
Stage 2: MoE~\cite{b12}$^{\dagger}$ & 384 & 0.405 & 0.882 & 0.402 & 0.130 \\
\hline
% ---- new row directly above the two Markov Prior rows ----
Triplet Memory, 2 SWA models (seeds 42, 1024), flip, matched to the Markov rows & 768 & 0.4659 & 0.9126 & 0.4817 & 0.1352 \\
```

Caption sentence to append to the `tab3` caption:
```latex
$^{\dagger}$Value as published in~\cite{b12} (three decimals); a single run at 384~px with no checkpoint, seeds or standard deviation available, so all other rows (512--1024~px) differ from it in resolution as well as in method, and no difference to it is tested for significance. The re-evaluated Stage-1 row uses the same weights as the 384~px row: the resolution change alone moves mAP by $-0.016$.
```
Text sentence for the Markov increment (after the "What this does not show" paragraph or in Section on Markov results):
```latex
Against the matched two-seed Triplet Memory flip ensemble (last-but-two row of Table~\ref{tab3}), the Markov Prior adds $+0.019$ mAP (95\% CI $+0.010$ to $+0.028$), $+0.005$ mAUC ($+0.003$ to $+0.007$), $+0.021$ mF1 and $-0.002$ mECE when the previous report is supplied, and $+0.001$ mAP ($-0.005$ to $+0.008$) and $+0.001$ mAUC ($0.000$ to $+0.002$) when it is withheld (paired bootstrap over 17{,}270 images, 2{,}000 resamples; the CIs do not include model-to-model variation, and the original and rerun model pairs differ by $0.007$ mAP).
```
Bibitem for CheXGCN (page range left out because I did not verify it): `\bibitem{bGCN} Bingzhi Chen, Jinxing Li, Guangming Lu, Hongbing Yu, and David Zhang, ``Label co-occurrence learning with graph convolutional networks for multi-label chest X-ray image classification,'' IEEE Journal of Biomedical and Health Informatics, vol. 24, no. 8, Aug. 2020.` (Authors, title, volume/issue and date taken from the PDF header.)

Also fix, while editing: the paper's sentence "The calibration regression observed for both Stage-2 variants" (line ~211) is wrong for MoE on the paper's own Table I (MoE mECE 0.130 < Stage 1 0.134); flagged earlier in the 2026-10-08 entropy-vs-MoE note.

### 5. What would make the MoE comparison sound (sketches only; nothing launched)
Ranked by cost:
1. **384-px evaluation of the existing Triplet Memory checkpoints** (cheap, about 35 min each, one GPU). Settles nothing about training at 384, but gives a resolution-matched lower bound for Triplet Memory (the models are trained at 512-1024, so the mismatch penalises them) and, run on Stage 1, settles the 0.016 gap hypothesis: `sbatch scripts/eval_perclass.sh train/checkpoint3/Model_20260119_062652/model_best.pth 384 res384` (the same script with a Triplet Memory `model_swa.pth` / `model_best.pth` for the 512-px seeds). If a Triplet Memory model scored at 384 still exceeds 0.405, "Triplet Memory beats MoE" survives even the unfavourable resolution.
2. **Train Triplet Memory at 384 px** from the same Stage-1 checkpoint: `train/train_2_v3.py --img-size 384 --stage1-ckpt train/checkpoint3/Model_20260119_062652/model_best.pth` with the remaining args of the Table-III 512-px `args:` line (job 46457 fix_only recipe or the 6-seed `seedsweep` args), seeds 86, 123, 1024 at least, plus its triplet-off control (`--triplet-lambda 0`). Gives a same-resolution, seeded, post-fix row against MoE's 0.405 and the seed s.d. MoE lacks. Check `args:` in `/home/psytp7/logs/` before launching.
3. **Re-run MoE.** No code is wired in: `convnext3.py` is not imported, the MoE plumbing in `convnext.py` was removed on 2026-08-25 (it may be recoverable from git history before that date; not checked here), and the b12 MoE recipe (2 more epochs, most backbone parameters frozen) would have to be reconstructed. Run 3 seeds at 384 and at 512 to get the MoE s.d. and the 384-vs-512 effect on MoE directly. Highest cost, only way to remove the unknown MoE resolution effect.
4. **Leaderboard check.** MoE has a test number (0.4599, 3rd place, development / test, frontal, patient-disjoint); no Triplet Memory model has been scored there. A submission of a no-history Triplet Memory ensemble (the Markov Prior with history cannot be used on test) would give the only like-for-like MoE-vs-Triplet-Memory number on the metric that matters, with no resolution confound at the model-selection level (any resolution may be used on both).

## Caveats
- **Resolution.** MoE is 384 px; every Triplet Memory / Markov row is 512-1024 px. The MoE-anchored deltas, including the headline +0.0797 mAP for Triplet Memory + Markov Prior (history supplied), are indicative, not causal. No resolution-matched Stage-2 comparison exists in the repo.
- **MoE row:** a single published number, 3 d.p., no seeds, no s.d., no artifact; fold identity (b12: about 17,000 validation images; ours 17,270) unverified; "additional 2 epochs, mostly frozen backbone" is b12 text, unchecked.
- **Original impl. row is pre-fix** (CSV class-weight order, legacy embedding, softmax entropy, `>=1` shared label, bank 96). Triplet Memory rows are post-fix (v3). I did not re-read the v5 `args:` lines in this analysis (`--metric-embed query`, `--class-weight-order shard` are assumed from the earlier v5 reports), so the Markov arms inherit that assumption.
- **Selection.** The greedy 3-model row is selected on the validation fold (optimistic); the fixed 4-model and Markov rows are not; best-epoch rows (Stage 1, MoE, 512-px best epochs) are max-over-epochs on the scoring fold.
- **Validation fold is not patient-disjoint** (77% patient overlap with train, 53% share their StudyID, 30% lateral). The Markov with-history result is a longitudinal simulation; leaderboard effect 0.
- **Bootstrap** resamples images with models fixed; it excludes seed-to-seed variation. The Markov original-vs-rerun gap (0.007 to 0.008 mAP) is 3-4 times the Triplet Memory seed s.d. of about 0.002. One validation fold; two seeds per arm; the 4-model Markov ensemble mixes original and rerun members.
- **Matched baseline is 2 models vs 4.** See Finding 2.
- **GCN comparison** uses the paper's printed numbers; CheXGCN's baseline was presumably retrained by its authors; I did not audit it.

## Not measured
- Any Stage-2 run at 384 px; Triplet Memory scored at 384 px; MoE at 512 px.
- Triplet-off or gate-off control.
- mAUC for the label-graph (route A) arms.
- Head / medium / tail breakdown of the MoE-anchored deltas (no MoE per-class result exists) or of the Markov increment on the 4-model ensemble in this note (earlier reports give per-group results for the 2-seed arms).
- Leaderboard score of any Triplet Memory or Markov model.
- Patient-disjoint evaluation of the Markov with-history effect (impossible by design).

## Notable
- **Table I already contains the MoE row; Table III does not.** The request is satisfied by adding the MoE and Stage-1 rows to Table III, not by introducing a new baseline.
- **The matched Markov increment with history withheld (the only test-time condition) is ~0:** mAP +0.0008 [-0.0052, +0.0082], mAUC +0.0009 [0.0000, +0.0018] for the 4-model arm, with original and rerun disagreeing in sign on mAP (+0.0033 vs -0.0048). The headline 0.4847 is a history-supplied simulation; the paper's Table III already shows both conditions, but the "Markov Prior" label can be read as a usable improvement.
- **The label-co-occurrence analogue of CheXGCN was negative in our setting**; what helps is a patient-history prior, a different mechanism. The CheXGCN comparison should therefore be framed as context on magnitude, not as a head-to-head.
- The MoE-anchored deltas are resolution-confounded in an unknown direction; the repo's only quantification of resolution (Stage 1, 384 -> 512: -0.016 mAP, -0.010 mAUC) applies to a train/test mismatch and does not transfer to MoE.
- CheXGCN's text says 0.7% for the U-Zeros gain, whereas its own numbers give 0.9 points (0.823 -> 0.832).

## Next steps
1. Orchestrator: decide on adding the three rows and the matched 2-model row to Table III (LaTeX above) and the CheXGCN bibitem; archive `boot.py` / `res.json` into `analysis/` if the CIs are cited in the paper.
2. Cheapest MoE de-confounding: 384-px evaluation of Stage 1 (res384) and of the Triplet Memory 512-px checkpoints (about 35 min each).
3. If a stronger claim than "indicative" is wanted: train Triplet Memory at 384 px with 3 seeds (and the triplet-off control), or re-implement MoE; neither was launched.
4. Update memory `paper-revision-confounds` with: Table I has MoE and Table III does not; the withheld-history Markov increment is ~0 with sign instability; MoE-anchored deltas are resolution-confounded.
