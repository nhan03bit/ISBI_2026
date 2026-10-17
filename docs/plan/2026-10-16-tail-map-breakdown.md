# Where do the arms and the resize fix differ? The minority (tail) classes, 2026-10-16

**Question (user).** The resize fix gave only a slight positive change, and the arms tie on macro mAP. Is the all-class benchmark the wrong lens, so that a difference on the minority classes is being averaged away? Also: does `train_2_v0.py` (ASL only) accidentally use Triplet Memory?

**Answer.**
- **v0 does not use Triplet Memory.** Checked in the code, the saved checkpoints, the training logs and the eval logs (section 1).
- **Entropy (C) against ASL (A): null on the tail as well.** Tail AP −0.003 (all views) / −0.002 (frontal), 2/5 seeds, both intervals span 0. The macro average is not hiding an Entropy effect.
- **Markov with history (Dwith): the gain grows toward the tail, but it is patient-history recall, not image reading.**
  - The gain is +0.008 head, +0.019 medium, +0.024 tail on all views (5/5 seeds at every level).
  - On the tail it does not survive the image bootstrap (all views [−0.001, +0.055]).
  - It sits on tail positives whose patient already had the same finding in a training image. Same network, history on vs off: +0.036 on those (image interval [+0.005, +0.063]) and −0.034 on the others ([−0.058, +0.001]; rare10 [−0.045, −0.002]) (section 3).
  - On patients absent from train fold 1, the head and medium gain is about +0.002, and its interval includes 0.
- **Markov withheld (Dzero): no tail effect.** Its +0.013 frontal tail comes from one class (Hydropneumothorax, 7 val positives).
- **The resize fix is mostly a tail effect.** Same weights, `INTER_AREA` instead of `INTER_LANCZOS4` at eval, over all 26 runs:
  - all views: macro +0.0035, tail +0.0096 (23/26 runs up, image interval [−0.0001, +0.0216])
  - frontal: macro +0.0068, tail +0.0214 (26/26 runs up, image interval [+0.0056, +0.0344]), against medium +0.0027 and head +0.0004
  - the classes that gain are thin-line and small-air findings (Pneumothorax, Subcutaneous Emphysema, pneumoperitoneum, azygos lobe, Nodule); large-shape findings lose a little in every run (section 4)
  - so the user's hunch holds for the resize fix: the 30-class average dilutes a tail-concentrated gain about 3×
- **The internal val can resolve tail contrasts only down to about ±0.015 (all views) or ±0.02–0.05 (frontal).** Two of the seven tail classes have fewer than 10 val positives (section 5).

**Tools and outputs** (offline, no training; flip-TTA SWA dumps of the 5-seed unseeded Table 1 runs, lanczos unless stated):
- `analysis/tail_breakdown.py` → `analysis/out/tail_breakdown.json`, log `analysis/out/tail_breakdown_full.log`. Groups, per class, patient-history split; 1,000 image-bootstrap resamples.
- `analysis/eval_interp_table.py` → `analysis/out/eval_interp_table.json`, log `analysis/out/eval_interp_table.log`.
  - new section: area − lanczos per group and per class (300 resamples), and per group for the training arms
  - path fix: it now finds the training-arm dumps, which `submit_stage2_v0.sh` writes to `probs_v0_flip_<tag>/` with only a `_area` file suffix
- Groups are by train prevalence (`evaluate/class_groups.py`, CXR-LT 2024 thresholds):
  - head > 10%: 3 classes
  - medium 1–10%: 20 classes
  - tail < 1%: 7 classes
  - extreme tail < 0.5%: 6 classes
  - rare10: the 10 rarest
- Frontal = PadChest Projection PA / AP / AP_horizontal (11,973 of 17,270 val images).
- Arms: A = ASL v0, C = Entropy, Dwith / Dzero = Markov prior with / with withheld history.
- Two intervals for every contrast:
  - **seed**: paired t-CI over the 5 seeds (training noise)
  - **image**: 95% bootstrap over val images of the 5-seed mean difference (val sampling noise, which dominates on the tail)

## 1. Does v0 use Triplet Memory? No
| Check | Result |
|---|---|
| Code, `train/train_2_v0.py` | The only loss is ASL on `model(x)[0]`; the embedding output is discarded. AdamW sees only `ConvNeXt2` parameters. No memory bank, triplet, gate, prior or MoE code; nothing is imported from v5. |
| Model class | `from convnext import ConvNeXt2` resolves to `train/convnext.py` (`train_2_v0.sh` runs `cd train`), which returns `(logits, embedding)`. |
| Saved weights | All 8 v0 `model_swa.pth` (5 Table 1 seeds, seeded, sched4, trainarea) hold exactly the 420 Stage-1 keys; none match memory / triplet / gate / prior / expert. |
| Training logs | Every v0 run logs `Stage-1 init: checkpoint3/Model_20260119_062652/model_best.pth (missing=0 unexpected=0)` and `v0 (ASL only)`; no triplet or memory lines. |
| Eval logs | `evaluate_tta_v5.py` loads every v0 checkpoint as plain `ConvNeXt2 (missing=0 unexpected=0)`, so no randomly initialised memory or prior module is added at eval. |
| Stage-1 init | Trained by the repo-root `train.py`: ASL only (γ− 2, γ+ 0, clip 0.05), no triplet term in the current file (docs/result/2026-10-08-stage1-original-baseline.md). The same checkpoint initialises every v5 arm, so it cannot separate v0 from v5. |

**Trap noted:** the repo-root `convnext.py` is the Stage-1 model file (imported by the root `train.py`). Its `ConvNeXt2.forward` returns bare logits, so if a Stage-2 script imported it, `model(x)[0]` would pick the first sample's logits. v0 is safe only because the job script `cd`s into `train/`. ASL equalling Entropy is explained by the near-inactive triplet term (`docs/plan/2026-10-15-triplet-memory-debug.md`), not by contamination.

## 2. Arms by frequency group
Group mean AP (5-seed mean):

| Group (classes) | View | ASL | Entropy | Markov, history | Markov, withheld |
|---|---|---|---|---|---|
| Head (3) | all | 0.7632 | 0.7644 | 0.7714 | 0.7629 |
| Medium (20) | all | 0.4620 | 0.4621 | 0.4806 | 0.4596 |
| Tail (7) | all | 0.3317 | 0.3287 | 0.3553 | 0.3361 |
| Rare10 (10) | all | 0.3541 | 0.3519 | 0.3757 | 0.3556 |
| Head (3) | frontal | 0.7767 | 0.7778 | 0.7834 | 0.7767 |
| Medium (20) | frontal | 0.4790 | 0.4790 | 0.4930 | 0.4770 |
| Tail (7) | frontal | 0.3713 | 0.3697 | 0.4069 | 0.3841 |
| Rare10 (10) | frontal | 0.3923 | 0.3909 | 0.4219 | 0.3996 |

Contrasts against ASL: mean (seed wins of 5); image interval.

| Group | View | Entropy − ASL | Markov history − ASL | Markov withheld − ASL |
|---|---|---|---|---|
| Head | all | +0.0012 (5/5); [+0.0005, +0.0021] | +0.0082 (5/5); [+0.0065, +0.0099] | −0.0003 (2/5); [−0.0012, +0.0005] |
| Medium | all | +0.0001 (2/5); [−0.0008, +0.0009] | +0.0186 (5/5); [+0.0158, +0.0216] | −0.0024 (0/5); [−0.0034, −0.0014] |
| Tail | all | −0.0030 (2/5); [−0.0148, +0.0069] | +0.0236 (5/5); [−0.0009, +0.0550] | +0.0044 (1/5); [−0.0110, +0.0260] |
| Rare10 | all | −0.0022 (2/5); [−0.0103, +0.0051] | +0.0216 (5/5); [+0.0047, +0.0445] | +0.0015 (1/5); [−0.0094, +0.0168] |
| Head | frontal | +0.0011 (5/5); [+0.0002, +0.0021] | +0.0067 (5/5); [+0.0048, +0.0086] | +0.0000 (1/5); [−0.0010, +0.0010] |
| Medium | frontal | +0.0000 (3/5); [−0.0010, +0.0011] | +0.0141 (5/5); [+0.0109, +0.0177] | −0.0020 (1/5); [−0.0031, −0.0008] |
| Tail | frontal | −0.0016 (2/5); [−0.0210, +0.0156] | +0.0356 (5/5); [−0.0124, +0.0830] | +0.0128 (4/5); [−0.0155, +0.0511] |
| Rare10 | frontal | −0.0014 (2/5); [−0.0152, +0.0105] | +0.0296 (5/5); [−0.0036, +0.0619] | +0.0073 (4/5); [−0.0126, +0.0345] |

Seed t-CIs for the tail rows:
- Entropy: [−0.020, +0.014] all views, [−0.035, +0.032] frontal
- Markov with history: [+0.010, +0.037] all, [+0.011, +0.060] frontal
- Markov withheld: [−0.019, +0.028] all, [−0.021, +0.047] frontal

The extreme-tail group (6 classes) follows the tail row in every case. Full table: `tail_breakdown_full.log`.

- **Entropy:** the only interval clear of 0 is head +0.001, which is negligible. Tail and rare10 are centred near −0.002.
- **Markov withheld** is slightly below ASL on medium (−0.002, both intervals exclude 0) and level elsewhere.
- **Markov with history** is consistent across training seeds at every level. On the tail, val sampling cannot confirm it (image intervals include 0).

Per class, the 10 rarest (all views, 5-seed mean AP; differences against ASL with seed wins):

| Class | Train + | Val + (frontal) | ASL | Entropy − ASL | Markov history − ASL | Markov withheld − ASL |
|---|---|---|---|---|---|---|
| Hydropneumothorax | 38 | 7 (4) | 0.297 | −0.025 (2) | +0.063 (4) | +0.081 (4) |
| pneumoperitoneo | 52 | 9 (5) | 0.306 | +0.001 (2) | −0.049 (0) | −0.008 (2) |
| Subcutaneous Emphysema | 179 | 29 (23) | 0.586 | +0.002 (3) | +0.018 (4) | −0.013 (2) |
| Mass | 296 | 49 (34) | 0.076 | −0.004 (1) | +0.035 (5) | −0.005 (2) |
| azygos lobe | 346 | 58 (44) | 0.557 | +0.001 (2) | +0.012 (4) | −0.004 (3) |
| Pneumothorax | 420 | 71 (50) | 0.367 | −0.001 (2) | +0.038 (4) | −0.016 (2) |
| calcified densities | 835 | 140 (91) | 0.133 | +0.005 (4) | +0.049 (5) | −0.005 (1) |
| Emphysema | 1,232 | 206 (129) | 0.285 | −0.001 (2) | +0.018 (5) | −0.001 (2) |
| aortic atheromatosis | 1,430 | 239 (169) | 0.168 | −0.003 (1) | +0.018 (5) | −0.006 (1) |
| Hernia | 1,641 | 274 (185) | 0.766 | +0.004 (3) | +0.015 (5) | −0.008 (0) |

- No class favours Entropy in more than 4 of 5 seeds.
- The Markov-with-history gains are largest on findings that persist across a patient's studies: Mass, calcified densities, Pneumothorax.
- The Markov-withheld tail mean is carried by Hydropneumothorax (+0.081 on 7 positives); the other six tail classes are at or below ASL.

## 3. Patient history: where the Markov gain comes from
77% of internal-val images (13,304 / 17,270) belong to a patient with a train-fold image. Of the tail-class val positives, 55–77% per class have the same finding in a train image of the same patient. The two rarest classes have no positives at all among patients absent from train.

Tail positives split by same-class patient history. Negatives are all val images; 5-seed mean (seed wins); image interval:

| Tail positives | ASL AP | Entropy − ASL | Markov history − ASL | Markov withheld − ASL | Markov history − withheld |
|---|---|---|---|---|---|
| Patient had the finding in train (258) | 0.305 | −0.006 (2/5); [−0.027, +0.008] | **+0.030 (5/5); [−0.002, +0.061]** | −0.006 (1/5); [−0.018, +0.006] | **+0.036 (5/5); [+0.005, +0.063]** |
| Patient did not (105) | 0.234 | +0.005 (4/5); [−0.011, +0.016] | **−0.016 (0/5); [−0.047, +0.044]** | +0.019 (5/5); [−0.016, +0.070] | **−0.034 (0/5); [−0.058, +0.001]** |

Over the 10 rarest classes, history − withheld is +0.035 [+0.013, +0.054] on positives with history and −0.028 [−0.045, −0.002] on positives without. Both intervals exclude 0.

Patients with no train-fold image (3,966 val images):

| Group | ASL AP | Markov history − ASL | Markov history − withheld |
|---|---|---|---|
| Head | 0.739 | +0.002 (3/5); [−0.001, +0.004] | +0.000 (5/5); [−0.000, +0.001] |
| Medium | 0.445 | +0.002 (3/5); [−0.001, +0.006] | +0.001 (5/5); [−0.000, +0.003] |
| Tail (5 classes with positives, 59 positives) | 0.435 | −0.055 (0/5); [−0.132, +0.009] | −0.001 (1/5); [−0.007, +0.002] |

On images whose patient has a train image, the medium gain is +0.022 (5/5), against +0.002 on unseen patients.

- **History helps where the patient's earlier label repeats and hurts where the finding is new.** Markov history − withheld is the same network evaluated with and without the prior.
- This is label recall through the train/val patient overlap (the internal val is not patient-disjoint: `docs/plan/2026-10-01-padchest-metadata-eda.md`), consistent with the 0 leaderboard gain of the patient-history models.
- The −0.055 unseen-patient tail is not a real deficit. All three v5 arms show it against ASL. Most of it is Subcutaneous Emphysema with 2 positives (ASL 0.87 against 0.64), plus Pneumothorax with 4. The image intervals span 0: Entropy − ASL [−0.105, +0.023], Markov history − ASL [−0.132, +0.009], Markov withheld − ASL [−0.130, +0.010].

## 4. The resize fix by frequency group
Area − lanczos at eval, same weights, all 26 runs with both dumps: the 5 seeded seed-42 runs (A, C, Dwith, Dzero, P), the 20 unseeded Table 1 runs (A, C, Dwith, Dzero × 5 seeds) and sched4. Mean over runs (runs up); 95% image bootstrap of the mean:

| Group | All views | Frontal |
|---|---|---|
| Macro (30) | +0.0035 (26/26) [+0.0010, +0.0065] | +0.0068 (26/26) [+0.0030, +0.0102] |
| Head (3) | +0.0002 (18/26) [−0.0013, +0.0014] | +0.0004 (23/26) [−0.0013, +0.0021] |
| Medium (20) | +0.0019 (26/26) [+0.0005, +0.0032] | +0.0027 (26/26) [+0.0010, +0.0043] |
| Tail (7) | **+0.0096 (23/26) [−0.0001, +0.0216]** | **+0.0214 (26/26) [+0.0056, +0.0344]** |
| Rare10 (10) | +0.0078 (26/26) [+0.0002, +0.0160] | +0.0160 (26/26) [+0.0041, +0.0254] |

Intervals: 300 image resamples (`eval_interp_table.log`). Extreme tail: +0.0107 [−0.0001, +0.0247] all views, +0.0244 [+0.0064, +0.0399] frontal.

Per class, frontal (runs up of 26):

| Largest gains | Losses in at least 25 of 26 runs |
|---|---|
| Hydropneumothorax +0.0621 (22; 4 positives) | hypoexpansion −0.0057 (0) |
| pneumoperitoneo +0.0342 (20; 5 positives) | vascular hilar enlargement −0.0056 (0) |
| Pneumothorax +0.0221 (26; 50 positives) | aortic elongation −0.0014 (1) |
| Subcutaneous Emphysema +0.0205 (24; 23 positives) | Support Devices −0.0014 (1) |
| Nodule +0.0147 (26) | cardiomegaly −0.0013 (0) |
| azygos lobe +0.0086 (26), Hernia +0.0085 (26), bronchiectasis +0.0071 (26) | |

**Reading.** The originals are about 2,800–3,000 px and are shrunk about 4×. `INTER_LANCZOS4` interpolates with a fixed 8-tap kernel in source pixels, so like `INTER_LINEAR` it does not low-pass to the output resolution; `INTER_AREA` box-averages the source pixels under each output pixel, which antialiases.
- The findings that gain are thin lines or small air or density structures: the pleural line of a pneumothorax, subdiaphragmatic free air, the azygos fissure, subcutaneous air streaks, nodules. Aliasing breaks these up.
- The findings that lose slightly are judged by large-scale shape (heart size, lung volume, hila, aorta). There the slightly softer antialiased image costs a little against the aliased training input.
- Macro mAP weights the tail 7/30, so it shows a quarter of the tail gain.

**The two retrain arms** (seed 42, single runs, flip-TTA SWA), against the seeded ASL baseline under the same eval filter; 95% image intervals:

| Arm (eval) | Macro | Frontal | Head | Medium | Tail (all / frontal) |
|---|---|---|---|---|---|
| Train with area, job 52967 (area vs area) | 0.4660 vs 0.4660: −0.0000 [−0.0061, +0.0047] | +0.0024 [−0.0067, +0.0127] | +0.0010 | +0.0010 | −0.0035 / +0.0063, both intervals span 0 |
| LR fully annealed in 4 epochs, sched4, job 52969 (area vs area) | 0.4630 vs 0.4660: −0.0030 [−0.0075, +0.0018] | +0.0043 [−0.0065, +0.0134] | −0.0028 [−0.0042, −0.0016] | −0.0039 [−0.0054, −0.0026] | −0.0006 / +0.0299, both intervals span 0 |
| sched4 (lanczos vs lanczos) | 0.4595 vs 0.4605: −0.0010 [−0.0053, +0.0033] | +0.0049 [−0.0049, +0.0144] | −0.0023 | −0.0027 | +0.0043 / +0.0279, both intervals span 0 |

- **Training with area adds nothing beyond evaluating with area.** The whole gain is the eval filter. Against the lanczos-evaluated baseline it looked like +0.0054, which is the eval filter alone.
- **Annealing the LR within the 4 trained epochs hurts head and medium classes.** Val mAP peaks at epoch 2 and falls to 0.4445 by epoch 4 as the LR reaches 0 (train loss 0.029 against 0.032). The "LR still at 54% of peak" explanation for the plateau is ruled out.
- Both are single runs (same-seed reruns differ by about 0.005), so only the head / medium loss of sched4 is resolved.

## 5. How small a tail effect can this val set see?
- **Tail val positives per class:** 7, 9, 29, 49, 58, 71, 140 (frontal 4, 5, 23, 34, 44, 50, 91). The two rarest classes have no positives among unseen patients.
- **Image-interval half-width for a 5-seed contrast on the tail mean:** about ±0.011 on all views and ±0.018 on frontal for arms that make similar predictions (Entropy − ASL). It is ±0.03 to ±0.05 when the arms disagree more (Markov history − ASL).
- **Seed intervals are of the same size,** about ±0.017 (all views) and ±0.034 (frontal) for Entropy − ASL. A 5-seed tail contrast under about 0.02 (all views) or 0.03–0.04 (frontal) cannot be resolved on this split.
- **One class can carry the tail mean:** Hydropneumothorax (7 positives) moves the 7-class mean by up to 0.012 on its own.

## Recommendations
1. **Use `INTER_AREA` for every reported eval.** It is free, it lifts the tail by +0.010 (all views) / +0.021 (frontal), and the leaderboard is frontal.
   - The 8 area re-evaluations that hit the 3 h limit (52942, 52944, 52951–52953, 52962–52964) all ran on node rogue, which made no progress. Every one on dart / tamar finished in about 47 min. They were resubmitted as jobs 53131–53138 with rogue excluded (`submit_reeval_interp.sh` now has `ONLY=` and `EXCLUDE=`, default rogue) and all finished on 2026-10-16.
   - **Table 1 under area** (5 seeds, mean ± sd): ASL 0.4651 ± 0.0030 (frontal 0.4899), Entropy 0.4652 ± 0.0032 (0.4896), Markov with history 0.4830 ± 0.0029 (0.5080), withheld 0.4649 ± 0.0029 (0.4941).
     - Paired, Entropy − ASL is +0.0001 [−0.0042, +0.0045], Markov withheld − ASL −0.0002 [−0.0062, +0.0058] (frontal +0.0042 [−0.0021, +0.0105]).
     - Every conclusion from the lanczos table holds under area (`eval_interp_table.log`).
2. **Report tail AP next to macro mAP, with both the seed and the image interval.** Do not claim tail differences smaller than about 0.02 (all views) or 0.03 (frontal) from this split.
3. **Entropy:** the tail confirms the null result; no further per-class mining is warranted.
4. **Markov with history:** do not present its tail (or medium) gain as image-level improvement. If it appears in the paper, show the unseen-patient row or the history split next to it.
