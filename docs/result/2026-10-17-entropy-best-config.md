# Best configuration of Entropy + Triplet Memory Bank (ranking of every available checkpoint and fixed ensemble) — 2026-10-17

**Status.** The candidate list, the five ensembles, the control and the submission-candidate rule (section 1) were written to disk at 16:28 BST on 2026-10-17 (file mtime), before any metric of this document was computed. Sections 2 onward are filled in after the run. Companion plan document (subsets where Entropy might beat ASL): [docs/plan/2026-10-17-entropy-edge-cases.md](../plan/2026-10-17-entropy-edge-cases.md).

**Directory:** `train/checkpoint_Triplet_3/{unified768,unified768_seeded,push,res_sweep,seedsweep,sweep2}/` (Entropy + Triplet Memory); control `train/checkpoint_ASL/v0_768/`
**Script / recipe:** no new training. Offline scoring of cached flip-TTA probability dumps with `analysis/entropy_best_config.py` (writes `analysis/out/entropy_best_config.json`). Entropy checkpoints come from two code paths, never pooled in a mean: `train/train_2_v5.py` (arm c, 768 px) and `train/train_2_v3.py` (push, res_sweep, seedsweep, sweep2). Control `train/train_2_v0.py`.
**Slurm job(s):** training jobs of the candidates are listed in section 2; no job was submitted for this report.

## 1. Pre-registration (fixed before any result)

### 1.1 Candidates
Everything is flip-TTA, lanczos eval (the one filter every run has), uniform mean of probabilities for ensembles, each member at its own native test scale (the training resolution). All rows share the row order of `analysis/out/probs_unified_flip/keys.txt` (17,270 internal-val images).

Single Entropy / v3 entropy-gated checkpoints (16):

| ID | Checkpoint (under `train/checkpoint_Triplet_3/`) | Weights | Scale | Code |
|---|---|---|---|---|
| U42, U123, U456, U789, U1024 | `unified768/c/seed_<S>/Model_run/` | `model_swa.pth` | 768 | v5 |
| S42 | `unified768_seeded/c/seed_42/Model_run/` (seeded sample shuffle) | `model_swa.pth` | 768 | v5 |
| P768-42, P768-1024 | `push/img768_dp01_seed{42,1024}/Model_run/` | `model_swa.pth` | 768 | v3 |
| P896-86 | `push/img896_dp01_seed86/Model_run/` | `model_swa.pth` | 896 | v3 |
| P1024-86 | `push/img1024_dp01_seed86/Model_run/` | `model_swa.pth` | 1024 | v3 |
| R640-86, R640-1024 | `res_sweep/img640_seed{86,1024}/Model_run/` | `model_best.pth` | 640 | v3 |
| R768-86 | `res_sweep/img768_seed86/Model_run/` | `model_best.pth` | 768 | v3 |
| W512-42, W512-1024 | `seedsweep/seed_{42,1024}/Model_run/` | `model_best.pth` | 512 | v3 |
| D512-86 | `sweep2/dp01/seed_86/Model_run/` | `model_best.pth` | 512 | v3 |

Ensembles, membership fixed now (no member is added or dropped after seeing results):

| ID | Members | Size |
|---|---|---|
| E1 | U42, U123, U456, U789, U1024 (the 5 Table 1 Entropy seeds) | 5 |
| E2 | E1 + S42 | 6 |
| E3 | P768-42, P768-1024, P896-86, P1024-86 | 4 |
| E4 | E1 union E3 | 9 |
| E5 | P768-42, P1024-86, R640-86 (the old greedy-3, reference; its members were chosen on this same val set) | 3 |

Control, not eligible for the rule: **A5** = the 5 ASL `train_2_v0.py` seeds 42 / 123 / 456 / 789 / 1024 (`train/checkpoint_ASL/v0_768/seed_<S>/Model_run/model_swa.pth`, 768, flip-TTA), same recipe and size as E1. The five ASL singles and the seeded ASL seed 42 are printed as reference rows.

Not rankable (no flip-TTA probability dump exists; a GPU evaluation would be needed): `push/cont1024_from768_s86`, and the regularisation-wave runs `reg/img1024_dp01_seed{42,1024}` (no-TTA SWA mAP 0.4618 and 0.4567 in their logs), `reg/img768_*`. They are left out of the ranking rather than scored from different artefacts.

### 1.2 Metrics reported for every row (lanczos, internal val)
All-view mAP; frontal mAP (PA / AP / AP_horizontal, 11,973 images); frontal & unseen-patient mAP (3,572 images; the val patient has no image in `data/train_fold_1.csv`); tail AP (mean AP of the 7 classes with train prevalence < 1%, all views); mECE (15-bin L1 on all probabilities, all views, lower is better). Subset mAP = macro AP over the classes with at least 1 positive and 1 negative in the subset (the `entropy_ablation.subset_map` convention).

### 1.3 Submission-candidate rule
1. Eligible: the 16 single checkpoints and E1-E5.
2. Score: frontal mAP (lanczos, flip-TTA).
3. T = the eligible row with the highest score. W = every eligible row whose score is within 0.002 of T's score.
4. Winner = the row of W with the fewest member checkpoints (a single checkpoint counts as 1, so a single within 0.002 of T would win); ties go to the higher frontal mAP.
5. The report gives the winner's exact checkpoint paths and test scales. The choice is made on the internal validation split, which is also the early-stopping and SWA-epoch-selection set, is not patient-disjoint (77% of val images share a patient with train) and is 30% lateral. It is therefore optimistic (winner's curse); section 3.4 measures how optimistic. (Cross-reference renumbered after the run; no change of substance.)

### 1.4 Fair control and uncertainty
- E1 against A5 (same recipe, same size), paired image bootstrap (B = 2000, 17,270 images resampled together for both arms) of the difference in all-view mAP, frontal mAP, frontal & unseen mAP and tail AP, under lanczos and again under area. Area dumps: `analysis/out/probs_unified_flip_area/` and `analysis/out/probs_v0_flip_area/`.
- Ranking stability: the same image bootstrap applied to all 21 candidates gives the share of resamples in which each is first by frontal mAP, and each candidate's difference from E1 with a 95% interval.
- Selection optimism: 200 random patient-level halves of the val set; the rule above is applied on one half and the pick is scored on the other.
- Never mix lanczos and area in one comparison.

## 2. Configuration (from the `args:` line of every training log; all 16 single checkpoints read)
Shared by all 16 singles: Stage-1 init `checkpoint3/Model_20260119_062652/model_best.pth` (missing=0 unexpected=0), `--memory-size 2048 --metric-embed query --head-classes 5 --triplet-lambda 0.1`, `--class-weight-order shard`, anchor rule `mean` (the entropy gate; the default of both scripts), lr 1e-4. The control A5 uses the same Stage-1 init and `--class-weight-order shard` and no memory or triplet code.

| Group | Script | Resolution | Batch x accum | Epochs / patience / sched | Drop-path | Weights in the dump | Seeds | Training jobs |
|---|---|---|---|---|---|---|---|---|
| U (Entropy, Table 1) | `train_2_v5.py` | 768 | 4 x 12 | 4 / 2 / 8 | 0.1 | SWA of EMA epochs 2-4 | 42, 123, 456, 789, 1024 | 51832, 51843, 51857, 51871, 51883 |
| S42 (Entropy, seeded sample shuffle) | `train_2_v5.py` | 768 | 4 x 12 | 4 / 2 / 8 | 0.1 | SWA | 42 | 52650 |
| P (push wave) | `train_2_v3.py` | 768, 768, 896, 1024 | 4 x 12 | 4 / 2 / 8 | 0.1 | SWA | 42, 1024, 86, 86 | 49000, 48999, 48998, 48997 |
| R (res sweep) | `train_2_v3.py` | 640, 640, 768 | 4 x 12 | 8 / 3 / cosine over 8 | none | `model_best.pth` (EMA, best epoch) | 86, 1024, 86 | 47094, 47096, 47095 |
| W (seed sweep) | `train_2_v3.py` | 512 | 8 x 6 | 8 / 3 / cosine over 8 | none | `model_best.pth` | 42, 1024 | 47097, 46476 |
| D512-86 | `train_2_v3.py` | 512 | 8 x 6 | 8 / 3 / cosine over 8 | 0.1 | `model_best.pth` | 86 | 46469 |
| A (ASL control) | `train_2_v0.py` | 768 | 4 x 12 | 4 / 2 / 8 | 0.1 | SWA | 42, 123, 456, 789, 1024 | 52402, 52404, 52408, 52410, 52406 |

Known-bug check: `--class-weight-order shard` and `--metric-embed query` are in every one of the 16 logs (the post-fix settings; the `class_weights (shard order)` line is in every log), so no candidate has the near-inverted tail weighting or the pre-fix triplet embedding. Job 46471 (a first seed-42 attempt of group W) was OOM-killed after 42 min; the checkpoint used is from job 47097. The two code paths (v3 for P, R, W, D; v5 for U, S) were never averaged into one mean; only the ensembles E3-E5 mix them, member by member.

Evaluation of every dump: `evaluate/evaluate_tta.py` (v3 dumps, `analysis/out/probs_w1_flip`, `probs_w0_flip`, linked in `probs_w01_flip`) and `evaluate/evaluate_tta_v5.py` (v5 and v0 dumps), flip TTA, single scale equal to the training resolution, INTER_LANCZOS4 (`--interp lanczos`), internal validation split (17,270 rows, same `keys.txt` and `labels.npy` in every directory; asserted). Checks of this report's code: every single's all-view mAP equals `analysis/out/ens_select_w01_flip.json` or the `eval_*_s*.json` of its dump (worst |difference| 4.8e-7, 25 comparisons); E5 reproduces the old 0.4722 (0.472238); the frontal and frontal & unseen numbers equal `entropy_ablation.subset_map`; mECE equals torchmetrics.

## 3. Results (internal validation, lanczos, flip TTA; not the leaderboard)

### 3.1 Ranking, sorted by frontal mAP
Rank in brackets is among the 21 eligible rows. "First" = share of 2000 paired image resamples in which the row has the highest frontal mAP.

| Rank (frontal) | ID | Size | What | All-view mAP (rank) | Frontal mAP | Frontal & unseen mAP (rank) | Tail AP | mECE | First |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **E5** | 3 | P768-42 + P1024-86 + R640-86 (768 / 1024 / 640) | 0.4722 (1) | 0.4968 | 0.4752 (4) | 0.3571 | 0.1337 | 54.9% |
| 2 | **E3** | 4 | P768-42 + P768-1024 + P896-86 + P1024-86 | 0.4699 (2) | 0.4929 | 0.4794 (2) | 0.3497 | 0.1385 | 16.6% |
| 3 | **E4** | 9 | E1 + E3 | 0.4689 (3) | 0.4921 | 0.4781 (3) | 0.3448 | 0.1365 | 5.1% |
| 4 | U123 | 1 | v5, SWA, 768 px | 0.4653 (6) | 0.4915 | 0.4686 (14) | 0.3451 | 0.1351 | 10.4% |
| 5 | P1024-86 | 1 | v3, SWA, 1024 px | 0.4645 (8) | 0.4892 | 0.4719 (8) | 0.3430 | 0.1438 | 4.7% |
| 6 | **E1** | 5 | U42 + U123 + U456 + U789 + U1024 | 0.4658 (4) | 0.4890 | 0.4733 (7) | 0.3363 | 0.1349 | 0.0% |
| 7 | **E2** | 6 | E1 + S42 | 0.4657 (5) | 0.4877 | 0.4738 (5) | 0.3358 | 0.1350 | 0.0% |
| 8 | P768-42 | 1 | v3, SWA, 768 px | 0.4650 (7) | 0.4875 | 0.4706 (11) | 0.3475 | 0.1352 | 4.3% |
| 9 | P768-1024 | 1 | v3, SWA, 768 px | 0.4626 (9) | 0.4862 | 0.4814 (1) | 0.3375 | 0.1352 | 0.9% |
| 10 | S42 | 1 | v5, SWA, 768 px, seeded | 0.4625 (10) | 0.4850 | 0.4714 (10) | 0.3362 | 0.1356 | 2.0% |
| 11 | U789 | 1 | v5, SWA, 768 px | 0.4621 (11) | 0.4838 | 0.4734 (6) | 0.3322 | 0.1355 | 0.4% |
| 12 | U1024 | 1 | v5, SWA, 768 px | 0.4604 (13) | 0.4829 | 0.4715 (9) | 0.3272 | 0.1350 | 0.4% |
| 13 | U42 | 1 | v5, SWA, 768 px | 0.4605 (12) | 0.4824 | 0.4673 (15) | 0.3257 | 0.1346 | 0.1% |
| 14 | P896-86 | 1 | v3, SWA, 896 px | 0.4578 (14) | 0.4804 | 0.4702 (12) | 0.3193 | 0.1397 | 0.2% |
| 15 | R768-86 | 1 | v3, best, 768 px | 0.4570 (16) | 0.4785 | 0.4642 (19) | 0.3267 | 0.1433 | 0.0% |
| 16 | U456 | 1 | v5, SWA, 768 px | 0.4577 (15) | 0.4763 | 0.4668 (16) | 0.3132 | 0.1343 | 0.0% |
| 17 | R640-1024 | 1 | v3, best, 640 px | 0.4526 (17) | 0.4747 | 0.4699 (13) | 0.3175 | 0.1365 | 0.0% |
| 18 | R640-86 | 1 | v3, best, 640 px | 0.4485 (18) | 0.4717 | 0.4514 (21) | 0.3062 | 0.1223 | 0.0% |
| 19 | W512-1024 | 1 | v3, best, 512 px | 0.4480 (19) | 0.4697 | 0.4649 (17) | 0.3134 | 0.1306 | 0.0% |
| 20 | W512-42 | 1 | v3, best, 512 px | 0.4433 (20) | 0.4652 | 0.4545 (20) | 0.2971 | 0.1301 | 0.0% |
| 21 | D512-86 | 1 | v3, best, 512 px | 0.4429 (21) | 0.4641 | 0.4644 (18) | 0.2853 | 0.1367 | 0.0% |

Controls, not eligible (same metrics):

| Row | All-view mAP | Frontal | Frontal & unseen | Tail AP | mECE |
|---|---|---|---|---|---|
| **A5** (5 ASL seeds, the fair control of E1) | 0.4672 | 0.4896 | 0.4818 | 0.3421 | 0.1325 |
| ASL singles s42 / s123 / s456 / s789 / s1024 | 0.4656 / 0.4614 / 0.4590 / 0.4641 / 0.4585 | 0.4911 / 0.4828 / 0.4784 / 0.4861 / 0.4797 | 0.4775 / 0.4815 / 0.4704 / 0.4789 / 0.4719 | 0.3476 / 0.3315 / 0.3188 / 0.3398 / 0.3207 | 0.1335 / 0.1318 / 0.1322 / 0.1324 / 0.1328 |
| ASL single, seeded s42 | 0.4605 | 0.4803 | 0.4713 | 0.3269 | 0.1327 |
| Mean (sd) of the 5 Entropy singles U | 0.4612 (0.0028) | 0.4834 (0.0054) | 0.4695 (0.0028) | 0.3287 (0.0115) | 0.1349 (0.0005) |
| Mean (sd) of the 5 ASL singles | 0.4617 (0.0031) | 0.4836 (0.0052) | 0.4760 (0.0047) | 0.3317 (0.0123) | 0.1325 (0.0007) |

Reading:
- **First place depends on the metric.** E5 is first on all-view (0.4722) and frontal (0.4968) mAP. On frontal & unseen-patient mAP, the proxy closest to the leaderboard set, the single P768-1024 is first (0.4814), then E3 (0.4794), E4 (0.4781); E5 is fourth (0.4752). Gaps on that subset are 0.002-0.006 on 3,572 images, well inside the noise (section 3.4).
- **Best single checkpoint:** by all-view mAP U123 (0.4653), then P768-42 (0.4650) and P1024-86 (0.4645); by frontal mAP U123 (0.4915). The five 768 px Entropy seeds alone span 0.4577-0.4653 all-view (sd 0.0028) and 0.4763-0.4915 frontal (sd 0.0054), so single-checkpoint gaps below about 0.005 (all-view) or 0.01 (frontal) are seed noise. The 1024 px single (P1024-86) sits inside the spread of the 768 px SWA singles on all three metrics (all-view 0.4645 against 0.4577-0.4653; frontal 0.4892 against 0.4763-0.4915; frontal & unseen 0.4719 against 0.4668-0.4814), so no 768-to-1024 px effect is visible in single runs.
- **Older, smaller runs trail.** The 640 px and 512 px checkpoints (best-epoch EMA weights instead of SWA, and no drop-path except D512-86) are 0.009-0.018 below the mean of the five 768 px Entropy singles (0.4612) on all-view mAP and have tail AP 0.285-0.318. They matter only as diversity inside E5.
- **mECE** separates the groups, not the winners: v5 768 px runs 0.1343-0.1356, 896-1024 px runs 0.1397-0.1438, ensembles 0.1337-0.1385, the 512-640 px best-epoch checkpoints 0.122-0.137. Among the ensembles E5 (0.1337) is the best calibrated; the ASL ensemble A5 (0.1325) is better still.

### 3.2 The pre-registered rule and the submission candidate
Rule applied to frontal mAP: **T = E5 (0.4968)**. The window is 0.4948 and above; the next row is E3 at 0.4929 (0.0039 below), so W = {E5}. **Winner: E5**, three checkpoints, evaluated at each member's own training resolution with flip TTA:

| Member | Checkpoint (absolute path) | Test scale | Weights | Own all-view / frontal mAP (lanczos, flip) |
|---|---|---|---|---|
| P768-42 | `/home/psytp7/ISBI_2026/train/checkpoint_Triplet_3/push/img768_dp01_seed42/Model_run/model_swa.pth` | 768 | SWA | 0.4650 / 0.4875 |
| P1024-86 | `/home/psytp7/ISBI_2026/train/checkpoint_Triplet_3/push/img1024_dp01_seed86/Model_run/model_swa.pth` | 1024 | SWA | 0.4645 / 0.4892 |
| R640-86 | `/home/psytp7/ISBI_2026/train/checkpoint_Triplet_3/res_sweep/img640_seed86/Model_run/model_best.pth` | 640 | EMA best epoch (3) | 0.4485 / 0.4717 |

- **Ensemble:** uniform mean of the three flip-TTA probability vectors. Internal val (lanczos): all-view 0.4722, frontal 0.4968, frontal & unseen 0.4752, tail 0.3571, mECE 0.1337.
- **Inference command shape** (the existing leaderboard tool, written 2026-10-17): `sbatch scripts/predict_task1.sh --images-dir DIR --template labels.csv --out OUT.csv --interp area <P768-42 path>@768 <P1024-86 path>@1024 <R640-86 path>@640`. The three checkpoints hold the same 420-key plain ConvNeXt2 state dict as the v5 runs (key sets compared; `predict_task1.py` itself was not run on them).
- **Filter:** the table is ranked under lanczos, the one filter all 21 rows have. `predict_task1.py` defaults to `--interp area`, which raised every one of the 26 v5/v0 runs measured with both filters (macro +0.0035, frontal +0.0068; E1 0.4658 to 0.4708 all-view and 0.4890 to 0.4955 frontal under area). It has not been measured for these three v3 checkpoints (no area dumps exist), so the expected gain is an extrapolation; the area number of this ensemble is not known.
- **Selected on internal validation.** The choice rests on the same 17,270 images that picked the SWA epochs and the early-stopping epoch, which are not patient-disjoint (77% of val images share a patient with train) and are 30% lateral. E5's three members were themselves picked greedily on this split by all-view mAP (from 10 dumps), so its lead is optimistic by construction (section 3.4). The leaderboard offset between internal and test mAP is large and was not used here.
- **Equally defensible alternatives:** E3 (four SWA checkpoints, no member selection beyond "the push wave"): frontal 0.4929, all-view 0.4699, frontal & unseen 0.4794; E5 - E3 = +0.0039 frontal [-0.0077, +0.0130], +0.0023 all-view [-0.0041, +0.0073], -0.0042 frontal & unseen [-0.0115, +0.0028]. E4 (nine checkpoints) is within +0.0008 of E3 on frontal.

### 3.3 Fair control: 5 Entropy seeds (E1) against 5 ASL seeds (A5), same recipe and size
Paired image bootstrap, B = 2000 (mECE: B = 500); 95% percentile interval; E1 minus A5.

| Eval filter | All-view mAP | Frontal mAP | Frontal & unseen mAP | Tail AP | mECE |
|---|---|---|---|---|---|
| lanczos | -0.0014 [-0.0062, +0.0016] | -0.0006 [-0.0086, +0.0056] | -0.0085 [-0.0291, +0.0028] | -0.0058 [-0.0259, +0.0064] | +0.0024 [+0.0023, +0.0024] |
| area | -0.0011 [-0.0081, +0.0038] | -0.0035 [-0.0149, +0.0100] | -0.0023 [-0.0157, +0.0056] | -0.0042 [-0.0338, +0.0169] | +0.0024 [+0.0023, +0.0024] |

- **No ensemble-level gain from the Entropy objective.** Five Entropy seeds and five ASL seeds tie on every mAP metric; every point estimate is on ASL's side, no interval excludes zero. The only resolved difference is calibration: the Entropy ensemble has the higher (worse) mECE by 0.0024 in both filters, the same size as the per-seed gap (+0.0024, 5/5 seeds).
- Ensembling five same-recipe seeds adds +0.0046 (E1) and +0.0055 (A5) all-view mAP over the mean single run, +0.0057 and +0.0060 frontal: the same for both arms.
- Under area A5 reaches 0.4720 / 0.4991 (all-view / frontal) against E1's 0.4708 / 0.4955.
- On frontal & unseen mAP, the closest proxy to the test set, ASL is ahead at 768 px by a point estimate of 0.0023 (area) to 0.0085 (lanczos) for the ensembles, and 0.0065 for the mean single run (0.4760 against 0.4695, lanczos); the intervals include 0. The per-seed version is in [2026-10-17-entropy-edge-cases.md](../plan/2026-10-17-entropy-edge-cases.md) (S2).

### 3.4 How much of the ranking is real?
**Differences between the top rows (post hoc, paired image bootstrap B = 2000; `analysis/entropy_best_config_pairs.py`).**

| Pair | All-view mAP | Frontal mAP | Frontal & unseen mAP | Tail AP |
|---|---|---|---|---|
| E5 - E3 | +0.0023 [-0.0041, +0.0073] | +0.0039 [-0.0077, +0.0130] | -0.0042 [-0.0115, +0.0028] | +0.0073 [-0.0196, +0.0285] |
| E5 - E4 | +0.0033 [-0.0006, +0.0076] | +0.0047 [-0.0041, +0.0126] | -0.0029 [-0.0100, +0.0036] | +0.0123 [-0.0036, +0.0307] |
| E3 - E4 | +0.0010 [-0.0029, +0.0064] | +0.0008 [-0.0062, +0.0091] | +0.0013 [-0.0042, +0.0078] | +0.0049 [-0.0117, +0.0274] |
| E5 - E1 | +0.0064 [+0.0018, +0.0120] | +0.0078 [-0.0012, +0.0156] | +0.0019 [-0.0058, +0.0103] | +0.0208 [+0.0018, +0.0444] |
| E3 - E1 | +0.0041 [-0.0003, +0.0108] | +0.0039 [-0.0038, +0.0121] | +0.0061 [-0.0009, +0.0147] | +0.0134 [-0.0051, +0.0418] |
| E4 - E1 | +0.0031 [+0.0009, +0.0069] | +0.0030 [+0.0013, +0.0052] | +0.0048 [+0.0018, +0.0091] | +0.0085 [-0.0007, +0.0245] |
| E5 - A5 | +0.0050 [-0.0006, +0.0099] | +0.0071 [-0.0043, +0.0169] | -0.0066 [-0.0263, +0.0056] | +0.0149 [-0.0083, +0.0355] |
| E3 - A5 | +0.0027 [-0.0024, +0.0089] | +0.0033 [-0.0072, +0.0138] | -0.0024 [-0.0234, +0.0103] | +0.0076 [-0.0137, +0.0342] |
| E4 - A5 | +0.0017 [-0.0019, +0.0050] | +0.0024 [-0.0059, +0.0088] | -0.0037 [-0.0241, +0.0075] | +0.0027 [-0.0124, +0.0162] |

- Mixing resolutions helps a little: the nine-model E4 beats its own five-model subset E1 by +0.0031 all-view [+0.0009, +0.0069] and +0.0030 frontal [+0.0013, +0.0052], the one clean, selection-free gain in the table.
- E5, E3 and E4 cannot be told apart (intervals of E5 - E3 and E5 - E4 include 0 on all four metrics). None of them is distinguishable from the ASL ensemble A5 on frontal mAP, and on frontal & unseen mAP all of them are below A5's point estimate (0.4818).
- The matched-recipe MoE control that exists for these exact member sets (not an ASL control; [2026-10-11-moe-matched-eval.md](2026-10-11-moe-matched-eval.md)): all-view mAP TM 0.4699 / MoE 0.4716 for the four-model set (TM - MoE -0.0016 [-0.0051, +0.0023]) and 0.4722 / 0.4697 for the trio (+0.0026 [-0.0056, +0.0104]). No ASL checkpoints exist at 640, 896 or 1024 px, so the multi-resolution gain cannot be attributed to the Entropy objective or refuted for it.

**Rank stability (2000 paired image resamples).** E5 is first by frontal mAP in 54.9% of resamples; E3 16.6%, U123 10.4%, E4 5.1%, P1024-86 4.7%, P768-42 4.3%; the rule itself picks E5 in 51.1%. E1, the pre-specified same-recipe ensemble, is first in none.

**Selection optimism, patient-level split halves (200 random halves; the rule applied on one half, the pick scored on the other).** The picked row beats the average of the 21 candidates by 0.0167 frontal mAP on the half where it was picked and by 0.0082 on the other half (sd 0.0083): about half of the apparent advantage is lost (optimism 0.0085). Its mean rank on the held-out half is 6.0 of 21 (median 4). The ranking reproduces only partly across halves (Spearman 0.67 +/- 0.12). The pick is E5 in 43.5% of splits (87 of 200), then P1024-86 (27), U123 (21), E3 (19), P768-42 (17), S42 (11), E4 (7). Halves have half the images, so the optimism at full size is smaller than this, but not negligible.

**E5's own selection (post hoc, nested; `analysis/entropy_best_config_nested.py`).** E5's members were chosen on both halves, so the split-half check cannot discount it. Repeating the greedy selection (same procedure, all-view mAP, min gain 1e-4, pool of the 10 v3 dumps) inside each of 200 patient splits on one half and scoring the result on the other:

| Held-out half, nested greedy minus | All-view mAP (sd, share positive) | Frontal mAP (sd, share positive) |
|---|---|---|
| E1 | +0.0031 (0.0032, 85%) | +0.0028 (0.0047, 80%) |
| E3 | -0.0010 (0.0033, 40%) | -0.0009 (0.0056, 43%) |
| E4 | +0.0002 (0.0027, 54%) | -0.0002 (0.0048, 47%) |
| E5 (selected on all patients) | -0.0025 (0.0027, 4%) | -0.0037 (0.0054, 10%) |
| A5 | +0.0014 (0.0032, 70%) | +0.0018 (0.0057, 69%) |

The honest greedy ensemble averages 3.5 members, beats the same-recipe E1 on held-out patients by about +0.003, and does not beat the selection-free E3 or E4. E5 on the same held-out halves is +0.0025 (all-view) and +0.0037 (frontal) above the nested greedy ensemble: that is the size of E5's selection advantage. Of E5's apparent +0.0064 / +0.0078 lead over E1 (all-view / frontal), about 40-50% is selection (0.0025 of 0.0064, 0.0037 of 0.0078).

## 4. Comparison to baselines (internal validation; not leaderboard)
- vs Stage-1 backbone (0.385 internal, 384 px era): E5 0.4722, E1 0.4658: not like-for-like (recipe and resolution), shown for scale only.
- vs MoE Stage-2 (0.405 internal / 0.4599 official test, 3rd place): the published numbers use a different recipe; the matched-recipe MoE at the same member sets is 0.4716 (four models) and 0.4697 (trio) against 0.4699 and 0.4722 for E3 and E5.
- vs the paper's 5-seed entropy triplet (0.414 +/- 0.002 internal, 512 / 384 px era): E1 is +0.052 above, because of recipe and resolution, not of the objective (section 3.3).
- vs ASL, same recipe: E1 - A5 = -0.0014 [-0.0062, +0.0016] all-view (lanczos), -0.0011 [-0.0081, +0.0038] (area). Table 1 single-run means: ASL 0.4617, Entropy 0.4612 (lanczos).
- vs the current campaign: 0.4722 is the same ensemble as the old greedy-3 reference, so no new Entropy-only record exists; the 0.4805 Markov-with-history rows use patient history that the leaderboard split does not provide (0 gain on the leaderboard) and are not comparable.

## 5. Caveats / known-bug flags
- **Winner's curse and shared use of the split.** One split picks the SWA epochs, the early-stopping epochs, E5's members and now the winner. Section 3.4 puts the optimism of the rule at about 0.008 frontal mAP relative to the average candidate, and E5's selection advantage at about 0.003-0.004.
- **Not patient-disjoint, 30% lateral, one split.** 77% of val images share a patient with train and 53% share a study (identical labels). Frontal & unseen-patient (3,572 images) is the closest proxy but is small: two tail classes have no positives there and intervals are +/-0.01.
- **Mixed code paths and recipes.** E3-E5 combine v3 members with different drop-path, schedule and weight type (SWA against best-epoch EMA). The comparison is across members, not a controlled experiment.
- **Lanczos only.** The ranking uses the one filter all rows share; the recommended inference filter (area) is measured only for the v5 and v0 families.
- **Clean internal images only.** The internal validation images carry no added noise or geometric distortion, whereas the Task-1 image set described in [2026-10-17-task1-submission-data.md](../plan/2026-10-17-task1-submission-data.md) has about 25% noisy images (sigma about 30 grey levels) and some warped ones. Whether the three-member E5 is as robust to those perturbations as the four- or nine-member E3 / E4 is not measured by this ranking.
- **No flip-TTA dump** exists for `reg/img1024_dp01_seed{42,1024}` (no-TTA SWA 0.4618 and 0.4567), `push/cont1024_from768_s86` or the other `reg` runs, so they are not ranked; a 1024 px seed-42 model could belong in E3/E4.
- Pre-fix flags: none apply (shard weights and `--metric-embed query` in all 16 logs). Open paper confounds that touch this report: the easy (not hard) mining in all arms, and the softmax-entropy gate (every row uses it); the 384-vs-512 resolution confound does not apply (rows are 512 px and above and compared at matched or stated resolution).

## 6. Notable
- **The Entropy ensemble does not beat the ASL ensemble of the same size (E1 - A5 = -0.0014 all-view, -0.0006 frontal, lanczos), and its mECE is higher by 0.0024.** The single-run null of Table 1 carries over to ensembles.
- **The pre-registered rule re-selects the old greedy-3 ensemble (E5, 0.4722 all-view)**, but about 40-50% of its +0.0078 frontal lead over E1 is selection on this split (nested greedy), and E5 is indistinguishable from the selection-free E3 and E4.
- **On frontal & unseen-patient mAP the ASL ensemble A5 (0.4818) is above every Entropy ensemble** (E3 0.4794, E4 0.4781, E5 0.4752, E1 0.4733), though within noise.
- Two 1024 px checkpoints (seeds 42 and 1024: `reg/img1024_dp01_seed42`, `reg/img1024_dp01_seed1024`; no-TTA SWA 0.4618 and 0.4567) were trained but never dumped with flip TTA.

## 7. Not measured
- Area-filter scores of the v3 checkpoints, hence the recommended inference filter's gain on E5.
- Flip-TTA scores of the `reg` and `cont1024` checkpoints and any ensemble containing them.
- An ASL control at 640, 896 or 1024 px, so the multi-resolution gain cannot be assigned to the Entropy objective.
- Multi-scale TTA (several scales for one checkpoint); every member is single-scale.
- The leaderboard score of any row (the large internal-to-test offset means no internal gap here predicts a test gap).

## 8. Next step
1. Decide between E5 (the rule's winner, three inferences) and E3 / E4 (no member selection) for the leaderboard run; they differ by less than their noise. The candidate predictions on the Task-1 images (job 53230, per-model dumps) are described in [2026-10-17-task1-submission-data.md](../plan/2026-10-17-task1-submission-data.md); any of E3-E5 can be assembled from them without a rerun if its members are among the 11 checkpoints. If the question is the method, not the score: the Entropy objective gives no ensemble-level gain over ASL, so an ASL multi-resolution ensemble would be the matching control.
2. Cheap inference-only extension, if GPU time allows: flip-TTA dumps (area) of `reg/img1024_dp01_seed42` and `seed1024` and of the three E5 members, to put the v3 checkpoints on the same filter as the area-evaluated v5 and v0 runs and to enlarge the 1024 px pool.
3. If the submission is an ensemble of Entropy checkpoints, report it as "a multi-resolution ensemble of Entropy-trained checkpoints", not as evidence for the entropy gate (section 3.3 and the companion edge-case document).

Reproduce: `python -I analysis/entropy_best_config.py` (ranking, rule, E1 vs A5, stability, split-half selection; 2000 resamples, 8 workers, about 19 min), `analysis/entropy_best_config_pairs.py` (section 3.4 pairs), `analysis/entropy_best_config_nested.py` (nested greedy); outputs `analysis/out/entropy_best_config{,_pairs,_nested}.json`, logs next to them. Shared helpers: `analysis/edge_common.py`.
