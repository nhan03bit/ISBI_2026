# Numbers check of the rev4 ISBI 2027 paper, its model-name table, and the rev3 draft edits — 2026-10-02

**Type:** findings note (offline cross-check, not a training run).
**Files checked:**
- `docs/paper/isbi/paper.tex` (rev4, mtime 22:00; diff base `paper.tex.pre-rev4.bak`).
- `docs/paper/isbi/supplement.tex`, new model-name table `tab:names` (diff base `supplement.tex.pre-rev4.bak`; the rev4 diff is only the title plus that table).
- `docs/paper/markov/paper.tex` and `docs/paper/entropy/paper.tex`, each diffed against its `paper.tex.pre-rev3.bak`.

**Related:** [rev2 numbers check](2026-10-02-isbi2027-unified-paper-numbers-check.md), [047 provenance](2026-10-02-047-provenance-and-paper-corrections.md), [Markov A/B training](../result/2026-10-01-markov-ab-training.md), [push wave](../result/2026-09-26-push-wave.md), [HMM post-hoc](2026-10-01-markov-hmm-posthoc.md), [metadata EDA](2026-10-01-padchest-metadata-eda.md).
**Slurm jobs read (all COMPLETED, exit 0:0 per `sacct`, unless noted):** 43047-43051, 43998-44002, 44274-44278, 46472/46473/46476 and 47097-47099 (46471, 46474, 46475 were OUT_OF_MEMORY first attempts, re-run as 47097-47099), 47094, 48997-49000, 49375, 49376, 49405, 49406, 49424, 49442, 49445. Logs in `/home/psytp7/logs/`.

## Question
Does every number and model name in the rev4 paper match the primary artifacts? Does the new `tab:names` map the main-paper names to the right supplement names, scripts and jobs? Do the rev3 edits to the Markov and entropy drafts match their sources? Also: do internal codes leak into the main paper, and is every quoted 0.48 tied to the previous report?

## Data and method
- Artifacts read: Slurm logs (`args:`, `Done. Best`, `SWA weights ->`, `LR schedule`, `triplet hit rate`), `history.json` under `train/checkpoint_Triplet_2/accum_6`, `checkpoint_Triplet_3/{seedsweep,push,prior,res_sweep,sweep2}` and `checkpoint3`, `analysis/out/{ens_select_w01_flip,tta_w1_flip,tta_ab_v5_with,tta_ab_v5_zero}.json`, `analysis/out/markov_ab/{run.log,results.json}`, `analysis/out/markov_hmm/{results.json,oof_seed0.npz,run.log}`, `analysis/out/metadata_eda/{eda.json,run.log}`, `analysis/out/patient_markov/priors.pt`, the published challenge paper `docs/pdf/ISBI2026.pdf` (b12), and `sacct` for job names, submit lines and states.
- Code read: `train/train_2_v3.py` (schedule, SWA, embedding, class weights), `train_2_v2.py` and the Aug-26 snapshot `train_2_v2.py.bak2`, `train_2_v5.py`, `train/prior_conditioning.py`, `train/utils_update.py` (`sample_triplets_v13`), `analysis/patient_prior.py`, `analysis/markov_ab_compare.py`, `evaluate/class_groups.py`.
- Re-runs (all CPU, seconds to 2 minutes each, nothing left running): `python analysis/markov_check.py` end to end; per-class Normal-gating counts and the Normal/abnormal co-occurrence count; a 300-resample patient-clustered bootstrap of the 1024 px minus 512 px difference with three RNG seeds; per-class AP gains of the in-network prior on the primary subset (float32 dumps in `analysis/out/probs_ab_flip`); seen/unseen case-mix statistics; a matched two-seed baseline ensemble. These were ad hoc scripts in the session scratchpad, not saved in the repo.
- Every numeric token in `paper.tex` lines 54-212 was listed and accounted for in the tables below.
- Bibliography spot-check against arXiv, Crossref and Semantic Scholar metadata (public identifiers only).
- Status key: **OK** = matches source. **MISMATCH** = value or statement contradicted by an artifact. **CAVEAT** = value right, but wording, scope or provenance needs care.
- Sources are abbreviated: "run.log [n]" = `analysis/out/markov_ab/run.log` section n; "hmm" = `analysis/out/markov_hmm/results.json`.

## Findings

### Summary
**3 MISMATCHes** among 133 claim rows (108 OK, 22 CAVEAT, 3 MISMATCH). No transcription error between the drafts and the main paper; every Table I cell and every other quoted number reproduces from an artifact. Two of the three are rounding slips, one is a qualitative statement.

| # | Location | Issue |
|---|---|---|
| M1 | `isbi/paper.tex` L202 | "CI −0.016 to +0.020" for post-prediction minus in-network. Exact: +0.002099, CI −0.015497 to +0.019714. The 3-decimal value is **−0.015**; −0.016 is a double rounding of the 4-decimal −0.0155. The supplement and Markov draft give −0.0155, which is correct. |
| M2 | `markov/paper.tex` L58 (abstract, rev3) | "95% CI −0.011 to +0.008" for withheld minus EGT-768. Exact: −0.002632, CI −0.010887 to +0.007475. The 3-decimal upper bound is **+0.007**; +0.008 is a double rounding of +0.0075. The same CI is quoted correctly at 4 decimals elsewhere. |
| M3 | `isbi/paper.tex` L204 (softer variants at `isbi/supplement.tex` L669 and `markov/paper.tex` L579: the prior carries "information the current image does not contain ... such as devices and chronic findings") | "its gain comes mostly from findings that recur, such as **devices** and chronic changes". Per-class AP gains of the in-network prior on the primary subset (n = 4,061) contradict "devices": Support Devices +0.0055 / +0.0040 (seed 42 / 1024; baseline AP 0.937), central venous catheter +0.0032 / +0.0023 (0.906), sternotomy +0.0007 / −0.0060 (0.858). The gain is in chronic and structural findings: calcified densities +0.146 / +0.099, fracture +0.103 / +0.106, bronchiectasis +0.080 / +0.096, hyperinflated lung +0.076 / +0.062, aortic atheromatosis +0.061 / +0.070, vertebral degenerative changes +0.047 / +0.027, pleural thickening +0.045 / +0.045. Post-prediction H4 behaves the same (Support Devices +0.003, central venous catheter −0.0001, sternotomy −0.009). The "chronic changes" half of the claim holds; the "devices" half does not. |

**Top CAVEATs** (full list in the tables; none changes a number):
1. Tail ΔAP in the supplement ("5 of 7 tail classes, +0.0055; +0.0546, 6 of 7, at seed 1024") rests on 1-36 positives per class in the primary subset (Hydropneumothorax 1, pneumoperitoneo 4, Mass 7, azygos lobe 15, Subcutaneous Emphysema 18, Pneumothorax 24, calcified densities 36). The seed-42 tail mean is the sum of +0.146 and +0.125 gains and a −0.327 drop (pneumoperitoneo, 4 positives).
2. Step "+0.025" (row 3 to row 4): the difference of the printed entries (0.439 − 0.414). The exact difference of means is +0.0257 (rounds to 0.026). About 0.006 of it is best-epoch selection on the scoring fold (epoch-1 mean 0.4332 against best-epoch mean 0.4394 over the six 512 px seeds), which row 3 (one epoch) does not have.
3. "Fusing the same chain after the frozen EGT-768 instead gains as much" quotes H4 minus in-network. H4 is the richer arm (gap, sex/age/acuity, cross-class terms), not the pooled chain the network receives. The CI (±0.018) shows non-significance, not equivalence.
4. Supplement "0.4867 with the prior and 0.4693 without it": the 0.4693 is the same networks with the prior withheld. The matched two-seed baseline ensemble (v3 seed 42 + 1024, flip) scores **0.4659**, which the paper does not report.
5. 5,759 earlier-study images: only 80 belong to patients unseen in training (`has_prior & patient_unseen`, CIs span 0). The in-network and post-prediction gains are therefore measured almost entirely on patients that also appear in training.
6. Table I caption "All models start from the same Stage-1 checkpoint": logged for rows 4-11, but not for row 3 (jobs 43047-51) or the under-trained S3 runs (43998-44002). See section A.
7. `tab:names`: `train_2_v2.py` at HEAD is not the code that ran jobs 43047-51, and job 49445 executed `analysis/markov_hmm_check.py` (the model module is `markov_hmm.py`).
8. `entropy/paper.tex` bibitem b25 still reads "...: a negative result", although its own rev3 text and the Markov draft's new title say patient history helps.

### A. Table I (`isbi/paper.tex` L135-162)
Columns are mAP / mF1 / mAUC / mECE (b12 uses the same order). All 11 rows were compared cell by cell; no swap.

| Claim | Location | Paper value | Source value | Status |
|---|---|---|---|---|
| Row 1 Stage 1, 384 px | L144 | 0.385 / 0.395 / 0.875 / 0.134 | b12 Table 1 stage 1: identical. `checkpoint3/Model_20260119_062652/history.json` best epoch 6: 0.3849 / 0.3946 / 0.8746 / 0.1338 | OK |
| Row 2 MoE, 384 px | L145 | 0.405 / 0.402 / 0.882 / 0.130 | b12 Table 1 stage 2: identical. No MoE checkpoint in the repo; b12 is the only source | OK |
| Row 3 initial, 5 seeds, 512 px, mAP | L146 | 0.414 ± 0.002 | Jobs 43047-51 (seeds 86, 123, 456, 789, 1024), `Done. Best mAP`: 0.41395, 0.41289, 0.41082, 0.41558, 0.41553. Mean 0.413755, sample s.d. 0.001993. `history.json` identical | OK |
| Row 3 mF1 / mAUC / mECE | L146 | 0.407 / 0.889 / 0.143 | Means 0.40651 / 0.88893 / 0.14333 | OK |
| Row 3 config: 512 px, lr 3e-5, bank 96, one epoch | L146, L189 | as stated | Log config dict: lr 3e-5, wd 0.01, memory 96, epochs 1, patience 1, bs 8, accum 6; `Size of Image: 512`; `class_weights` printed in CSV order (pre-fix) | OK |
| Row 4 corrected, 6 seeds, 512 px, mAP | L147 | 0.439 ± 0.002 | `seedsweep/*/history.json` best epoch (= `Done. Best` in 46472, 46473, 46476, 47097-99): 0.44337, 0.43891, 0.44140, 0.43752, 0.43695, 0.43852. Mean 0.439446, sample s.d. 0.002462 (population 0.002248) | OK |
| Row 4 mF1 / mAUC / mECE | L147 | 0.454 / 0.900 / 0.128 | 0.45371 / 0.89995 / 0.12818 | OK |
| Row 5 EGT-768, seed 42, SWA | L148 | 0.4600 / 0.4685 / 0.9092 / 0.1354 | `push/img768_dp01_seed42` `swa_metrics` 0.45997 / 0.46851 / 0.90924 / 0.13538; job 49000 `SWA weights ->` line identical | OK |
| Row 6 EGT-1024, seed 86, SWA | L149 | 0.4619 / 0.4679 / 0.9138 / 0.1438 | `push/img1024_dp01_seed86` `swa_metrics` 0.46189 / 0.46793 / 0.91383 / 0.14384; job 48997 line identical | OK |
| Row 7 EGT-768 + flip, single model | L150 | 0.4650 | `ens_select_w01_flip.json` `single` for `push__img768_dp01_seed42__model_swa@768_flip.npy`: 0.464986 | OK |
| Row 8 ensemble + flip | L151 | 0.4722; members EGT-768, EGT-1024, 640 px; 640-1024 px | `ensemble_mAP` 0.472238, `n_val` 17,270; `chosen` = 768 s42 SWA, 1024 s86 SWA, `res_sweep/img640_seed86` best epoch (job 47094) | OK |
| Row 8 footnote, fixed four-model average | L158 | 0.4699 | `tta_w1_flip.json` `val_mAP` 0.469928 (1024 s86, 896 s86, 768 s42, 768 s1024; flip, probability average, no selection) | OK |
| Row 9 EGT-768 + prior, SWA | L154 | 0.4807 / 0.4863 / 0.9151 / 0.1342 | `prior/img768_qprior_seed42` `swa_metrics` 0.48073 / 0.48633 / 0.91508 / 0.13417; job 49375 SWA line identical | OK |
| Row 10 + prior, flip | L155 | 0.4839 | run.log [1] B-early s42 "all": 0.483868 | OK |
| Row 11 history withheld, flip | L156 | 0.4624 | run.log [1] B-early0 s42 "all": 0.462354 | OK |
| Input-px and flip columns | L144-156 | 384, 384, 512, 512, 768, 1024, 768, 640-1024, 768, 768, 768; flip on rows 7, 8, 10, 11 | log `--img-size` / `Size of Image`; dumps are `*_flip.npy` for rows 7, 8, 10, 11 and no TTA for the SWA rows | OK |
| Caption: 17,270 images; prior exists for 33% | L136 | 17,270; 33% | `labels.npy` 17,270; has_prior 5,759 / 17,270 = 33.35% | OK |
| Caption: "All models start from the same Stage-1 checkpoint" | L136 | same | `history.json` / `args:` record `checkpoint3/Model_20260119_062652/model_best.pth` for rows 4-11. Rows 3 (43047-51) and the S3 under-trained runs (43998-44002) log no checkpoint path; their code default changed over time (`scripts/submit_batch48_sweep.sh`: "now the default in train_2_v2.py after the stage1_full30 regression was found"). Supporting evidence: the explicit-checkpoint re-runs 44274-78 give best mAP 0.4121, 0.4100, 0.4123, 0.4158, 0.4113 (mean 0.4123 against 0.4138 for 43047-51), while the under-trained runs give 0.393; the Aug-26 snapshot hard-codes the converged file | CAVEAT |
| Caption/footnote: selection on the scoring fold | L136, L158 | only the ensemble is flagged | Rows 4-6 are best-epoch or SWA picks on this fold (SWA epochs are the three best by validation mAP; here 2-4 in every run). Row 4 per-seed epoch-1 mAP (logs) 0.4305, 0.4339, 0.4343, 0.4320, 0.4314, 0.4369 (mean 0.4332) against best-epoch 0.4394. Row 3 has no selection | CAVEAT |
| Seed not labelled | L148-156 | rows 5, 7, 9, 10, 11 | All are seed 42 (the seed-1024 pair is only in the text) | CAVEAT (presentation) |

### B. Prose numbers in `isbi/paper.tex`

| Claim | Location | Paper value | Source value | Status |
|---|---|---|---|---|
| Test mAP, third place | L55, L65, L198 | 0.4599, third | b12 abstract and Table 3: 0.4599, "securing third place" | OK |
| Abstract and intro: 0.385 to 0.414, MoE 0.405 | L55 | as stated | 0.3849 to 0.413755; b12 0.405. The 384 px (Stage 1, MoE) against 512 px (EGT) confound is disclosed in Setup and Results but not in the abstract | CAVEAT |
| Corrected, 512 px | L55 | 0.439 | 0.439446 | OK |
| Single model 768 / 1024 px | L55 | 0.460-0.462 | 0.45997 / 0.46189 | OK |
| Ensemble, flip, "members selected on the same fold; not yet scored on the leaderboard" | L55 | 0.472 | 0.472238; no EGT model has a leaderboard score (not measurable here) | OK |
| 0.460 to 0.481 (abstract, intro, conclusion) | L55, L67, L211 | 0.460 to 0.481 | 0.45997 to 0.48073 (job 49000 to 49375) | OK |
| "about +0.02 mAP" | L73 | +0.02 | +0.0207 (s42 SWA), +0.0204 (s1024 SWA), +0.0189 / +0.0194 (flip) | OK |
| "when the previous report is available" next to a full-fold number | L55, L67, L211 | 0.481 | 0.4807 is the all-image mAP in which only 33% of images have an earlier study. On the images that have one the gain is 0.4894 to 0.5276 (+0.038). Wording could be read as "all images have a report" | CAVEAT |
| Seed 1024 | L202 | 0.4582 to 0.4786 | 0.458206 (48999) to 0.478616 (49376) | OK |
| Flip, rows 7 and 10 | L202 | 0.4650 to 0.4839 | 0.464986 to 0.483868 | OK |
| Earlier-study images | L202 | 0.4894 to 0.5276 (5,759 images) | run.log [1] has_prior s42: v3 0.489401, B-early 0.527641 | OK |
| Paired, frontal with earlier study, seed 42 | L202 | +0.0270 (+0.0081, +0.0497) | `d_point` 0.027045, CI 0.008139 to 0.049684. The quoted value is the observed difference; the bootstrap mean is +0.0295 | OK |
| Same, seed 1024 | L202 | +0.0380 (+0.0265, +0.0547) | `d_point` 0.038001, CI 0.026501 to 0.054671 (bootstrap mean +0.0399) | OK |
| Convention differs between rows | L202, Supp. S6 | "paired difference" | The two lines above are observed differences with percentile CIs; +0.0375 (H4) and +0.002 (H4 minus in-network) are bootstrap means (`hmm`); +0.020 / +0.007 are labelled "bootstrap means" | CAVEAT |
| Post-prediction minus in-network, seed 42, primary | L202 | +0.002 (−0.016, +0.020) | `hmm` `H4 - B-early` primary: mean +0.002099, CI −0.015497 to +0.019714 | **MISMATCH (M1)** |
| "Fusing the same chain after the frozen EGT-768 gains as much" | L202 | gains as much | The paired number is for H4 (chain + gap + sex/age/acuity + cross-class edges), not the pooled chain. Pooled-chain H0 gives +0.0354, H1 +0.0477 (primary, seed 42) against in-network +0.0339 (same scoring); H1 minus in-network +0.0124 (−0.0076, +0.0348), exploratory. CI half-width about 0.018, so equivalence is not shown | CAVEAT |
| Withheld against row 7 | L202 | 0.4624 against 0.4650; −0.0026 (−0.0109, +0.0075) | 0.462354 against 0.464986; `d_point` −0.002632, CI −0.010887 to +0.007475. Seed 1024: +0.0037 (−0.0029, +0.0106) | OK |
| "leaderboard effect is 0" | L202 | 0 | By construction (b14: patient-level de-duplication, identifiers not released; verified in the arXiv text). Not a measurement | OK |
| Smoothing range | L204 | 0.0006 to 0.0053 | `markov_check.py` re-run: α 0.02 −0.0006, 0.05 −0.0012, 0.10 −0.0024, 0.20 −0.0053 (base 0.4722) | OK |
| Normal-gating range | L204 | 0.010 to 0.046 | α 0.25 −0.0103, 0.50 −0.0239, 1.00 −0.0459. The range spans α 0.25-1.0; at the winner's 0.5 it is 0.024 | OK |
| "hurting 28 of 30 classes at the winner's setting" | L204 | 28 of 30 | Recount at α 0.5: 28 worse, 1 better (Hydropneumothorax +0.0092), 1 unchanged (Normal); at α 0.25: 26 worse. b15 confirms α_ng = 0.5 | OK |
| Normal with an abnormal finding | L204 | 26 validation images | 26 of 17,270 (6,917 Normal positives) | OK |
| Image-only history | L204 | +0.009 | `hmm` H4-F primary s42: +0.0092 (−0.0013, +0.0181) | OK |
| "gain comes mostly from findings that recur, such as devices and chronic changes" | L204 | devices and chronic changes | See M3: devices ≈ 0, chronic findings carry the gain | **MISMATCH (M3)** |
| "explicit cross-label transition terms add nothing" | L204 | nothing | H3 to H4 primary: +0.0378 to +0.0375 (s42), +0.0435 to +0.0424 (s1024); H5 selected w_J = 0 in every fold | OK |
| Mechanism for label priors | L55, L204 | "do not help, because the ML-Decoder already models co-occurrence" | No ablation removes the decoder's query self-attention; the supplement words it as "we read this as". The abstract states it as a cause | CAVEAT |
| Metadata alone | L204 | held-out macro AUROC 0.76 | 0.76001 (patient-grouped 5-fold CV, training fold, all four groups) | OK |
| Stacking metadata | L204 | lowers mAP by 0.003 to 0.016 | 16 cells: −0.00258 (s42, all, patient) to −0.01580 (ensemble, frontal, all three groups); 15 of 16 CIs exclude 0 | OK |
| Data: training and validation folds | L187 | 103,303 / 17,270 | `class_groups.py` N = 103,303; `labels.npy` 17,270. `train_fold_1.csv` has 103,305 rows, `val_fold_1.csv` 17,274 (4 absent from the shards). The main paper does not say the folds hold about 120.6k of the 142,928 released training images (b14 Table 1) | CAVEAT |
| Earlier-study counts | L173, L191 | 5,759 of 17,270; 4,061 frontal | run.log [0]: has_prior 5,759, primary 4,061, no_prior 11,511 | OK |
| Chain pairs | L173 | 24,180 | `priors.pt` `chain_pairs` 24,180; `hmm` run.log "train pairs 24180" | OK |
| Step +0.029 over Stage 1 | L198 | 0.029 | 0.413755 − 0.3849 = 0.028855 | OK |
| Step +0.009 over MoE | L198 | 0.009 | 0.413755 − 0.405 = 0.008755 | OK |
| Step +0.025 (row 3 to 4) | L198 | 0.025 | 0.439446 − 0.413755 = 0.025691 (3 d.p.: 0.026). 0.025 is the difference of printed entries. About 0.006 is best-epoch selection (see A) | CAVEAT |
| Step +0.021 (row 4 to 5) | L198 | 0.021 | 0.459970 − 0.439446 = 0.020524 | OK |
| Step +0.002 (row 5 to 6, different seed) | L198 | 0.002 | 0.461892 − 0.459970 = 0.001921 | OK |
| Step +0.005 (flip, row 7) | L198 | 0.005 | 0.464986 − 0.459970 = 0.005016 | OK |
| Step +0.007 (ensemble) | L198 | 0.007 | 0.472238 − 0.464986 = 0.007252 | OK |
| Steps are not a chain | L198 | "adds ... 0.002 ... Flip TTA adds 0.005" | Row 6 (1024 px) is a side branch: flip TTA is measured on row 5 (768 px), not row 6. Summing all steps overshoots 0.4722; the text says "approximate" | CAVEAT |
| mECE regression | L198 | 0.134 to 0.143 | 0.1338 to 0.1433 | OK |
| "From a less converged Stage-1 checkpoint the initial gain halves" | L198 | halves (Supp. S3) | 43998-44002 mean 0.39306 against `stage1_full30` 0.3771 = +0.016, against +0.029. Checkpoint identity of 43998-44002 not logged (see A) | CAVEAT |
| 96-slot bank: 13-14% of first-epoch steps | L198 | 13-14% | Re-runs 44274-78 epoch 1: 13.2, 13.1, 14.0, 13.5, 13.5% (1708/12914 etc.). Jobs 43047-51 logged no rate; the re-runs use `--epochs 3 --patience 2` and the converged checkpoint; their best mAP reproduces row 3 within noise | OK |
| Corrected: 81-97% | L198 | 81-97% | Epoch 1, final recipe: 512 px bs 8: 96.4-96.6% (46472, 46473, 46476, 47097-99); 640-1024 px bs 4: 81.2% (47096), 81.4% (47094), 81.6%, 81.7%, 82.1%, 82.3% (47095, 48998, 48997, 49000). Union 81.2-96.6%. Early sweeps without the class-order fix logged 100% (45821-45824, `bug_repro` 46458) and are not "corrected" | OK |
| Fixed average | L198 | 0.4699 | 0.469928 | OK |
| 77% share a patient; 30% lateral | L208 | 77%, 30% | `markov_check.py`: 0.770 (10,978 shared patients); 5,234 / 17,270 = 30.3% | OK |
| Ensemble on frontal, unseen-patient subset | L208 | 0.4752 | 0.4752 (n = 3,572; 28 classes with a positive; selected-on-fold ensemble) | OK |
| Resolution gain seen to unseen | L208 | +0.020 to +0.007 (bootstrap means), not significant | Bootstrap re-run, 3 RNG seeds (300 resamples each): seen means +0.0192 to +0.0199, CIs from [+0.0047, +0.0358] to [+0.0061, +0.0334] (point +0.0180); unseen means +0.0069 to +0.0078, CIs [−0.0201, +0.0282], [−0.0153, +0.0282], [−0.0159, +0.0260], all including 0 (point +0.0070) | OK |
| "shrinks" from +0.020 to +0.007 | L208 | shrinks | Not tested: the unseen CI comfortably contains the seen estimate, so the data do not show the gain differs. Subset scores are over 28 classes, not 30 | CAVEAT |
| Models compared | L208 | differ in seed, stochastic depth, weight averaging | 1024 px s86 SWA with drop-path 0.1 against 512 px s1024 best epoch without | OK |
| Bank 96 / 2,048; λ 0.1; margin 0.5 | L98, L133 | as stated | log config `memory_size 96`; v3 `--memory-size 2048`; `--triplet-lambda 0.1`; margin 0.5 | OK |
| ASL γ, clip, weights | L133 | γneg 2, γpos 0, clip 0.05, "inverse log frequency" clipped to [0.5, 2] | cfg gamma_neg 2.0 / gamma_pos 0.0; `clip=0.05`; code `log(1/freq)`, divided by mean, `np.clip(.., 0.5, 2.0)`; shard-order log: Hydropneumothorax 2.000, Normal 0.500. Wording "inverse log frequency" means log of the inverse frequency | OK |
| Softmax entropy, anchors at or above batch mean, easy mining, ≥ 1 shared non-head label, negatives share none | L100-125 | as stated | `sample_triplets_v13`: `calculate_entropy_from_logits(logits)` default softmax; `entropy[i] < entropy.mean()` skipped; argmin positive / argmax negative; `--head-classes 5` gives `head_class_ids=[6, 24, 17, 9, 12]` in the logs of 46472, 49000, 49375 = Normal, pleural effusion, cardiomegaly, Support Devices, aortic elongation (the CSV-order runs 45823 / 46458 log `[0, 3, 2, 8, 1]`, the misaligned ids) | OK |
| ℓ2-normalised mean-pooled query embedding | L95 | ℓ2-normalised | `train_2_v3.py:840` `F.normalize(embeddings, dim=1)` after `h.mean(dim=1)`; also normalises anchor/pos/neg at :852-854 | OK |
| Corrected recipe | L189 | lr 1e-4, bank 2,048, EMA 0.999, 8-epoch cosine per update, 5% warm-up, early stopping, 512 px, six seeds | args `--lr 1e-4 --memory-size 2048 --epochs 8 --patience 3`; `--ema-decay` default 0.999; log "warmup 860 / total 17217" (512 px) = 5.0%; `warmup_frac` default 0.05 | OK |
| EGT-768 / 1024 recipe | L189 | batch 4, accum 12, four of eight epochs, drop-path 0.1 | args `--batch-size 4 --accum-steps 12 --epochs 4 --sched-epochs 8 --drop-path-rate 0.1 --swa-last-k 3`; log "warmup 860 / total 17216" | OK |
| "average of the EMA weights of the last three epochs" | L189 | last three | `_save_swa` averages the **K best-monitored epochs**. In all five SWA runs used (49000, 48999, 48997, 49375, 49376) `swa_epochs` is [2, 3, 4], which are also the last three, so the number is unaffected | CAVEAT |
| Initial recipe: AdamW, lr 3e-5 constant, wd 0.01, bs 8, accum 6, 512 px, one epoch | L189 | as stated | bak2:546 AdamW; log config; `history.json` `lr = [0.0]` in all five runs; bak2:547 `CosineAnnealingLR(T_max=cfg["epochs"])` and :711 `self.sched.step()` once per epoch, so lr was 3e-5 throughout the single epoch; the 43047 log has no `LR schedule: warmup` line. The Aug-27 source is not preserved, so this rests on the `lr` values plus the Aug-26 snapshot | OK |
| Five implementation differences; two fixed, three re-described | L71, L193 | as stated | every v3 `args:` has `--metric-embed query --class-weight-order shard`; 43047-51 predate both (CSV-order weights in the log) | OK |
| Prior: Q_c offset, u and W zero-initialised, dropout 0.3 | L178-183 | as stated | `prior_conditioning.py`: `offset = r_c * u_c + W r`, `u` (30×768) zeros, `W` Linear(31→768, no bias) zeros; `--prior-dropout 0.3` per image | OK |
| Training history from the training fold only; validation history from both folds | L191 | as stated | `patient_prior.py` L13-14, L163-164 | OK |
| 384 px (b12) against 512 px (ours) | L193 | as stated | b12: "384x384 during training and evaluation"; `Size of Image: 512` | OK |
| Ensemble members and method | L189 | greedy forward selection, probability average, selected on the scored fold | `ens_select.py`; `tta_*.json` `logit_avg: false`; the fixed four-model average has no greedy step but shares the early-stopping fold | OK |
| No leaderboard score; no gate ablation | L198, L211 | stated as limits | None exists (not measurable here) | OK |

### C. Internal codes and 0.48 mentions in the main paper
| Check | Result | Status |
|---|---|---|
| `v3`, `v4`, `v5`, `B-early`, `B-late`, `H0`-`H5`, `mk1`, `A-strong`, `.py`, `train_2`, job or Slurm identifiers, `b25`, "companion" | `grep -E` over `isbi/paper.tex`: the only hits are the e-mail address (`ititiu20216`) and five-digit page numbers in the bibliography. No internal code or job number in the text, tables or captions | OK |
| Every quoted 0.48 is tied to the previous report or history | Six lines contain "0.48": L55 and L67 ("when that/the previous report is available"), L154 and L155 (Block B, header "previous report available"), L202 ("previous report available"), L211 ("when the previous report is available"). Row 11 (0.4624, withheld) is not a 0.48 | OK |

### D. Supplement `tab:names` (`isbi/supplement.tex` L67-87)
| Claim | Location | Paper value | Source value | Status |
|---|---|---|---|---|
| EGT initial = `train_2_v2.py`, 43047-51 | L76 | `train_2_v2.py`; 43047-51 | Jobs 43047-51: names `isbi2026_train2v2_accum6_seed*`, COMPLETED, Aug 27. The file at HEAD (mtime Sep 30, first committed Sep 19) differs from the code that ran: it has a per-step `LambdaLR` schedule that logs `LR schedule: warmup ...` (absent from the 43047 and 44274 logs) with `--warmup-frac` default 0.03, plus `--class-weight-order` and a stage-1 argument. Re-running it with the original arguments would therefore not reproduce the constant LR of row 3. Nearest preserved version of the Aug-27 code is `train_2_v2.py.bak2` (Aug 26) | CAVEAT |
| EGT corrected = v3 seed sweep, `train_2_v3.py` | L77 | `train_2_v3.py`; "seed sweep" (no job ids) | Jobs 46472, 46473, 46476 and 47097-47099 (seeds 86, 123, 1024, 42, 456, 789) ran `train_2_v3.py` (`args:` `--seed N ... --epochs 8 --patience 3`). First attempts 46471, 46474, 46475 were OOM-killed. No job ids given in the table | OK |
| EGT-768 = 49000 (seed 42), 48999 (seed 1024) | L78 | 49000 s42, 48999 s1024 | 49000 `img768_dp01_s42` (`--seed 42`), 48999 `img768_dp01_s1024` (`--seed 1024`), both `--img-size 768`, drop-path 0.1, SWA. Table S2 has two 768 px SWA rows (seed 86 without drop-path, 0.4566; seeds 42 / 1024 with drop-path), so "v3, 768 px, SWA" alone is ambiguous; the job ids disambiguate | OK |
| EGT-1024 = 48997 | L79 | 48997 | `img1024_dp01_s86`, `--seed 86 --img-size 1024`, SWA 0.4619 | OK |
| EGT ensemble = `analysis/ens_select.py` | L80 | script only | `ens_select.py` produces `ens_select_w01_flip.json`. The members' jobs are not listed: 49000, 48997 and the 640 px model 47094 (best epoch, no drop-path, not SWA); the four-model average also needs 48998 (896 px) and 48999 | CAVEAT (completeness) |
| EGT-768 + Markov prior = `train_2_v5.py`, 49375, 49376 | L81 | as stated | `train_2_v5.py`; 49375 `--seed 42`, 49376 `--seed 1024`, both with `--prior-file ... --prior-dropout 0.3 --prior-lr-mult 10`; COMPLETED | OK |
| Post-prediction = `analysis/markov_hmm.py`, 49445 | L82 | `markov_hmm.py`; 49445 | 49445 = `markov_hmm_full`, COMPLETED, output `analysis/out/markov_hmm/`. Its submit line runs `analysis/markov_hmm_check.py`; `markov_hmm.py` is the model module that script imports. The Reproducibility section (L708) names both | CAVEAT |
| Supplement names used | L76-82 | "original code / implementation", "v3", "greedy three-model ensemble", "in-network patient-history prior", "patient HMM, arms H0-H5" | Each name occurs in the supplement body (S1, S2, S4, S5, S6). The HMM table also holds E, H4-acq and H4-F, which "H0-H5" does not list | OK |
| Supplement section pointers from the main paper | paper L183, L193, L198, L202, L204, L208 | S4, S1, S2/S3, S6 | `supplement.aux`: S1 corrections, S2 v3, S3 maturity and utilisation, S4 full method (S4-F HMM), S5 setup, S6 results (S6-D HMM, S6-E metadata, S6-F subsets). Both logs show no undefined references | OK |

### E. Rev3 edits to the two source drafts
Diffs (GNU `diff`): `entropy/paper.tex` against `.pre-rev3.bak` has 7 hunks; `markov/paper.tex` has 37 (title, abstract, contributions, Fig. 1 and the trainable Markov-layer section replaced by the patient-history prior, in-network table, discussion, conclusion, reproducibility). Every hunk that contains a number, a job id or a method statement is in the table below; the remaining hunks are renames (B-early to "in-network", B-early0 to "withheld") and TikZ geometry.

| Claim | Location | Paper value | Source value | Status |
|---|---|---|---|---|
| ℓ2-normalised embedding | entropy L103 | "mean-pool ... and ℓ2-normalize" | `train_2_v3.py:840`; also v2 `train_2_v2.py:672`, bak2:614 | OK |
| Constant LR in the initial recipe | entropy L166 | "constant learning rate (a cosine schedule was configured but is stepped once per epoch)" | `history.json` `lr = [0.0]` (all five), bak2:547 and :711; no `LR schedule` line in 43047; `train_2_v3.py:584-586` comment documents the old per-epoch behaviour | OK |
| Hit rates, re-runs | entropy L239 | 13.1-14.0% epoch 1; 11.9-12.8% later | 44274-78: epoch 1 13.2, 13.1, 14.0, 13.5, 13.5; epoch 2 12.6, 12.6, 12.8, 12.5, 12.5; epoch 3 12.2, 11.9, 12.4, 12.0, 11.9 (range 11.9-12.8; the old 12.6 upper bound was wrong) | OK |
| "same settings and a three-epoch schedule" | entropy L239 | same settings | Also `--patience 2` and an explicit `--stage1-ckpt`; epoch-1 dynamics are the same (lr 3e-5 at epoch 1) | OK |
| 5% linear warm-up, per-update cosine | entropy L243 | 5% | `warmup_frac` default 0.05; logs: warm-up **860 of 17,216** optimizer steps at 768/1024 px (49000, 48999, 48997, 49375, 49376; `steps_per_epoch` 25,825, accum 12) and 860 of 17,217 at 512 px (accum 6); 860/17,216 = 4.995%. The drafts state "5%" and never the counts | OK |
| v3 six-seed s.d. | entropy L256 (Table III) | ±0.002 / 0.002 / 0.004 / 0.006 | mAP 0.002462, mAUC 0.001992, mF1 0.004197, mECE 0.006231 (sample) | OK |
| Stochastic depth + SWA | entropy L272, markov L575 | +0.002 to +0.005, "768 px comparison also differs in seed" | 768 px no-drop-path SWA s86 0.4566 against drop-path SWA s42 0.4600 (+0.0034) and s1024 0.4582 (+0.0016); same-seed 512 px A/B `sweep2` dp01 0.4399 against fix_only 0.4349 (+0.0050, best epoch, no SWA). The two ends use different comparisons | OK |
| Stale citation title | entropy L296 (bibitem b25) | "Markov-chain priors ... : a negative result" | The Markov draft's title is now "When Do Markov Priors Help ...? Patient History Helps, Label Graphs Do Not"; entropy L272 itself says a patient prior helps with a report | CAVEAT |
| b11 author list | entropy L293, isbi L221 | "Xuan Zhong Feng" | arXiv 2602.22092v2 lists "Xuan Zhong Feng". b12's printed reference and arXiv 2604.15555 print "Fengnian Zhao" | OK |
| Markov abstract: label-graph numbers | markov L58 | −0.0006 to −0.0053; 0.010 to 0.046 | as in B | OK |
| Markov abstract: single model 0.460 to 0.481 (0.458 to 0.479 at a second seed) | markov L58 | as stated | 0.45997 to 0.48073; 0.45821 to 0.47862 | OK |
| Markov abstract: +0.0375 on frontal earlier-study images | markov L58 | +0.0375 | `hmm` H4 s42 primary bootstrap mean 0.037521 (CI 0.0228-0.0513) | OK |
| Markov abstract: withheld −0.003, CI −0.011 to +0.008 | markov L58 | CI upper +0.008 | −0.002632, CI −0.010887 to +0.007475 | **MISMATCH (M2)** |
| Markov abstract: +0.009; stacking 0.003-0.016; +0.020 to +0.007 | markov L58 | as stated | +0.0092; −0.00258 to −0.01580; bootstrap means +0.0196 / +0.0074 | OK |
| In-network setup: 5,759 of 17,270; prior dropout 0.3; lr ×10 (1e-3 against 1e-4) for u and W; four of eight epochs; batch 48; λ 0.1; bank 2,048; drop-path 0.1 | markov L302 | as stated | 49375/49376 `args:`; `train_2_v5.py` L736-739 `prior_lr_mult` | OK |
| Normal-gating class counts | markov L340 | 28 of 30 at α ≥ 0.5; 26 at 0.25 | 28 / 26 (recount) | OK |
| In-network mAP rise | markov L364 | +0.019 at both seeds (0.4650 to 0.4839; 0.4626 to 0.4820); SWA 0.4600 to 0.4807, 0.4582 to 0.4786 | +0.0189 / +0.0194; 0.464986 to 0.483868, 0.462579 to 0.482019; SWA 0.45997 to 0.48073, 0.45821 to 0.47862 | OK |
| In-network table, 24 cells (seeds 42, 1024 × v3 / + prior / withheld × All / Earlier / Frontal earlier / None) | markov L368-388 (same table in supplement L456-476) | e.g. 0.4650, 0.4894, 0.4909, 0.4529; 0.4839, 0.5276, 0.5179, 0.4562; 0.4624, 0.4811, 0.4783, 0.4562; seed 1024: 0.4626, 0.4807, 0.4791, 0.4534; 0.4820, 0.5265, 0.5171, 0.4548; 0.4663, 0.4830, 0.4838, 0.4548 | `markov_ab/results.json` `subsets`: all 24 cells equal to 4 d.p. (checked programmatically); the two copies of the table are identical text | OK |
| Subset sizes in table caption | markov L368 | 5,759 / 4,061 / 11,511 | run.log [0] | OK |
| No-earlier-state mAP | markov L364 | 0.4562 against 0.4529 (seed 42) | 0.456167 against 0.452870; bootstrap difference +0.0035 [−0.0060, +0.0286] (CI includes 0; not stated) | OK |
| Head / medium / tail ΔAP, primary, seed 42 | markov L364 | +0.0175 (3/3), +0.0360 (20/20), +0.0055 (5/7) | run.log [4] and my float32 recount: +0.0175 (3/3), +0.0360 (20/20), +0.0055 (5/7). Groups equal `evaluate/class_groups.py` (3 head, 20 medium, 7 tail); `markov_ab_compare.py` imports `frequency_groups` from it, which resolves the open question in the Markov A/B result note | OK |
| Tail seed 1024 | markov L364 | +0.0546 (6/7) | +0.0546 (6/7); head +0.0122 (3/3), medium +0.0361 (18/20) | OK |
| Tail evidence base | markov L364, supplement L454 | tail means quoted without positive counts | Positives per tail class in the 4,061-image subset: Hydropneumothorax 1, pneumoperitoneo 4, Mass 7, azygos lobe 15, Subcutaneous Emphysema 18, Pneumothorax 24, calcified densities 36. Seed-42 tail mean includes pneumoperitoneo −0.327. The 10× seed difference (+0.0055 against +0.0546) is noise-sized | CAVEAT |
| Withheld against baseline | markov L364 | 0.4624 against 0.4650 (−0.0026, −0.0109 to +0.0075); 0.4663 against 0.4626 (+0.0037, −0.0029 to +0.0106) | `d_point` −0.002632 [−0.010887, +0.007475]; +0.003689 [−0.002907, +0.010556] | OK |
| Two-seed ensemble | markov L364 | 0.4867 with the prior, 0.4693 without | `tta_ab_v5_with.json` 0.486723, `tta_ab_v5_zero.json` 0.469262 (flip). "Without" = same networks, prior zeroed. Matched baseline ensemble (v3 s42 + s1024, flip) = 0.4659 (+0.0208 for the prior with; +0.0034 for withheld). Not reported | CAVEAT |
| H4 against in-network, other seeds / scopes | markov L398 | +0.0021 (−0.0155, +0.0197) s42; −0.0022 (−0.0204, +0.0158) s1024; full fold −0.0037, −0.0022; withheld against E −0.0027, +0.0004 | `hmm`: +0.002099 [−0.015497, +0.019714]; −0.002207 [−0.020361, +0.015814]; −0.003734, −0.002230; −0.002718, +0.000439 | OK |
| H1 against in-network (exploratory) | markov L400 | +0.0124 (−0.0076, +0.0348); +0.0081 (−0.0126, +0.0285) | +0.012386 [−0.007649, +0.034793]; +0.008055 [−0.012636, +0.028493] | OK |
| H4-F against in-network | markov L402 | −0.0265 (−0.0467 to −0.0063) | −0.026509 [−0.046704, −0.006345] | OK |
| HMM table rows "In-network" and "Withheld" (renamed in rev3, numbers unchanged) | markov L428-429 | e.g. in-network s42 +0.0191, +0.0339, +0.0014; withheld s42 −0.0027, −0.0071, +0.0014 | `hmm` B-early / B-early0 entries equal at 4 d.p. (all 36 values) | OK |
| "perform the same" | markov L73-75, L579 | "the two fusion points perform the same" / "equivalent within noise" | Same evidence as the +0.002 row: non-significance with a ±0.018 CI | CAVEAT |
| Discussion and conclusion numbers | markov L575-579, L612 | 0.460 to 0.481; H4 about +0.04; image-only +0.009 | as above | OK |
| Reproducibility job ids | markov L618 | 49375, 49376, 49000, 48999, 49405, 49406, 49442 | `sacct`: 49405 `ev_ab_v5_with`, 49406 `ev_ab_v5_zero`, 49442 `markov_ab_compare` (submit line runs `analysis/markov_ab_compare.py`) | OK |

### F. Other spot-checks
| Claim | Source | Status |
|---|---|---|
| b15 gating formula and α_ng = 0.5 | arXiv 2602.13430: `p_c <- p_c * (1 - p_0)^alpha_ng`, "alpha_ng = 0.5 in our submission" | OK |
| b14 de-duplication and re-identification quotes; 3,020 frontal images, 1,620 + 1,400 | arXiv 2604.15555 text found | OK |
| CheXGCN name; b17 journal details (JBHI 24(8) 2292-2302, 2020) | Semantic Scholar abstract ("which we term the CheXGCN"); Crossref | OK |
| BioViL-T, ML-Decoder, triplet-loss (Hermans) identifiers | arXiv 2301.04558, 2111.12933, 1703.07737 | OK |
| Seen against unseen case mix (supplement S6-F) | Normal prevalence 0.576 against 0.348, labels per image 1.27 against 1.57, frontal 90.1% against 63.1%; 28 classes with a positive in frontal-unseen | OK |
| Earlier-study images from unseen patients | run.log [0]: has_prior & patient_unseen n = 80 (1.4% of 5,759); in-network ΔmAP on it +0.0136 / +0.0641 with CIs spanning 0 | CAVEAT (see Summary 5) |
| Compiled PDFs | `paper.pdf` (5 pages) and `supplement.pdf` (13 pages) contain the Table I numbers; no undefined references or overfull boxes in either log | OK |

## Recommendation
Edit `isbi/paper.tex`, then the matching lines in the drafts:
1. L202: "CI −0.016 to +0.020" to "−0.015 to +0.020" (or quote 4 decimals, −0.0155).
2. `markov/paper.tex` L58 abstract: "+0.008" to "+0.007".
3. L204: replace "such as devices and chronic changes" with the per-class evidence, for example "mostly chronic and structural findings (calcified densities, fracture, bronchiectasis, aortic atheromatosis, hyperinflated lung); devices are already near ceiling (AP 0.91-0.94) and gain nothing". Align `isbi/supplement.tex` L669 and `markov/paper.tex` L579 ("such as devices and chronic findings") the same way.
4. L198: "adds 0.025" to "adds 0.026", or keep it and add "(about 0.006 of it is best-epoch selection)".
5. Table I: add "seed 42" to rows 5, 7, 9-11 (or the caption), and say in the caption that rows 4-6 pick the best epoch / SWA on this fold.
6. L202: say the +0.002 is for the post-prediction arm that also uses gap, sex/age/acuity and cross-class terms, and that the interval does not show equivalence; or quote H0 / H1 against in-network.
7. Supplement S6-C: write "0.4693 with the prior withheld; the matched two-seed baseline ensemble scores 0.4659".
8. Supplement S6-C and the Markov draft: give the positive count per tail class, or drop the tail ΔAP sentence.
9. Abstract: "does not help; this is consistent with the ML-Decoder already modelling co-occurrence" in place of "because"; state the 384 against 512 px difference when quoting 0.385 to 0.414.
10. `tab:names`: add the seed-sweep jobs (46472, 46473, 46476, 47097-47099), the ensemble members (49000, 48997, 47094) and the four-model average (48997-49000); change "markov_hmm.py; 49445" to "markov_hmm_check.py (model: markov_hmm.py); 49445"; footnote that `train_2_v2.py` was edited after jobs 43047-51 (Aug-26 snapshot `train_2_v2.py.bak2`).
11. `entropy/paper.tex` L296: update the b25 title to the Markov draft's current title.
12. `isbi/paper.tex` L189, `isbi/supplement.tex` L104 and `entropy/paper.tex` L243: replace "last three epochs" by "the three best epochs by validation mAP (epochs 2-4)".

## Caveats
- The MoE row (row 2) and the Stage-1 numbers are traceable only to b12 (Stage 1 also to the repo history file).
- The Aug-27 source of `train_2_v2.py` is not preserved; "constant LR" and "converged checkpoint" for jobs 43047-51 rest on `history.json` `lr`, the Aug-26 snapshot and mAP levels, not on a logged path.
- The bootstrap intervals were re-run for the 1024-versus-512 px result only (different RNG streams give upper bounds that move by up to 0.0013; the documented lower bounds reproduce). The other intervals were read from the stored result files, not recomputed.
- Per-class gains in M3 and the tail counts come from float32 dumps scored on the primary subset; the float16 out-of-fold file gives slightly different per-class values for the same arms (for example central venous catheter +0.0057 against +0.0032) with the same pattern.
- External literature claims beyond those in section F (for example the supplement's table of the other winners) were not checked.

## Not measured
- Leaderboard score of any EGT model or ensemble.
- A gate-off or triplet-off EGT control.
- Per-class head / medium / tail results for the EGT stage itself (no `eval_result*.json` exists for those checkpoints).
- mF1, mAUC and mECE for rows 7, 8, 10 and 11 (shown as "--"; the flip dumps exist in `analysis/out/probs_ab_flip` and `probs_w01_flip`, so they could be filled).
- Gain of the in-network prior on patient-unseen images with an earlier state (n = 80).
- A second validation fold; all ensemble, epoch and SWA choices share the scoring fold.

## Notable
- The "devices" example in the patient-prior mechanism sentence contradicts the per-class AP gains in this repo (M3). It survived the rev4 rewrite; the supplement discussion and the Markov draft carry a softer form of it.
- The two CI errors are both double roundings from 4 to 3 decimals; the supplement values are right.
- No transcription error exists between the drafts, the supplement and the main paper; Table I, the hit rates, the proxy numbers, the label-graph numbers and the in-network table all reproduce from artifacts.
- Resolved from the earlier Markov A/B note: its head / medium / tail grouping is the canonical `evaluate/class_groups.py` split (3 / 20 / 7).

## Next steps
- Apply the recommendations above, rebuild both PDFs, and re-run this check on the changed lines.
- Optional: compute mF1, mAUC and mECE for rows 7, 8, 10, 11 from the cached dumps so the "--" cells can be filled.
- Optional: report the matched two-seed baseline ensemble (0.4659) in the supplement.
