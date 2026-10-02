# Plan: rebuild the Markov layer as a post-prediction patient HMM / MRF prior

*Approved plan (2026-10-01), copied from plan mode. Builds on [`2026-09-30-patient-markov-research.md`](2026-09-30-patient-markov-research.md) and [`2026-09-30-markov-ab-training-plan.md`](2026-09-30-markov-ab-training-plan.md).*

## Context
The user wants the Markov component rebuilt from scratch, outside the network. ConvNeXt is only the feature extractor. Its classifier output P(Y|X) becomes the **emission** of a Markov model whose **prior / transition** P(Y_t | Y_{t-1}, patient metadata) carries the patient information. Nothing is inserted before or inside ConvNeXt.

Where we are (all internal val, `data/val_fold_1.csv`):
- **A**, the v4 in-network label-graph layer, is negative: SWA −0.003 to −0.005. A-strong (job 49377, lr-mult 100) reached SWA 0.4575 against 0.4600 for the v3 baseline (49000).
- **B-late** was a per-class 2-state chain on the prior report's labels with one global w. It gave +0.032 on has-prior images (CI excludes 0) but only +0.003 on full val. That result came from fixed w, no calibration and no metadata.
- **B-early** (v5, prior fed into the ML-Decoder queries; jobs 49375/49376, finished 2026-10-01 07:36) reached **SWA 0.4807 (s42) and 0.4786 (s1024)**, against 0.4600 for v3. It is not yet evaluated per subset or written up.
- **The question:** can a decoupled, interpretable post-hoc HMM with metadata-conditioned transitions match B-early without touching the network? The user also asked for an EDA of metadata–disease associations in the PadChest CSV.

**Scope:**
- Leaderboard effect is **0 by construction**. Test patients are de-duplicated, IDs and metadata are withheld, and joining dev/test to the CSV is forbidden because the CSV holds their labels.
- This is a clinical-longitudinal simulation on the internal folds, as material for `docs/paper/markov/`.
- The current Stage-2 head is ConvNeXt2 + ML-Decoder. The MoE path in v3 is inactive. The HMM takes any probs dump plus keys, so an MoE checkpoint can be plugged in unchanged.

## User decisions (2026-10-01)
- **History modes:** support both.
  - *Observed*: Y_{t-1} = the prior study's report labels.
  - *Filtered*: a true forward pass over the patient's earlier images.
- **Metadata:** sex, age at study, time gap, and view/acuity. An EDA of the CSV ranks further metadata–label associations.
- **Old code:** keep it and build new. `train/markov_layer.py`, `train/prior_conditioning.py`, `train_2_v4/v5.py`, `evaluate_tta_v4/v5.py`, `analysis/patient_prior.py` and `patient_markov_check.py` stay untouched, because they load or back the finished checkpoints and reports.
- **Within-study label–label MRF edges:** an ablation switch only. They are off in the main model, because A was negative.

## Data rules
- **CSV reads use a column whitelist (`usecols`, `dtype=str`).**
  - Allowed: ImageID, StudyID, PatientID, StudyDate_DICOM, PatientBirth, PatientSex_DICOM, Projection, ViewPosition_DICOM.
  - The EDA may also read the non-label DICOM acquisition fields (Modality, Manufacturer, exposure, resolution, rows/cols, window, MethodProjection, Pediatric).
  - **Never read:** Report, Labels, Localizations, LabelsLocalizationsBySentence, labelCUIS, LocalizationsCUIS, MethodLabel, ReportID, ImageDir.
  - The loader asserts these columns are absent.
- **Rule change (user-approved):** PatientBirth and sex were excluded before for data minimisation. They are now allowed, for train/val fold images only.
- **Labels come only from `label_list`** in `data/{train,val}_fold_1.csv`, in shard order (`label_info.pt`). Every joined ImageID is asserted to be in the folds, so no dev/test image is ever joined.
- **Prior model parameters are fitted on train-fold (patient, date) states only.**
  - Emission calibration and fusion weights are cross-fitted on val: 2 patient-grouped folds, each fitted on one half and scored on the other, repeated for split seeds {0, 1, 2}.
  - Val history uses train ∪ val studies, as in B-late and B-early, so the arms stay comparable.

## Model (`analysis/markov_hmm.py`)
The state is one (patient, date): the OR of that date's study labels, as in `patient_prior.date_states`.

Metadata features φ(m):
- sex {F, M, unknown};
- age = study year − birth year, as a spline basis (knots set from the EDA);
- acuity on that date: the most acute projection, AP_horizontal > AP > PA/L;
- g(Δ) = log1p(days since the previous state).

| Component | Form | Fitted on |
|---|---|---|
| Initial prior (no earlier state) | logit π⁰_c(m) = α⁰_c + γ⁰_c·φ(m) | train-fold first states |
| Transition | logit π_c = α_c + β_c·y_{t-1,c} + κ_c·y_{t-1,c}·g(Δ) + λ_c·g(Δ) + Σ_k B_ck·y_{t-1,k} + γ_c·φ(m_t) | train-fold consecutive pairs |
| Transition, `table` variant | a_c / b_c from `patient_prior.fit_chain`, Laplace-smoothed | same (regression anchor) |
| Emission | ℓ_c(x) = logit p̃_c(x) − logit π̄_c, where p̃ = temperature (one shared slope) + per-class intercept with L2 shrinkage, and π̄ = train prevalence | val, cross-fit |
| Within-study MRF (ablation) | pairwise J (30×30 symmetric) from pseudo-likelihood on train states; damped mean-field over the current state, weight w_J | train states / w_J cross-fit |

- **Fitting:** all 30 per-class logistic models are fitted as one batched torch model with L-BFGS and L2 (stronger L2 on B). The penalty is chosen by patient-grouped 5-fold CV log-loss on train pairs. There are no new dependencies; the venv has torch and scipy but not sklearn.
- **Observed mode, target image i on date s:**
  - q = π(y_{s-1}, m_s, Δ) when an earlier state exists, otherwise π⁰(m_s).
  - Score z_c = logit p̃_c(x_i) + w_h·(logit q_c − logit π̄_c) with a prior, or + w_0·(logit π⁰_c − logit π̄_c) without one.
- **Filtered mode (factorised Boyen–Koller belief):**
  - b₁ = σ(logit π⁰ + w_e·ē₁), where ē is the mean ℓ over that date's images.
  - Predict: q_c = b_c·σ(η_c(1, b₋c)) + (1 − b_c)·σ(η_c(0, b₋c)). This is exact when B = 0 and mean-field otherwise.
  - Update: b_s = σ(logit q + w_e·ē_s).
  - The target uses q from b_{s-1}, plugged into the same z formula.
  - Only the target image itself enters at the current date, so there is no same-date sibling fusion, which would be an unrelated multi-view gain.
- **Weights:** w_h, w_0, w_e and w_J are chosen per fit-half by maximising macro mAP over the grid {0, 0.25, …, 1.5}, as CV-w did.
- **API:** `MarkovHMM.fit(...)`, `.save/.load` (`analysis/out/markov_hmm/model.pt`), and `.apply(probs, keys, mode)` returning fused probs, so any emission dump can be post-processed.

## Arms (pre-specified; primary arm = H4)
| Arm | What |
|---|---|
| E | emission only: 49000 SWA (s42), 48999 SWA (s1024), greedy-3 ensemble; cached `probs_w01_flip` |
| H0 | `table` transition, w_h = 1, no calibration. **Must reproduce B-late exactly.** |
| H1 | H0 + calibration + cross-fitted w_h |
| H2 | + gap terms κ, λ |
| H3 | + metadata in transition and initial prior (w_0 > 0, which now also moves no-prior rows) |
| **H4** | + cross-class temporal edges B (full temporal MRF prior) |
| H5 | H4 + within-study MRF edges (ablation) |
| H4-acq | H4 + EDA-flagged acquisition fields (only if the EDA's train-CV shows a held-out gain; flagged as care-setting proxies) |
| H4-F | H4 in filtered mode (image-only history) |
| Comparators | B-early s42/s1024 SWA (prior with / zero), A-strong. From the pending A/B eval dump. |

## Steps
0. **Prerequisite, already approved on 2026-09-30 (A/B plan step 4.3):**
   - Run the eval dump of the 49375–49377 SWA checkpoints, plus 49000 and 48999, into `analysis/out/probs_ab_flip/` with `keys.txt` (`scripts/evaluate_tta_v5.sh`).
   - Then the `cxrlt-research-analyst` subagent writes `docs/result/2026-10-0X-markov-ab-training.md`.
1. **`analysis/patient_meta.py` (new).**
   - The whitelisted loader: per image, sex, birth, date, projection and StudyID/PatientID. It has a date-state table with metadata and acuity, and a `featurize()` for φ.
   - It reuses `patient_prior.{label_matrix, decode, logit, PADCHEST, LABEL_INFO, study_table, date_states, lookup_prior}`.
2. **`analysis/metadata_eda.py` (new; load the `dataviz` skill before writing the figure code).** Train fold only for the screening; val is used only for the incremental check.
   - **Univariate screen:** each metadata field × 30 classes.
     - Units: study for patient/study fields, image for view/acquisition fields.
     - Categorical fields: prevalence per level, Cramér's V, mutual information, and log-OR with a patient-clustered bootstrap CI.
     - Continuous fields: single-feature AUROC and binned prevalence curves.
     - Benjamini–Hochberg FDR across the screen.
   - **Multivariate:** a per-class metadata-only logistic model under patient-grouped 5-fold CV, giving held-out AUROC/AP against prevalence, with feature-group drop-out (patient / temporal / acquisition).
   - **Transition modulation:** persistence a_c by sex, age band and acuity, which motivates the conditioned transition.
   - **Incremental value over the image:** on val, cross-fitted stacking of logit p + metadata against logit p alone, as ΔmAP with a patient-clustered CI.
   - Outputs go to `analysis/out/metadata_eda/` (`eda.json`, `assoc.csv`, `figures/`). The age knots and acquisition-feature flags feed step 3.
3. **`analysis/markov_hmm.py` (new):** the library described above.
4. **`analysis/markov_hmm_check.py` (new): the runner.** It writes `analysis/out/markov_hmm/{run.log, results.json, model.pt}` and runs E/H0–H5/H4-acq in observed mode for both seeds and the ensemble.
   - **Held-out transition diagnostics:** log-loss and AUROC of the nested prior models on train-CV pairs. This shows what each prior term explains, without touching val.
   - It reuses `patient_markov_check.{subset_map, per_class_ap, by_patient, bootstrap}`, `markov_ab_compare.{stratum_map, eligible}`, `markov_check.{KEY_ORDER, PROBS, ENSEMBLE, FRONTAL}`, `ens_select.macro_map` and `class_groups.{load_train_prevalence, frequency_groups}`.
   - It has a `--limit N` dry-run mode.
   - Run it with `srun -c 4 --mem=16G -p general` (CPU).
5. **GPU: one eval job.** Dump the 49000 SWA flip probs on the train shards:
   - `sbatch --time=8:00:00 -x colossus scripts/evaluate_tta_v5.sh --val-shards-dir /data/psytp7/wds_shards_train_raw --dump-probs analysis/out/probs_train_flip <49000 swa>@768`.
   - Then run H4-F.
   - Train-image emissions are **in-sample** (memorised), so H4-F is optimistic on rows whose history is in train. The clean test is the subset whose whole history is in the val fold (small n; the runner reports it).
6. **Reports and memory.**
   - Save this plan as `docs/plan/2026-10-01-markov-hmm-redesign-plan.md`.
   - The `cxrlt-research-analyst` subagent writes `docs/plan/2026-10-0X-padchest-metadata-eda.md` and `docs/plan/2026-10-0X-markov-hmm-posthoc.md`. These are offline analyses, so they go in docs/plan, linked to the A/B results doc.
   - Update memories: the campaign memory (Markov verdict), `data-split-patient-leakage` (the sex/birth rule change), and a new memory for the HMM model location and verdict.

## Evaluation and decision rule (fixed before any numbers)
- **Subsets:**
  - full val;
  - has-prior;
  - **has-prior & frontal (PRIMARY)**;
  - has-prior & study-not-in-train;
  - no-prior (where H3+ now acts);
  - patient-unseen (descriptive);
  - filtered-clean (H4-F only).
- **Metrics:**
  - macro mAP, and a paired 300-sample patient-clustered bootstrap Δ against E at the matched seed;
  - head/medium/tail ΔAP;
  - changed-finding stratum AP (new vs absent, persistent vs resolved), as the copy check;
  - mECE before and after;
  - the range over split seeds.
- **Win (internal, longitudinal):** H4 beats E on the primary subset with a CI excluding 0, at both seeds.
- **Matches B-early:** the paired Δ(H4 − B-early, matched seed) CI includes 0 or is positive on the primary subset, and on full val.
- **Guards:**
  - H4 is not worse than E on no-prior rows, and not worse on new-finding stratum AP;
  - H5 and H4-acq are reported as ablations only.
- **Leaderboard verdict:** not applicable (0), stated explicitly.

## Files
- **New:**
  - `analysis/patient_meta.py`
  - `analysis/metadata_eda.py`
  - `analysis/markov_hmm.py`
  - `analysis/markov_hmm_check.py`
  - outputs under `analysis/out/{metadata_eda, markov_hmm, probs_train_flip}/`
- **Untouched:** every existing Markov, v4/v5, eval and prior file listed under User decisions. `scripts/evaluate_tta_v5.sh` is reused as is, with the `--time` override at submit.

## Verification
- **Asserts inside the runner:**
  - H0 reproduces B-late (`analysis/out/patient_markov/results.json`) to 1e-4: has-prior w = 1 gives 0.5299, full val 0.4752.
  - All weights = 0 gives the emission exactly. The ensemble base is 0.4722.
  - Filtered mode with history beliefs set to one-hot labels equals observed mode exactly.
  - The unpenalised intercept + β logistic recovers the unsmoothed chain counts to 1e-3.
  - w_J = 0 gives H5 = H4.
  - The CSV column whitelist holds, and every key is in the folds.
  - The val row order reproduces `labels.npy`.
- **EDA sanity:** expected strong associations show up, for example age ↔ aortic elongation and cardiomegaly, and AP_horizontal ↔ Support Devices and effusion.
- **Run order:** a `--limit 2000` dry run first, then the full CPU runs (minutes). The GPU dump takes about 3–4 h for the 11 train shards at 768 px with flip. Check `sacct` for any missing artifact.
