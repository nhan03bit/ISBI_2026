# Plan: Patient-history Markov chain vs. Markov fine-tune layer (research study)

*Approved plan (2026-09-30), copied from plan mode. Findings: [`2026-09-30-patient-markov-research.md`](2026-09-30-patient-markov-research.md). Under the 2026-09-30 document policy the report lives in `docs/plan/`, not `docs/result/`.*

## Context
The user asks two things:
1. How to integrate the PadChest patient metadata (`data/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv`)
   into a Markov chain.
2. Which approach is best:
   - **A.** a Markov layer as a fine-tune head on ConvNeXt (label-graph);
   - **B.** a patient-history Markov chain applied "before" ConvNeXt.

What we already know:
- **A has been tested and is negative.**
  - Post-hoc smoothing: −0.0006 to −0.0053 mAP.
  - The trainable v4 layer (jobs 49063–49065): SWA −0.003 to −0.005 vs job 49000. The gates stay near 0.
  - Sources: `docs/result/2026-09-27-v4-markov-layer.md`, `docs/paper/markov/paper.tex`.
- **B cannot be used on the leaderboard.**
  - The test patients are de-duplicated against training, and patient IDs are withheld.
  - The CSV contains the `Report` and `Labels` of **every** PadChest image, including the PadChest-GR
    evaluation images. Joining test or dev images to it would read out their own labels.
  - The data-use terms forbid that, and forbid re-identification.
  - So B can only be studied as a **clinical-deployment simulation on internal data**. The user chose
    this offline simulation.
- **New finding (plan-phase check, read-only):**
  - 53.2% of val images (9,189) share their StudyID with a train image, and 100% of those have
    identical label lists. PadChest labels come from the report, which covers every view in a study.
  - The internal val is therefore partly label-leaked at the study level, which is stronger than the
    77% patient overlap in the paper.
  - The user chose: quantify it in the report and record it in memory. The paper is not edited.
- **Prior-study availability:** 33.4% of val images (5,761) have a strictly earlier study of the same
  patient; 31.9% have one in the train fold; 33.9% of frontal val images have one.

**Outcome:** a research report that answers "A vs B". It gives quantitative evidence for B in the
simulated setting, a rules/availability verdict for the leaderboard, and a literature review.

## Hard rules
- Never join or look up dev/test/leaderboard images in the PadChest CSV.
- Load only `ImageID, StudyID, PatientID, StudyDate_DICOM, Projection` from it, with `dtype=str`.
  Never load `Labels`, `Report`, `PatientBirth` or sex.
- Labels come only from the 30-class `label_list` of `data/{train,val}_fold_1.csv`.
- Report aggregates only: no IDs, no per-patient rows.
- No training or eval jobs, and no GPU. CPU analysis runs via `srun` (README: nothing
  compute-heavy on the login node).

## Steps

### 1. New script `analysis/patient_markov_check.py` (CPU)
**Reuse** from `analysis/markov_check.py` (import it; its `main` is guarded):
- `PROBS`, `ENSEMBLE`, `LABEL_INFO`, `KEY_ORDER`, `FRONTAL`, `SUBSET_MODELS`;
- the `train_label_matrix` pattern, mapping `label_list` to shard/logit class order via `label_info.pt`;
- the row alignment in `val_subsets()` (`va = read_csv(val).iloc[np.load(KEY_ORDER)]`);
- `ens_select.macro_map`.

Point `PADCHEST` at the `data/` copy, falling back to `/data/psytp7/`.

**Sections**
1. **Study-level leak.** The fraction of val images whose StudyID is in train, and whether their
   labels are identical. Score the greedy-3 ensemble (flip) on:
   - study-in-train vs not;
   - the same, restricted to frontal images;
   - "patient unseen" (a clean superset check).

   Differences get a patient-clustered bootstrap: 300 resamples, 95% percentile interval, stated
   explicitly.
2. **Patient Markov chain (the answer to question 1).**
   - Collapse to one 30-label vector per StudyID.
   - Sort each patient's studies by `StudyDate_DICOM`. Drop same-date different-study pairs, since
     their order is ambiguous.
   - Build consecutive-study pairs from **train-fold patients only**.
   - Per class c, fit a 2-state chain with Laplace smoothing:
     - a_c = P(y_t=1 | y_{t−1}=1)
     - b_c = P(y_t=1 | y_{t−1}=0)
     - the marginal π_c
   - Also report a_c and b_c by gap bin (≤30 d, 31–365 d, >365 d) where n ≥ 50.
   - Output a per-class persistence table (head/medium/tail via `evaluate/class_groups.py`).
3. **History-only predictor.**
   - For each val image with a strictly earlier study (different StudyID), take the most recent prior.
   - Score each class with π_c(i) = a_c if the prior has c, else b_c.
   - Report mAP on the has-prior subset. It is tie-heavy; say so.
4. **HMM late fusion (approach B done properly).** The ConvNeXt output is the emission; the chain is
   the prior:
   - logit q_c = logit p_c + w·(logit π_c(i) − logit π_c);
   - w = 1 is the pure Bayes update, with nothing fitted;
   - w is also fitted by 2-fold patient-grouped CV over the has-prior val images, on the grid
     {0, 0.25, …, 1.5}.

   Report baseline vs fused mAP, with a bootstrap Δ and CI, on:
   - has-prior (all);
   - has-prior & frontal;
   - has-prior & prior-in-train;
   - has-prior & patient-unseen-in-train (prior in val). This is the cleanest clinical simulation.
   - all of the above restricted to the study-not-in-train images, to separate out the leak.

   Also project the effect on the full val set, and state that the leaderboard effect is 0 because
   no history exists there.
5. Print everything in labelled blocks, like `markov_check.py`. Save the tables as JSON in
   `analysis/out/patient_markov/`.

Run: `srun -c 4 --mem=16G -p general --time=0:45:00 .venv/bin/python analysis/patient_markov_check.py | tee analysis/out/patient_markov/run.log`

### 2. Literature review (main session; the subagent has no web tools)
Use WebSearch and WebFetch. Get all citation metadata from arXiv, Crossref or PMC, never from memory.
- Longitudinal and prior-study CXR classification. Candidates to check: BioViL-T, MS-CXR-T,
  CheXTemporal, time-modulated LSTM for longitudinal CXR, CheXRelNet / Chest ImaGenome comparisons.
- HMMs for disease progression in imaging.
- Evidence on shortcut learning when prior labels or reports are fed as model input.

Save the notes to the scratchpad.

### 3. Report via the `cxrlt-research-analyst` subagent
Self-contained prompt:
- Inputs: the script output, the JSONs, the literature notes, the v4 report and the markdown note.
- Target: `docs/plan/2026-09-30-patient-markov-research.md`.
- Required: a decision table for A vs B-late-fusion vs B-early (input-level) with the criteria
  - available at test time;
  - allowed by the rules;
  - internal evidence;
  - cost;
  - shortcut/robustness risk.
- Required: a "Notable" section.
- It must verify every number against `run.log` and check shard-order class alignment.

If it returns text instead of writing the file, save it myself and add a main-session notes section,
as last time.

**Expected verdict** (to be confirmed by the numbers):
- **Leaderboard:** neither A nor B helps. A is tested negative; B is unavailable and forbidden.
- **Clinical/longitudinal study:** B as post-ConvNeXt HMM filtering (late fusion) rather than
  input-level conditioning. It is principled (the emission × transition factorisation), needs no
  backbone retraining, degrades to the population prior when there is no history, and avoids the
  prior-label shortcut.
- Early fusion is described only, not run. It would need a GPU job via `data-scientist`.

### 4. Memory (skill step 4)
- `data-split-patient-leakage.md`: add the study-level leak (53.2%, identical labels), the ensemble
  study-in-train vs not scores, and the rule that the CSV has labels for the evaluation images, so
  it must never be joined to dev/test.
- `stage2-map-improvement-campaign.md`: a one-line pointer to the new report.

## Files
- **New:** `analysis/patient_markov_check.py`, `analysis/out/patient_markov/*`,
  `docs/plan/2026-09-30-patient-markov-research.md`.
- **Edited:** the two memory files.
- **Not touched:** the paper, the training code, and any test/dev data.

## Verification
1. Sanity checks inside the script:
   - Section 1 must reproduce the plan-phase counts (9,189 / 53.2%; 5,761 / 33.4%).
   - The baseline ensemble mAP on all rows must equal 0.4722, the `markov_check.py` value.
   - Aligned labels must equal `labels.npy` exactly (`assert`).
   - w = 0 fusion must equal the baseline exactly.
   - Laplace-smoothed a_c and b_c must lie in (0, 1).
   - No prior may share a StudyID or a date with its target (`assert`).
2. Confirm with grep that the script never reads the `Labels`, `Report` or `PatientBirth` columns and
   never touches any dev/test path.
3. Cross-check every number in the report against `run.log` (the subagent, then my spot-check).
4. Relay to the user: the report path, a 2–3 line verdict on A vs B, and the leak magnitude.
