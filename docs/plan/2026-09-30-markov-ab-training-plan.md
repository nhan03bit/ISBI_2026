# Plan: train and compare A (label-graph Markov layer) vs B (patient-history Markov chain)

*Approved plan (2026-09-30), copied from plan mode. Findings it builds on: [`2026-09-30-patient-markov-research.md`](2026-09-30-patient-markov-research.md).*

## Context
`docs/plan/2026-09-30-patient-markov-research.md` compared the two ideas, but only partly by training:
- **A** (a trainable label-graph Markov layer after the ML-Decoder) was trained once at seed 42 (jobs 49063–49065). It was negative, but its gates barely moved from zero.
- **B** (a patient-history Markov chain) was only simulated offline (job 49370: 4 CPU minutes, which is why it "returned so fast"). On val images with a prior study, late fusion gave +0.032 mAP. **B-early** (feeding the patient's chain prior *into* the network) has never been trained.

The user wants a full training comparison to decide which approach is more effective, and has approved GPU jobs. Decisions made:
- B-early injects the prior into the ML-Decoder **label queries**.
- The GPU budget is **3 training jobs**: B-early at seeds 42 and 1024, and A retried with a 10× larger Markov learning-rate multiplier.

**What the result can and cannot say:**
- **Leaderboard.** B cannot be used on the leaderboard. Test patients are de-duplicated against training and IDs are withheld, and the PadChest CSV holds the evaluation images' labels, so dev/test must never be joined to it. For the leaderboard, only A is admissible.
- **Internal val.** The A-vs-B comparison is decided there, as a longitudinal simulation. Both conclusions are reported separately.

## Hard rules (unchanged)
- No dev/test image is ever joined to the PadChest CSV. Only read ImageID / StudyID / PatientID / StudyDate_DICOM / Projection, as `dtype=str`.
- Labels come only from `data/{train,val}_fold_1.csv` `label_list`, in `label_info.pt` (shard) order.
- **Training-time priors use train-fold history only, never val labels.** Val priors use train ∪ val history (a clinical simulation, the same as in the research study).

## Design: every arm sees the same information
The prior vector is r = [logit π_c(prior) − logit prevalence_c]₍c=1..30₎ ⊕ has_prior. Here π comes from the per-class 2-state chain fitted on consecutive train-fold studies, and r = 0 when there is no prior. It is exactly the quantity B-late adds to the logits. So B-early vs B-late isolates *where* the chain prior enters (the network's queries vs its output), and nothing else.

| Arm | What | Training | Seed(s) |
|---|---|---|---|
| v3 baseline | existing jobs 49000 (s42) and 48999 (s1024) | none (exists) | 42, 1024 |
| A | existing v4 mk1 / mk2 / mk1fix | none (exists) | 42 |
| A-strong | v4 mk1, `--markov-lr-mult 100` | **new job** | 42 |
| B-late | baseline probs + chain fusion (w=1 and CV-w) | none (CPU) | 42, 1024 |
| B-early | label-query conditioning on r | **2 new jobs** | 42, 1024 |

All new runs use the 49000 recipe:
- 768 px, drop-path 0.1;
- 4 of 8 cosine epochs, SWA over the last 3 EMA epochs;
- triplet λ 0.1, memory 2048, bs 4 × accum 12;
- `--class-weight-order shard`, `--metric-embed query`;
- the same Stage-1 checkpoint (`checkpoint3/Model_20260119_062652/model_best.pth`).

## Steps

### 1. Shared prior builder (CPU): `analysis/patient_prior.py` (new)
- Move the chain and prior logic out of `analysis/patient_markov_check.py`: the date-state OR, the chain fit (a, b, Laplace), the strictly-earlier `merge_asof` lookup, and the 1-study ambiguity exclusion. `patient_markov_check.py` then imports it.
- Its CLI writes `analysis/out/patient_markov/priors.pt` containing `{class_names, a, b, prevalence, train: {fname: r31}, val: {fname: r31}}`, keyed by the shard `fname` (= ImageID `.png`). History for the train entries is train-fold only; for the val entries it is train ∪ val.
- **Regression check:** rerun `patient_markov_check.py` after the refactor. Its `run.log` must be byte-identical apart from timing. Also assert that the val r from `priors.pt` equals the fusion `log_ratio`.

### 2. Model code: B-early and the A-strong launcher
- **`train/ml_decoder.py`:** `Decoder.forward(self, x, mask=None, query_offset=None)`. After `tgt` is built (line ~371): `if query_offset is not None: tgt = tgt + query_offset.transpose(0, 1)`. The default `None` leaves v3/v4 unchanged.
- **`train/prior_conditioning.py` (new; mirrors `train/markov_layer.py`)**
  - `PriorQueryConditioner`: offset[b, c] = r[b, c]·u_c + W·r[b].
    - u is 30×768; W is `Linear(31→768, bias=False)`. **Both are zero-initialised.**
    - There is no bias, so no-prior samples (r = 0) always get exactly zero offset.
    - It computes in fp32 with autocast disabled, as the Markov layer does.
  - `ConvNeXt2Prior(ConvNeXt2)`: `forward(x, prior=None)` runs `forward_features` → pos-enc → `head(x, query_offset=cond(prior))`.
  - `STATE_PREFIX = "prior_cond."`.
  - A loader that dispatches between v3, v4 (Markov) and v5 (prior) checkpoints, reusing `load_convnext2_for_state_dict`. It raises if any prior key fails to load.
- **`train/train_2_v5.py` (new)** is a copy of `train_2_v4.py`, following the repo's version-per-file convention (v3/v4 stay untouched for reproducibility). The additions:
  - Flags: `--prior-file`, `--prior-dropout 0.3` (train only), `--prior-lr-mult 10`.
  - `DecodeAndTransform` gains a `prior_lookup`. `decode_and_transform` looks up `sample["fname"].decode()` and returns `(x, y, r)`. Dropout is deterministic from `stable_seed_from_key(key, epoch)`.
  - Unpacking at L841 and L952 becomes `(x, y, r)`, and the calls at L853 and L963 pass `prior=r`.
  - The conditioner params get their own optimizer group (lr × mult, wd 0). They are excluded from the LLRD coverage assert, as `refine_ids` are.
  - A per-epoch `prior:` log line reports |u| and |W| and the batch has-prior fraction.
  - Sanity checks: `create_model` asserts that the missing keys are exactly `prior_cond.*`, and a run fails if the prior-lookup hit rate is below 99% of the fold images.
- **`scripts/train_2_v5.sh` and `scripts/submit_stage2_v5_prior.sh` (new)**, copying the v4 pair. They share v4's COMMON args, `SMOKE=1` support, `-x colossus` and `-p amp48,ada24`. Output goes to `checkpoint_Triplet_3/prior/img768_qprior_seed{S}/Model_run`.
- **`scripts/submit_stage2_v4_markov.sh`:** add `_lr${LR_MULT}` to the out-dir and job name when `LR_MULT != 10`, so A-strong writes to `markov/img768_mk1_lr100_seed42/` and cannot overwrite the existing mk1 results.

### 3. Pre-submission checks (CPU, via `srun`)
- **Identity at init:** in eval mode, load Stage-1 into `ConvNeXt2` and `ConvNeXt2Prior`. On random inputs with random r, the logits must be **exactly equal** (max |Δ| = 0).
- **Gradient flow:** one backward pass must give u and W non-zero gradients, and the offset must be exactly 0 for r = 0.
- **Prior sanity:** look up 1,000 train and 1,000 val `fname`s and confirm the has-prior fractions (val ≈ 0.333, per the research study).

### 4. GPU jobs
1. **Smoke tests** (40 batches, per the repo pattern): v5 seed 42 and A-strong. Check that the logs show the missing keys equal to `prior_cond.*` only, the `prior:` / `markov:` lines, and no errors.
2. **Full runs, 3 jobs, about 10 h each, in parallel:** v5 at s42 and s1024, and v4 mk1 with lr-mult 100 at s42.
3. **Eval dump**, with a new **`evaluate/evaluate_tta_v5.py`** (a wrapper in the style of v4 around `evaluate_tta.py`):
   - Flip-TTA probabilities, plus **`keys.txt` (the fnames in row order)**. This finally verifies the reconstructed `val_key_order.npy`.
   - A `--prior-file`, with `--prior-mode {with,zero}` for v5 checkpoints.
   - Into `analysis/out/probs_ab_flip/`, dump SWA checkpoints for: 49000, 48999, mk1, mk2, mk1fix, mk1_lr100, qprior s42 (with and zero), and qprior s1024 (with and zero).

### 5. Comparison (CPU): `analysis/markov_ab_compare.py` (new)
- Reuse `macro_map`, the subset masks and the patient-clustered bootstrap from `patient_markov_check.py` (via `patient_prior.py`).
- Assert that `keys.txt` equals the fold-CSV order implied by `val_key_order.npy`, and that labels equal `labels.npy`.
- **Subsets:**
  - full val;
  - has-prior;
  - **has-prior & frontal (PRIMARY)**;
  - has-prior & study-not-in-train;
  - no-prior;
  - patient-unseen (n = 80, descriptive only).
- **Metrics:**
  - macro mAP and paired bootstrap Δ vs the matched-seed baseline (300 resamples, 95% percentile CI);
  - head/medium/tail ΔAP;
  - B-early vs B-late, paired;
  - **stratified "changed-finding" AP** (the copy-risk test). Per class, over has-prior images, AP within the prior-negative stratum (new vs absent) and within the prior-positive stratum (persistent vs resolved). Macro over classes with ≥ 5 positives and ≥ 5 negatives per stratum. B-late leaves within-stratum ranking unchanged by construction; B-early can change it either way.
- **Decision rule (fixed before the numbers):**
  - **Internal / longitudinal winner:** the arm with the largest primary Δ vs baseline whose CI excludes 0 and whose sign replicates at seed 1024 (A-strong has one seed, flagged). B-early vs B-late is judged on its paired Δ.
  - **Guard 1:** B-early with zero priors must not be worse than the baseline on no-prior images (the CI must not be entirely below 0).
  - **Guard 2:** B-early must not be worse than the baseline on new-finding stratum AP (the copy check).
  - **Leaderboard verdict:** A or A-strong vs baseline on full val and on frontal only. B is not applicable.

### 6. Reports and memory (per the document policy)
- After approval, save a copy of this plan as `docs/plan/2026-09-30-markov-ab-training-plan.md`.
- The `cxrlt-research-analyst` subagent writes the **training/eval results** of the 3 new runs and the eval dumps to `docs/result/<date>-markov-ab-training.md`, checked against known bugs.
- I write the **verdict** (A vs B-late vs B-early, with the decision-rule outcome) to `docs/plan/<date>-markov-ab-verdict.md`. The two files link to each other.
- Memory: update `stage2-map-improvement-campaign.md`, and `data-split-patient-leakage.md` if the keys check changes anything about the row order.

## Files
- **New:** `analysis/patient_prior.py`, `analysis/markov_ab_compare.py`, `train/prior_conditioning.py`, `train/train_2_v5.py`, `evaluate/evaluate_tta_v5.py`, `scripts/train_2_v5.sh`, `scripts/submit_stage2_v5_prior.sh`.
- **Edited:**
  - `train/ml_decoder.py` (an optional kwarg only);
  - `analysis/patient_markov_check.py` (imports the shared logic; same output);
  - `scripts/submit_stage2_v4_markov.sh` (the lr tag).
- **Untouched:** `train_2_v3.py`, `train_2_v4.py`, `markov_layer.py`, `convnext.py`, `evaluate_tta.py`, `evaluate_tta_v4.py`.

## Verification
- **Before GPU:**
  - the `patient_markov_check.py` regression is byte-identical;
  - val r equals the B-late `log_ratio`;
  - `ConvNeXt2Prior` at init is exactly equal to `ConvNeXt2`, with non-zero conditioner gradients;
  - the smoke logs are clean.
- **After training:**
  - every log has `Done. Best` / `SWA weights ->` lines;
  - `sacct` states are checked for any missing artifact;
  - `keys.txt` matches `val_key_order`;
  - for existing checkpoints, the dumped 49000 SWA probs reproduce `probs_w01_flip` exactly.
- **Timeline:** smoke tests about 15 min, then about 10–12 h of training in parallel, about 1–2 h of eval, and minutes of CPU comparison. I will watch the jobs and relay the verdict.
