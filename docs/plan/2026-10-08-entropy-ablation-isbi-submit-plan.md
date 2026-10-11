> Copy of the plan-mode plan approved on 2026-10-08 (docs policy). It is also the pre-registration of the ablation's primary contrast (C − B on mAP, paired by seed), saved before any ablation run was submitted.

# Plan: entropy ablation (supervisor's missing validation) + 4-page ISBI 2027 submission

## Context
The supervisor wants a clean like-for-like ablation before the entropy gate can be credited:

- baseline → triplet/memory bank **without** entropy → triplet/memory bank **with** entropy;
- resolution, schedule and checkpoint fixed;
- head/medium/tail AP reported.

The 4-page ISBI paper should focus tightly on that result. Today no run switches off the gate or the triplet term:
- all 67 `args:` lines have `--triplet-lambda 0.1`;
- `train_2_v3.py` hard-codes the gate in `utils_update.sample_triplets_v13`, at L251-258.

So the 0.439 → 0.472 gains cannot be attributed to entropy (memory: stage2-map-improvement-campaign, 2026-10-02).

Deliverables:
1. the ablation runs and their analysis;
2. a new entropy-only paper in `docs/paper/isbi_submit/` that meets ISBI 2027 rules.

ISBI 2027 rules (biomedicalimaging.org/2027/papers):
- 4-page paper deadline is **Mon 26 Oct 2026**, 23:59 EDT, via EDAS.
- All technical content goes in pages 1-4. An optional page 5 ($200 fee) may hold only references, the mandatory "Compliance with Ethical Standards" section and acknowledgments.
- Review is single-blind.
- AI-generated text, figures or code must be disclosed in the acknowledgments.

## User decisions (2026-10-08)
- **Controls:** both "without entropy" controls, no gate (B) and random gate (B′), with 6 seeds each.
- **MoE-study jobs:** **hold** the 7 pending jobs 51153-51159 until every ablation run has started, then release them. Their 11 dependent evals (51181-51189) simply wait. The 2 running MoE jobs (51151/51152) continue.
- **Resolution:** **512 px only.** The 768/1024 px results appear in the paper only as the scaled method, stated as not ablated.
- **Layout:** **4 pages of technical content + page 5** for references, ethics and acknowledgments (the $200 fee is paid at registration).

## Fixed setting (reuses the existing 6-seed entropy arm)
The fixed recipe is the v3 512 px seed-sweep recipe (`scripts/submit_stage2_v3_seedsweep.sh`, jobs 46472-47099):

- `--accum-steps 6 --batch-size 8 --monitor mAP --epochs 8 --patience 3 --lr 1e-4 --memory-size 2048 --metric-embed query --head-classes 5 --class-weight-order shard`
- `--stage1-ckpt checkpoint3/Model_20260119_062652/model_best.pth`
- 512 px, aug-base-seed 42, seeds 42/86/123/456/789/1024.

These runs serve as the entropy arm C:
- per seed: 0.4414 / 0.4385 / 0.4389 / 0.4375 / 0.4369 / 0.4434;
- best epoch 2-3, stopped at epoch 5-6, 7.4-12.1 h per run.

Code parity is verified: the config logged by job 47097 (Sep 10) matches today's v3 except `rrc_scale_min` (new key, default 0.9 = the old hard-coded value). The git diffs since then are behaviour-neutral. The seed fixes shard order and dataloader RNG, so arms with the same seed see the same data order and can be compared seed by seed.

| Arm | Stage-2 objective | Flag(s) | Runs |
|---|---|---|---|
| A baseline | ASL only (triplet weight 0; bank and mining still run, zero gradient → same code path) | `--triplet-lambda 0` | 6 new |
| B no entropy | ASL + triplet/bank, **every** sample is an anchor | `--anchor-rule all` | 6 new |
| B′ random gate | ASL + triplet/bank, random anchors, **same count** the gate would pick | `--anchor-rule random` | 6 new |
| C entropy | ASL + triplet/bank, anchors with entropy ≥ batch mean (method) | `--anchor-rule mean` (default) | 6 existing + rerun of seeds 86 and 1024 as a reproducibility check (training is non-deterministic beyond seed; v5 rerun s.d. 0.003) |

Stage 1 at 512 px (0.3688; head/medium/tail 0.705/0.377/0.203; `eval_result.stage1.json`) is a context row only, because Stage 1 was trained at 384 px.

## Step 0 — housekeeping
- Save the previously approved MoE-rebuild plan (in this conversation's context or its transcript) to `docs/plan/2026-10-08-moe-matched-baseline-plan.md` (docs policy).
- Save this plan, once approved, as `docs/plan/2026-10-08-entropy-ablation-isbi-submit-plan.md`. It is the pre-registration of the primary contrast below, written before any result exists.
- `scontrol hold 51153 51154 51155 51156 51157 51158 51159`.
- Release them (`scontrol release …`) once every ablation run is RUNNING or done. The background waiter checks this condition.

## Step 1 — code (backward compatible; defaults reproduce every existing run)

**`train/utils_update.py`**
- `_anchor_gate(entropy, rule, generator=None)`: when a generator is passed, "random" draws `torch.randperm` from that CPU generator. The global CUDA RNG is then untouched, so the dropout and shuffle streams match arm C.
- `sample_triplets_v13(..., anchor_rule="mean", generator=None, stats=None)`:
  - replace the hard-coded `entropy[i] < entropy.mean()` skip with `gate = _anchor_gate(entropy, anchor_rule, generator)`;
  - if a `stats` dict is passed, accumulate the number of gated anchors and the number of mined triplets into it.
  - "mean" must stay bit-identical to today. `train_2_v5.py` and other callers keep the defaults.

**`train/train_2_v3.py`**
- New flag `--anchor-rule {mean,all,random}` (default `mean`), stored in `cfg`, and printed in the existing `v3:` log line.
- Trainer: `self.anchor_gen = torch.Generator().manual_seed(seed + 1)`, used only by "random".
- Pass the rule, generator and a per-epoch stats dict to the sampler.
- New epoch-end log line `[Epoch e] anchors/step=… triplets/step=… rule=…`, also stored in `history["triplet_stats"]`.
- Nothing else changes.

**`scripts/submit_stage2_v3_entropy_ablation.sh`** (modelled on the seed-sweep and nolam scripts; supersedes the unsubmitted nolam sweep)
- Run specs are `arm:seed`, with arms `asl | nogate | randgate | gate`.
- Base args are exactly the seed-sweep's; each arm adds its flag from the table.
- Output: `train/checkpoint_Triplet_3/ablation512/<arm>/seed_<S>/Model_run`. Job name: `isbi2026_abl_<arm>_s<S>`.
- Slurm: `-p amp48,ada24` (plus `amp20` if the smoke test fits in 20 GB), `-x colossus`, `-c 8`, `--mem 64G`, `--time 24:00:00`.
- `SMOKE=1` runs 40 training and 20 validation batches.
- Each training job is followed by an `--dependency=afterok` call to `scripts/eval_perclass.sh <ckpt> 512 <arm>_s<S>`. This writes `eval_result.<tag>.json` (macro, group mAP, per-class AP, val positives) and `preds.<tag>.npz`, and reproduces the training-time mAP exactly (seed 86: 0.438524 both).
- Submission order puts seeds 86, 123 and 1024 of every arm first, so a 3-seed read exists early.
- Also submit `eval_perclass.sh` now for the three entropy-arm seeds that lack it: `seedsweep/seed_{42,456,789}` → tags `s42 / s456 / s789`.

## Step 2 — checks, then submit
1. CPU unit check (scratch script, not committed):
   - the new `sample_triplets_v13(rule="mean")` returns the same triplets as a frozen copy of the old function, on random logits/labels/bank including entropy ties;
   - "all" gates B samples; "random" gates exactly the count of entropy ≥ mean and leaves `torch.get_rng_state()` unchanged.
2. `SMOKE=1` for all four arms. In the logs:
   - `anchors/step` ≈ about half of 8 (mean and random), 8 (all), any value (asl, with zero gradient);
   - `triplet_lambda=0.0` for asl;
   - no Traceback.
   One smoke job runs on `amp20` to test whether 20 GB fits at 512 px with batch 8.
3. Submit the full set: A ×6 + B ×6 + B′ ×6 + C rerun ×2 = **20 runs** (about 10 h each, about 200 GPU-h), plus the 20 eval chains and the 3 missing C evals.
4. A background waiter on all job IDs, then `sacct` triage (`Done. Best`, `Early stopping`, OOM).

## Step 3 — analysis: `analysis/entropy_ablation.py` (new)
Reuse:
- `analysis/per_class_report.py`: `_ap_per_class`, `_group_means`, `_bootstrap_group_ci`;
- `evaluate/class_groups.py`: `build()`, giving head 3 / medium 20 / tail 7 classes by train prevalence >10% / 1-10% / <1%;
- the fast paired-bootstrap metric from `analysis/moe_compare.py`.

Asserts:
- every `preds.*.npz` has identical labels;
- every run's recomputed mAP equals `history.json` `best_metric` within 1e-4;
- group values match `eval_result.*.json`.

**Per run:** mAP, mAUC, mF1@0.5 and mECE (torchmetrics, as in training); per-class AP; head/medium/tail AP; best epoch; anchors/step; triplets/step; hit rate.

**Per arm:** mean ± s.d. over the 6 seeds.

**Pre-registered contrasts**, paired by seed: mean Δ, 95% t-CI (df 5), exact two-sided sign-flip p (64 patterns) and wins out of 6.
- **Primary: C − B on mAP.** This is the entropy effect.
- Secondary: C − B′ (which anchors vs how many); B − A (triplet + bank); C − A (full method); the same contrasts on head/medium/tail AP, mAUC, mF1 and mECE.

**Image-level paired bootstrap** (2,000 resamples) of the 6-seed probability-average ensemble per arm, for Δ mAP and Δ group AP. It shows validation-sample uncertainty alongside the seed variance.

Reproducibility: C rerun − C original for seeds 86 and 1024.

Secondary subset:
- if the row order equals the reconstructed val key order (label equality with `analysis/out/probs_v5_orig_recheck_flip/labels.npy`), add the primary contrast on frontal views;
- reading Projection for val-fold images only.

Outputs: `analysis/out/entropy_ablation.json`, and `analysis/out/entropy_ablation_table.tex`, which is the only source of numbers for the paper.

## Step 4 — reporting (cxrlt-research-analyst skill)
Delegate to the `cxrlt-research-analyst` subagent, with absolute run paths, log names and the JSON. It writes:
- results to `docs/result/<date>-entropy-ablation-512.md`;
- findings and a claim check (what the paper may and may not say) to `docs/plan/<date>-entropy-ablation-findings.md`.

Then:
- give the user the headline and 2-3 lines;
- update memory: `stage2-map-improvement-campaign`, replacing "gate not ablated" with the result; `paper-revision-confounds`, noting `isbi_submit`.

**Decision point:** the framing follows the primary contrast.
- If C − B is significant, the claim is "entropy gating helps".
- If not, the paper reports it honestly. Possible framings are equal accuracy with fewer anchors, or the gain being due to the triplet/bank and not to entropy. The framing is agreed with the user before the results section is written.

## Step 5 — paper: `docs/paper/isbi_submit/` (new; `docs/paper/isbi/` stays untouched)

**Template**
- Use the official ISBI kit that the 2027 author page links: `biomedicalimaging.org/2025/wp-content/uploads/sites/106/2024/07/ISBI_template-master.zip` (reachable, 256 KB).
- Download it into a fresh directory `isbi_submit/template_src/`, inspect the `.sty`/`.bst`, and copy only what the build needs.

**Scope: entropy only**
- No Markov, no MoE experiments, no original-implementation (pre-fix) numbers.
- The ISBI 2026 MoE entry (b12, test mAP 0.4599, 3rd place) is cited in the Introduction as context only.

**Layout:** pages 1-4 hold all technical content. Page 5 holds only references, "Compliance with Ethical Standards" and acknowledgments. Draft structure:
1. Abstract (~150 words) and Index Terms.
2. Introduction (~½ column): long tail and co-occurrence; the gap; 3 contributions; the controlled ablation is the core one.
3. Method (~1 column): Stage-1 model; cross-batch memory bank; entropy-gated anchors (Eqs.); mining and loss. Fig. 1 is a compact TikZ overview drawn from the code. The Entropy.pdf p.6/p.7 figures contradict the method and are not used.
4. Setup (~½ column):
   - data: CXR-LT 2026 Task 1, 30 classes; image-level internal split of 17,270 validation images that is not patient-disjoint (disclosed);
   - the fixed recipe, the arms and the metrics;
   - head/medium/tail definition and validation positives (tail classes have only 7-140 positives each);
   - the statistics.
5. Results (~1¼ columns):
   - **Table 1**: A / B / (B′) / C, mean ± s.d. of mAP, mAUC, mF1, mECE and head/medium/tail AP;
   - paired Δ with CIs, either in the table or in the text;
   - optional Fig. 2: per-class ΔAP for C − B, as TikZ bars;
   - anchors/step;
   - one scaling sentence: 768/1024 px single model 0.460/0.462 and greedy ensemble 0.472. These are internal validation, stated as not ablated, and from the full recipe (SWA, drop-path, flip).
6. Limitations (internal validation, patient overlap, softmax entropy over independent labels, easy mining, one dataset) and Conclusion.
7. Compliance with Ethical Standards; Acknowledgments (funding and conflicts of interest, plus the **mandatory AI-use disclosure**, worded for the user to confirm); References.

**Method facts to state exactly as implemented** (verify each against `train_2_v3.py` / `utils_update.py` while writing):
- softmax entropy over the 30 logits;
- gate H ≥ batch mean;
- positives share ≥1 label outside the 5 most frequent labels, and negatives share none;
- easy mining: closest positive, farthest negative;
- ℓ2-normalised mean of the decoded query embeddings;
- 2,048-slot FIFO bank of detached embeddings;
- λ 0.1, margin 0.5, triplet loss averaged over valid triplets;
- ASL γ−2 γ+0 with inverse-log-frequency class weights;
- AdamW 1e-4, 5% warm-up, cosine over 8 epochs, EMA 0.999 evaluated, patience 3;
- effective batch 48; 512 px; Stage 1 is ConvNeXt (5 stages) + ML-Decoder with ASL at 384 px.

**Rules:**
- every number comes from `entropy_ablation.json` or a cited `docs/result` report;
- a `\TBD` macro marks gaps, and the build check fails while any remains;
- no leaderboard claim for this method.

**Authors and title:** as in `docs/paper/isbi/paper.tex` (single-blind allows names).

## Files
- **Modified:**
  - `train/utils_update.py`;
  - `train/train_2_v3.py`.
  Both changes are backward compatible. Hold the MoE jobs **before** editing, then run `python -c "import utils_update"` right after the edit, because v5 imports the same module.
- **New:**
  - `scripts/submit_stage2_v3_entropy_ablation.sh`
  - `analysis/entropy_ablation.py`
  - `docs/paper/isbi_submit/{paper.tex, figures/, template files}`
  - `docs/result/<date>-entropy-ablation-512.md`
  - `docs/plan/<date>-entropy-ablation-findings.md`
  - the two plan copies from Step 0
- **Reused unchanged:**
  - `scripts/eval_perclass.sh` + `evaluate/evaluate.py`
  - `evaluate/class_groups.py`
  - `analysis/per_class_report.py`
  - the bootstrap code in `analysis/moe_compare.py`

## Timeline (deadline Mon 26 Oct)
| When | What |
|---|---|
| Oct 8-9 | Steps 0-2: code, unit check, smoke tests, submit |
| Oct 9-14 | Training, 2-4 waves depending on contention. Meanwhile, paper skeleton: template, method, setup, Fig. 1 (no result numbers) |
| ~Oct 14-16 | Step 3 analysis, Step 4 report, framing decision with the user |
| by Oct 20 | Full draft built and checked. Then a buffer for supervisor edits until submission |

If GPUs are scarce, the paper can go out with ≥3 seeds per arm, stated as such. The remaining seeds keep running.

## Verification
- Unit check and smoke checks (Step 2).
- **Regression:**
  - default `--anchor-rule mean` gives a v3 config identical to job 47097's apart from the run arguments;
  - `eval_perclass` on `seedsweep/seed_42` reproduces 0.4414 (history).
- **Analysis asserts** (Step 3), and per-arm means cross-checked against `history.json` by hand for 2 runs.
- **Paper:**
  - `latexmk -pdf` builds with 0 undefined references and no `\TBD`;
  - page images rendered with PyMuPDF (`python3 -c "import pymupdf"` works; no pdftoppm) and inspected, so that technical content ends on page 4 and page 5 holds only references, ethics and acknowledgments;
  - fonts embedded (PyMuPDF font list);
  - every number in the paper traced to its JSON or report line.
