# How could Triplet Memory + Entropy Gate beat the ASL baseline? Plan, 2026-10-16

**Question (user).** The ASL baseline is a proven 0.46 mAP. Using `train/train_2_v5.py`, how can Triplet Memory + Entropy Gate beat it?

**Short answer.**
- **As implemented, it cannot.** Raising λ, harder mining on the pooled embedding, or re-tuning the gate will not change that (section 1).
- **Two routes could work:**
  - **Route 1, memory readout at inference.** Use the memory as a k-nearest-neighbour readout of training-set embeddings, blended with the classifier where its entropy is high. Cheap and testable without training. The probe is running: jobs 53157 / 53158, about 2–3 h on amp16.
  - **Route 2, class-conditional memory loss in training.** Make the training-time memory act on the per-class embeddings the classifier reads, with a dense loss and a bounded gradient. Needs code and about 12 h training runs on the congested amp48 / ada24 queue.
- **Both are method changes.** The paper must describe the method that is run and report every variant tried (section 4).

## 0. What "beat" has to mean
- **Same eval filter.** Since 2026-10-16, evals use `INTER_AREA`, which lifts every arm.
  - **ASL under area: 0.4651 ± 0.0030 over the 5 Table 1 seeds** (frontal 0.4899), 0.4660 seeded seed 42 (`analysis/eval_interp_table.py`, 2026-10-16). Entropy under area is 0.4652, so Entropy − ASL is +0.0001 [−0.0042, +0.0045].
  - Comparing a new method under area against the lanczos 0.4617 would show a fake +0.003 to +0.005.
- **Same recipe and a paired comparison over ≥ 5 seeds.** The smallest detectable paired 5-seed difference is about 0.005 (all views). Tail claims need ≥ 0.02–0.03 (`2026-10-16-tail-map-breakdown.md`, section 5).
- **Report all views, frontal (the leaderboard is frontal-only) and tail,** each with the seed and the image interval.

## 1. Why the current method cannot beat ASL
From `2026-10-15-triplet-memory-debug.md` (probes 1–3), each with the change that would be needed:

| Mechanism | Evidence | Consequence |
|---|---|---|
| Easy mining (closest positive, farthest negative in a 2048 bank) | Hinge active 0.2% (Stage-1) / 1.2% (trained) of mined triplets | L_tri ≈ 0, so there is almost no gradient |
| Metric embedding = `normalize(mean_c h_c)` | Its push on `h` lies 0.16% along the readout vectors `w_c` (random direction: 3.6%) | Even when active it barely reaches the logits `h_c · w_c` |
| Rare, large hinge gradients share AdamW state with ASL | √v̂ inflated 1.4–5×; drift −15% at λ 1.0 | Raising λ slows ASL instead of adding information |
| Gate = softmax entropy over 30 multi-label logits ≥ batch mean | Mostly separates "Normal-only" images from the rest; 0.63–0.69 overlap with a binary-entropy gate | It selects anchors for a loss that is almost always zero |

**Ruled out as fixes:**
- **Raising λ** gives the optimizer artefact.
- **Hard or semihard mining on the pooled embedding** gives 100% active hinges, but cosine with the ASL gradient is still 0.09–0.15 and the direction is wrong.
- **Re-tuning the gate threshold** changes which anchors are picked, not whether the loss is live.

## 2. Route 1: entropy-gated memory readout at inference (probe running)
**Idea.**
- **Memory:** the training-set per-class query embeddings `h_c`.
- **Readout:** for a test image, a k-nearest-neighbour vote over the memory, `p_knn[c] = Σ_j softmax(s_j/τ) y_j[c]`.
- **Gate:** `p_knn` is blended into the classifier where the classifier is uncertain. "Memory" and "entropy gate" then do real work at test time.
- **Role of triplet training:** it becomes "make the memory retrieve better". That is testable by comparing memories built from the ASL and Entropy checkpoints.

**Probe** (`analysis/memory_readout_probe.py`, `scripts/memory_probe.sh`):
- Seeded ASL (job 53157) and seeded Entropy c (job 53158), seed 42, same data stream.
- One no-TTA area pass over all 103,305 train and 17,274 val images.
- Per-class `h_c` neighbours and pooled-embedding neighbours (K = 100), plus prototype scores.
- **Same-patient train images are excluded from every neighbour list.** A same-patient neighbour leaks the study label: 77% of val images share a patient with train, and the leaderboard test set is assumed patient-disjoint.
- Smoke job 53156 passed. The extracted logits match the earlier area eval of the same checkpoint (max |Δp| 0.004, fp16).

**Evaluation** (`analysis/memory_readout_eval.py`, protocol fixed in its docstring before any result):
- **Gate families:**
  - none (constant α)
  - per image-class binary entropy
  - the training gate's rule: softmax entropy ≥ mean
  - a random gate with the same share of images, as a control
  - kNN only
  - a prototype-score blend
- **Grid:** K {10, 25, 50, 100} × τ {0.01, 0.03, 0.1} × α {0.1, 0.2, 0.3, 0.5} × embedding {per-class, pooled}.
- **2-fold cross-fitting by patient:** chosen on one half of the val patients, scored on the other; paired image bootstrap against the classifier alone.

**Decision rule** (fixed now):
1. **Memory adds information** if a cross-fitted blend beats the classifier with the 95% image interval above 0 on all-view mAP, for the ASL checkpoint.
2. **The entropy gate adds value** only if the binary or softmax gate beats both the constant-α blend and the random-gate control.
3. **Triplet training helps the memory** only if the Entropy checkpoint's kNN-only and best-blend mAP exceed ASL's by more than the seed noise. That needs more than one seed: one pair cannot separate it from data-order noise (same-seed reruns differ by about 0.005).
4. **If (1) holds,** a paper-grade claim needs the probe on all 5 seeds of A and C: 10 probes × about 2.5 h on amp16, about 2 days with 2 GPUs. Report it as "ASL + entropy-gated memory readout" against ASL, and attribute nothing to triplet training unless (3) holds.
5. **If (1) fails,** a memory of these embeddings carries no information beyond the classifier. Route 2 is then the only option, and its prior odds drop.

**Route 1 result, Entropy checkpoint only (job 53158, evaluated 2026-10-17 by `analysis/memory_readout_eval.py`, `analysis/out/memory_readout_eval.log`):**
- **Setup:** 17,270 val against 103,281 train images; no TTA, area. 77% of val images have a same-patient train image, and all of those were excluded from the neighbour lists. Cross-fitted by patient halves; 500 paired image resamples.
- **Classifier alone: 0.4641** (frontal 0.4894; tail 0.3443).

| Family | mAP change, 95% CI | Frontal | Tail | Config chosen (fold 0 / fold 1) |
|---|---|---|---|---|
| Binary-entropy gate per (image, class) | **+0.0021 [+0.0005, +0.0046]** | +0.0039 [+0.0001, +0.0102] | +0.0046 [−0.0016, +0.0154] | per-class h_c, K 25, τ 0.01, α 0.3 / same |
| Softmax-entropy gate (training rule) | +0.0011 [−0.0002, +0.0035] | +0.0032 [−0.0000, +0.0095] | +0.0047 | per-class, K 25 / K 50, τ 0.01, α 0.1 |
| Random gate, same share (control) | −0.0002 [−0.0020, +0.0020] | +0.0006 | +0.0013 | varies |
| No gate (constant α) | −0.0035 [−0.0076, −0.0004] | −0.0022 | −0.0041 | per-class K 25 / pooled K 100 (unstable) |
| kNN only | −0.0173 [−0.0231, −0.0125] | −0.0141 | −0.0355 | per-class, K 100 |
| Prototype blend | −0.0004 [−0.0014, +0.0015] | +0.0003 | +0.0009 | b 0.25 |

**Against the decision rules:**
1. **The memory carries a little information beyond the classifier, but only through the gate.** The binary-gated blend's interval is above 0 (+0.0021), while ungated blending hurts (−0.0035). This is for the Entropy checkpoint; the ASL control was cancelled by the user's decision, so whether the same holds for ASL features was not measured.
2. **The entropy gate is what makes the memory useful.** The binary gate beats the constant blend and the random-gate control by point estimate; a direct paired gate − random interval was not computed. Both halves of val chose the same config for it, and per-class `h_c` neighbours beat pooled ones. That supports Route 2's choice of the per-class embedding and the per-(image, class) Bernoulli-entropy gate.
3. **Triplet credit: not assessable** without the ASL probe.
4. **The effect is small** (+0.002 all views, +0.004 frontal, one seed) and is an inference add-on that would likely help ASL too. It is a diagnostic and a candidate inference component for a Route 2 model, not a claim that Triplet Memory training beats ASL.

## 2b. User decision, 2026-10-16
"No need to further improve the ASL baseline; focus on Entropy only (`train_2_v5.py`)."
- **ASL stays fixed** at 0.4651 (area, 5 seeds) / 0.4660 (seeded seed 42).
- **The ASL memory probe (job 53157) was cancelled.** It only measured "ASL + memory readout".
- **The Entropy probe (53158) continues** as a diagnostic of how class-structured the Entropy model's `h_c` already is. Its result is not a method claim: a readout added at test time would help ASL equally.
- **Route 2 is the focus.** It is implemented below (2026-10-16).

## 3. Route 2: class-conditional memory loss on `h_c` (implemented 2026-10-16)
**Code:**
- `train/class_memory.py`: new module. `utils_update.py` (CRLF) is untouched.
- `train/train_2_v5.py`: `--metric-loss nca` and the `--nca-*` flags. The default `--metric-loss triplet` keeps the old code path.
- Launcher arms in `scripts/submit_stage2_v5_unified.sh`: `n` (entropy gate), `na` (no gate), `nr` (random gate, same count), plus `EVAL_INTERPS`.
- `scripts/train_2_v5.sh` gained the CUDA fail-fast.

**Checks:**
- **Unit test (CPU):** the vectorised loss equals a naive per-pair computation (0.797644 vs 0.797644). Gradients are finite. The random gate matches the entropy gate's count, and the warm-up skip works.
- **GPU smoke and parity (job 53159, `scripts/smoke_nca.sh`, A4000, 512 px), PASSED:**
  - **Parity:** the pre-change copy `train/_parity_v5_orig.py` and the new script with default flags give identical train loss (0.05230460440119108) and bit-identical EMA weights. The old Entropy method is untouched. Keep `_parity_v5_orig.py` until `train_2_v5.py` is committed.
  - **`nca` run (120 micro-batches, warm-up lowered to 64 images):**
    - active on 104 / 120 micro-batches
    - about 59 of 120 (image, class) pairs gated per micro-batch, 4.4 of them positive
    - its gradient on `h` is 0.44× ASL's, so the cap rarely engaged (mean scale 0.997); the old triplet's was about 1%
    - every one of the 420 weight tensors differs from the default run, and drift from Stage-1 is 2.4× larger after 10 optimizer steps
- **Pilot submitted 2026-10-16 23:00:** job 53160 (training), evals 53161 (lanczos) and 53162 (area) chained afterok.
  - Output `train/checkpoint_Triplet_3/unified768_seeded/n/seed_42/Model_run`; dumps `analysis/out/probs_unified_flip_seeded/n__seed_42__model_swa@768_flip[_area].npy`.
  - Read with `python -I analysis/eval_interp_table.py`: seeded row `N` and the "N vs A / N vs C" blocks.

**Pilot (pre-registered before any result):**
- `OUT_TAG=seeded EVAL_INTERPS="lanczos area" ./scripts/submit_stage2_v5_unified.sh n:42`
- It is the seeded Entropy recipe of job 52650 exactly (768 px, bs 4 × accum 12, 4 of 8 cosine epochs, SWA top-3, drop-path 0.1, seed 42, seeded stream), with the memory loss replacing the triplet term.
- Hyperparameters are the flag defaults, fixed before any run: τ 0.1, bank 4096, queue 128, λ 1, gradient cap 1× ASL on `h`, warm-up 1024 images.
- **Comparisons (flip-TTA SWA, area eval):** seeded Entropy c 0.4676 and seeded ASL 0.4660.
- **Same-seed noise floor:** about 0.005 (seed-42 reruns of one recipe: 0.4793 vs 0.4740).

**Decision rule:**
- **Promising:** pilot − seeded ASL ≥ +0.005 on all-view area mAP, and mECE not worse than ASL by more than 0.005.
  - Then launch seeds 123 / 456 / 789 / 1024 of `n` and, GPU permitting, the `na` / `nr` controls, for the paired 5-seed test against ASL's area runs.
  - The gate claim needs `n` > `na` and `n` > `nr`.
- **Below +0.005:** report it as a null pilot. No re-tuning of τ / λ / gate on the internal val to chase a gain. Any further variant is reported as a separate, disclosed attempt.
- **Liveness, read from `history.json` `nca_stats` (the reason for the redesign):**
  - the term must be active on most micro-batches after warm-up
  - the gradient ratio and scale must show it acting at roughly ASL's magnitude, not ~1% of it

### Original design notes (written before implementation)
**Design.** Each item answers one row of the table in section 1:
- **Embedding:** the per-class `h_c` that `logit_c = h_c · w_c + b_c` reads, not `mean_c h_c`.
- **Memory:** a per-class FIFO bank `M` (S × 30 × D, fp16, S ≈ 4096) with labels. Cross-batch memory matters here because a batch of 4 rarely holds two positives of a tail class.
- **Loss:** dense supervised-contrastive, always active, so there are no spikes. For an anchor (image i, class c) with `y_ic = 1`:
  - positives: memory entries with `y_jc = 1`
  - negatives: the hardest 256 entries with `y_jc = 0`, at temperature 0.1
  - formula: `L_ic = −mean_p log softmax(s/τ)_p`
- **Entropy gate, per (image, class):** only anchors whose classifier output is uncertain, binary entropy `Hb(p_ic)` above a threshold. That is the multi-label form of the paper's idea and replaces the softmax-over-30 image gate.
- **Optimizer safety:** bound the term's gradient on `h` to at most the ASL gradient norm (`torch.autograd.grad` on `h` only, cheap), or give the memory loss its own optimizer state.
- **Defaults off.** Flags keep `train_2_v5.py` byte-identical to every existing run. A parity smoke test (default flags against the current code, 40 batches) comes first, then a probe-2-style check that the term is live: share of active steps, gradient ratio, readout share.

**Cost and timeline** (ISBI deadline Mon 26 Oct):
- **Code and smoke:** about 1 day, no GPU beyond the smoke test.
- **Pilot:** seeded seed 42 at 768 px, about 12 h of training. It needs amp48 / ada24 (an A4000 is too small at bs 4), and on 2026-10-16 about 60 jobs from higher-fairshare users were pending there.
  - A like-for-like alternative that fits amp16: both arms at bs 2 × accum 24 (same effective batch 48). It is not the Table 1 recipe, so it is only a screening pilot.
- **Paper-grade test:** 5 seeds for each arm. 2–3 days if GPUs free up, so it may not finish before 26 Oct.
- **Expected effect:** uncertain. Supervised-contrastive auxiliaries on strong multi-label baselines typically add 0.3–1.5 mAP points with full training, and here training is 2–4 epochs of fine-tuning. A pilot can only screen for effects of at least about 0.01 (n = 1, same-seed noise about 0.005).

**Pilot result (2026-10-17): null.** Job 53160, read with `analysis/eval_interp_table.py` (95% bootstrap over val images, single run):
- **Live as designed:** active on 25,566 / 25,822 micro-batches in epoch 1 and on every one after; about 61 anchors per step; gradient 0.46-0.55× ASL. The bank and queues filled (smallest queue 38 in epoch 1, 128 by epoch 4).
- **Worse than both baselines** (flip-TTA SWA, area): 0.4543 vs seeded ASL 0.4660, **−0.0117 [−0.0181, −0.0057]**; vs seeded Entropy 0.4676, −0.0133 [−0.0218, −0.0066]. Lanczos: 0.4506 (−0.0099 vs ASL). mECE 0.180 vs 0.132 (ASL), so it is over-confident.
- **The loss sits in the rare classes** it was meant to help (vs ASL, area): head −0.0006, medium −0.0062, tail **−0.032 [−0.060, −0.009]**, rare10 −0.026. Untested guess: a rare class's 128-entry positive queue holds a handful of images seen many times, so the soft-neighbour target memorises them.
- **Per the rule above:** no seeds 123-1024, no `na` / `nr`, no re-tuning of τ / λ / gate on the internal val.

## 2c. Route 1 replication on the 5 Table 1 Entropy seeds (pre-registered 2026-10-17, before any result)
User, 2026-10-17: "investigate further on the positive direction shown where Entropy + Triplet Memory Bank could perform better". Route 1 is the only positive so far, and it was one seeded checkpoint.
- **Probes:** jobs 53219-53223, `scripts/memory_probe.sh --tag entropy_c_s<S>` on `train/checkpoint_Triplet_3/unified768/c/seed_<S>/Model_run/model_swa.pth`, S = 42/123/456/789/1024 (the Table 1 Entropy runs). Settings identical to 53158: 768 px, area, no TTA, K 100, same-patient train images excluded.
- **Eval:** `analysis/memory_readout_eval.py` with its protocol unchanged (same grid, 2-fold patient cross-fit, 500 image resamples), then a per-seed summary.
- **Primary:** binary-entropy-gated blend − classifier alone, all-view mAP, no TTA. **Holds** only if the mean over the 5 seeds has a 95% t-interval over seeds above 0 **and** at least 4 of 5 seeds are positive.
- **The gate counts** only if, in at least 4 of 5 seeds, the binary gate beats both the random-gate control and the constant-α blend.
- **Reported, not decisive:** frontal mAP, tail mAP (resolvable only to about 0.02-0.03), and the configs each fold chose (stability).
- **Exploratory, labelled as such:** the fold-chosen config applied to the flip-TTA area probabilities of each seed, against ASL's flip-TTA area run of the same seed index (Table 1 pairing). ASL gets no readout here (no ASL probe), so this is "Entropy + readout vs plain ASL", not a fair method comparison.
- **What a positive result would and would not show:** it would show that an entropy-gated memory readout at inference improves the Entropy model. It would NOT show that Triplet Memory training made the memory better, nor that it beats "ASL + the same readout": that needs the 5 ASL probes (rule 3 above), which are not run under the 2026-10-16 user decision.
- **If it fails** (interval includes 0, or fewer than 4 of 5 seeds positive), the +0.0021 of job 53158 is reported as a single-seed result that did not replicate.

## 4. Integrity rules for either route
- The method in the paper is the method that was run, and the earlier null results (512 px ablation, 768 px Table 1) stay disclosed.
- Every family, grid and variant tried is reported, not only the best.
- Internal-val retrieval always excludes same-patient images. Tuning is cross-fitted, never done on the reported half.
- Same eval filter (area) for every arm, and the final claims are paired over ≥ 5 seeds.
- If neither route clears section 0's bar, the paper stays a null-result paper for the entropy gate.
