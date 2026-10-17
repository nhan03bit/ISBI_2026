# Debug report: why Triplet Memory + Entropy trains to the same model as ASL only (train_2_v5.py), 2026-10-15

**Plan:** [2026-10-15-triplet-memory-debug-plan.md](2026-10-15-triplet-memory-debug-plan.md)
**Run metrics:** [docs/result/2026-10-15-seeded-asl-entropy-768.md](../result/2026-10-15-seeded-asl-entropy-768.md)
**Tools:**
- `analysis/triplet_probe.py` (probe 1, job 52831)
- `analysis/triplet_probe2.py` (probe 2, job 52833)
- `analysis/triplet_effect_offline.py` (CPU)
- `analysis/triplet_probe3.py` (probe 3, jobs 52889 and 52894; why λ 1.0 is worse)
- launcher `scripts/submit_triplet_probe.sh` (`MODE=2` for probe 2, `SMOKE=1` for 20-batch checks)

**Outputs:** `analysis/out/triplet_probe/`, `analysis/out/triplet_probe2/`.

No training code was changed. The probes replay the v5 training step on fixed checkpoints without optimizer steps:
- 768 px, batch 4, seeded train stream
- train mode (drop-path on)
- 512 warm-up batches to fill the 2048-entry bank
- `--metric-embed query`, `--head-classes 5`, `--anchor-rule mean`, λ = 0.1 unless stated

## Symptom
- **Observed:**
  - ASL only (`train_2_v0.py`) and Triplet Memory + Entropy (`train_2_v5.py`, λ 0.1) give the same mAP: seeded seed 42 flip-TTA 0.4605 vs 0.4625; 5 unseeded seeds 0.4617 vs 0.4612.
  - Only mECE differs consistently, with Entropy worse by +0.001 to +0.003.
- **Expected:** the triplet / cross-batch-memory term changes the learned model, as the paper claims.
- **Reproducibility:** always (5/5 unseeded seeds, plus the seeded pair).

## Root cause
- **Location:**
  - mining rule: `train/utils_update.py:121-122` (`_mine_pair`, `mining="easy"`)
  - default: `train/train_2_v5.py:1309`
  - applied in `sample_triplets_v13`, `utils_update.py:306-310`
- **Cause:** with easy mining (closest positive, farthest negative in a full 2048-entry bank), the hinge `d(a,p) − d(a,n) + 0.5` is already ≤ 0 for 98.8–99.8% of mined triplets. `TripletMarginLoss` is therefore ~0 and so is its gradient.
- **Mechanism:**
  - The term is not switched off, and nothing drops ASL. It is a self-extinguishing hinge on the easiest triplet.
  - The Stage-1 weights already satisfy it.
  - The little gradient it does produce is orthogonal to the classification gradient, and almost entirely orthogonal to the classifier readout. Its effect on predictions is label-agnostic and no larger than data-order noise.

## Evidence chain
1. **The path executes** (`history.json` `triplet_stats`, probe 1). About 80% of steps mine at least one triplet, about 1.1–1.3 per step. Gradient reaches the decoder and backbone through `last_h`, and the classifier logits are built from that same `h` (`ml_decoder.py:398-408`). ASL is computed on those logits (`train_2_v5.py:963-964`).
   - The memory bank reads back exactly: 0 H4 failures in 900 batches.
   - fp16 autocast and fp32 triplet gradients agree (cosine 0.99995–0.99998, norm ratio 0.997–1.001).
   - **Not a wiring, precision or ASL-drop bug.**
2. **The easy hinge is almost never active.** Probe 1, 300 batches, full bank:

   | Checkpoint | Easy L_tri / step | Active hinges | Easy-mined d(a,n) − d(a,p) | Semihard L_tri | Semihard active |
   |---|---|---|---|---|---|
   | Stage-1 init (start of Stage 2) | 0.0001 | 0.2% | 0.800 | 0.377 | 100% |
   | Seeded Entropy c (job 52650, EMA best) | 0.0001 | 1.2% | 0.670 | 0.384 | 100% |
   | Seeded ASL (job 52648, never trained with triplet) | 0.029 | 69% | 0.472 | 0.400 | 100% |

   - The margin is 0.5, so a gap of 0.67–0.80 between the easiest negative and positive means the hinge is slack.
   - Weights trained by ASL alone drift into violating it (69% active).
   - Entropy training keeps it at about 1%, so the term was active enough to hold a constraint that costs ASL almost nothing.
3. **Its gradient is about 2–10% of ASL's** (probe 2, 32 windows × 12 micro-batches, easy mining):

   | Measure | Stage-1 | c |
   |---|---|---|
   | λ‖∇tri‖ / ‖∇cls‖ per micro-batch | 0.002 | 0.019 |
   | λ‖∇tri‖ / ‖∇cls‖ per optimizer window (sum over 12) | 0.005 | 0.052 |
   | λ‖∇tri‖ / ‖∇cls‖ over all 32 windows | 0.007 | 0.098 |
   | cos(∇tri, ∇cls), window / all windows | 0.005 / 0.29 | 0.002 / −0.016 |
   | Windows with any triplet gradient | 1 of 32 | about 6 of 32 |

   - The "windows with gradient" row is inferred: the mean window coherence of the triplet gradient was 1/32 (Stage-1) and 6/32 (c), and each window with a single active micro-batch has coherence exactly 1. That matches about 1% active hinges × about 1.1 triplets × 384 micro-batches ≈ 5 active triplets in total.
   - **The triplet gradient is sparse, not small-and-dense.** Across 384 micro-batches only a handful of triplets carry any gradient. The ratio grows when summed only because ASL gradients cancel across windows (coherence 0.18, near the random 0.18).
   - **Gradient accumulation is not what damps the term.** It cancels ASL more than the triplet.
   - The triplet gradient is uncorrelated with the classification gradient, so to first order it does not change the ASL loss.
4. **The classifier readout ignores the triplet's push on `h`.** `logit_c = h_c · w_c + b_c`. The share of ∂L/∂h that lies along the readout vectors `w_c`:
   - triplet: 0.0016 (c), 0.0002 (Stage-1)
   - random direction: 0.036
   - ASL: 1.0 by construction

   The triplet's direct push on the logits is 0.02% of ASL's (c). This is because the triplet acts on `mean_c h_c`, so every query receives the same gradient `g/30`, and `g` is almost orthogonal to all 30 class readout vectors.
5. **Its indirect effect is through the shared backbone and is label-agnostic.** These are function-space derivatives on 64 val images, in eval mode, fp32, at the same step norm:
   - Along the accumulated triplet direction, logits move 1.54 (c) / 2.97 (Stage-1) rms per unit step.
   - Along the ASL direction they move 4.32 / 4.74.
   - Along a random direction they move 0.005.
   - So a triplet step does change predictions. But its size is 2–10% of an ASL step (row 3) and it is uncorrelated with the loss (cosine about 0). It acts as noise.
6. **The trained models differ only by noise.** Offline, SWA flip-TTA dumps, internal val:

   | Pair | Logit correlation | Mean \|Δlogit\| |
   |---|---|---|
   | ASL vs Entropy, same seeded stream | 0.983 | 0.178 |
   | ASL seeded vs unseeded (data order only) | 0.970 | 0.183 |
   | ASL seed 42 vs 123 | 0.970 | 0.183 |

   Weight distance relative to ‖ASL − Stage-1‖ (49.5):
   - 0.57 for λ 0.1, 0.95 for λ 1.0
   - against 1.07 for a data-order change

   The λ-dependence shows the term moves weights, but how much of the 0.57 is GPU non-determinism amplified over 100k steps is unknown, because no repeated seeded run exists. The classifier weights move least (0.26). The class biases move more than under data order (1.17 against 0.25), but their norm is about 0.005, too small to matter.
7. **Embedding geometry is not the cause** (probe 2 M5).
   - The bank's 0.87 mean cosine is a shared, dataset-wide component (norm 26 of 28). Removing the dataset mean gives a cosine of about 0.
   - The frozen query vectors contribute only cos 0.21–0.36 of that component.
   - Image variance of the mean-pooled embedding is 0.41–0.48 of the per-query variance, against 1/30 if the queries were independent. Mean-pooling does not wash out the image signal.
   - So the hinge is slack because easy mining picks a trivially satisfied pair, not because the embeddings collapsed.
8. **Strengthening the term confirms it.** Probe p (λ 1.0, job 52655), flip-TTA:
   - mAP 0.4557 (−0.0068 vs c)
   - mECE 0.1542 (+0.0186 vs c)

   A 10× weight on a mostly-zero hinge whose rare gradient is label-orthogonal does not add accuracy. Why it actively hurts (AdamW second-moment inflation) is in the probe 3 follow-up below.

**Therefore:** the near-identity ASL ≈ Entropy is explained by easy mining (magnitude), compounded by a triplet target that is orthogonal to the classifier readout (direction). The entropy gate chooses anchors for a loss that is almost always zero.

## Follow-up: why λ = 1.0 is worse (probe 3, jobs 52889 and 52894)
**Question:** if the triplet gradient is near zero at the trained checkpoints, why does λ 1.0 (arm p) lose mAP (0.4557 vs 0.4625) and calibration (mECE 0.1542 vs 0.1356)?

**Answer:** the hinge is rarely active, but each active triplet is large. At λ 1.0 its gradient is 7–10× the ASL gradient of the same micro-batch. These rare spikes inflate AdamW's second moment `v` (β2 0.999, a memory of about 1000 steps), so every later ASL update gets a larger denominator and moves less. The λ 1.0 model is ASL trained more slowly, not ASL trained toward a different target.

**Tools:** `analysis/triplet_probe3.py`, launcher `MODE=3` (`P3_ARGS` / `P3_TAG` to override). Outputs are in `analysis/out/triplet_probe3/` (constant LR 1e-4, 150 steps, λ 0 / 0.1 / 1) and `analysis/out/triplet_probe3_warm/` (the real warm-up of 860 steps, 600 steps, λ 0 / 1).

The short trainings in D start from Stage-1 and copy `train_one_epoch`:
- AdamW, wd 1e-2, GradScaler, accumulation 12
- the bank starts empty and fills during training
- easy mining, anchor-rule mean
- the same seeded stream for every λ
- no EMA

### Evidence
1. **Weight space, full runs (CPU, SWA weights).** I measured each run's distance from Stage-1, and its projection on the leave-one-out consensus of the unseeded ASL runs (always 3 runs per consensus).

   | Runs | ‖θ − θ_Stage-1‖ | Projection on the ASL consensus |
   |---|---|---|
   | 6 ASL | 49.40–49.57 | 0.668–0.684 |
   | 6 λ 0.1 | 49.10–49.19, every one below every ASL run | 0.666–0.676 |
   | λ 1.0 | **41.85** | **0.466** |

   - The λ 1.0 shortfall sits in backbone stages 2–3 (30.96 / 19.55 vs 36.96 / 25.16 for ASL). Head drift is unchanged (14.85 vs 14.80).
   - The size of the effect scales with λ: −0.7% of drift at λ 0.1, −15% at λ 1.0.
2. **AdamW `v` is inflated (probe 3 with warm-up).** The table gives rms √v̂ for λ 1.0 divided by the same quantity for λ 0, at the same step.

   | Step | Stages 0–1 | Stages 2–3 | Head |
   |---|---|---|---|
   | 20 (bank filling, LR 2e-6) | 4.5× | 5.0× | 1.9× |
   | 100 | 2.8× | 3.6× | 1.5× |
   | 300 | 2.3× | 3.1× | 1.4× |
   | 600 | 1.8× | 2.5× | 1.4× |

   - The early burst comes while the bank is still filling: about 27% of hinges are active in both arms at that point.
   - LR warm-up does not protect against this, because `v` accumulates g² whatever the LR is.
   - Later bursts at λ 1.0 (steps 200–400, 3–6% active) keep `v` high.
   - The inflation is largest and lasts longest in stages 2–3, the same place the full λ 1.0 run is short.
3. **Smaller steps, shorter drift.**
   - The per-step update norm at λ 1.0 is 0.6–0.9× that of λ 0.
   - By step 600, drift is 6.94 vs 8.94 (−22%). The constant-LR probe gives −15% at step 150, against −15% for the full run.
   - Cosine with the λ 0 drift is 0.5–0.8, and the projection is about 0.5.
   - The λ 0.1 arm (constant LR) follows λ 0 at cosine 0.87, with 2% less drift.
4. **The triplet does not pull the weights back toward Stage-1.** Per micro-batch, cos(−∇L_tri, θ − θ_Stage-1) is between −0.001 and +0.02, no larger than the same measure for ASL (±0.006); the random reference is 9e-5. Averaged over the instrumented micro-batches, cos(∇L_tri, ∇L_cls) is +0.02 to +0.05. Single micro-batches range from −0.5 to +0.45.
5. **Spike size.** λ 1-equivalent ‖∇L_tri‖/‖∇L_cls‖ on micro-batches with an active hinge:
   - training arms: mean 7.4–10.5 (n = 78–149)
   - the trained ASL checkpoint: 6.5 (n = 16)

   At λ 0.1 a spike is about ASL-sized. At λ 1.0 it dominates its 12-micro-batch window.
6. **Predictions behave like a less-trained ASL model.** On the seed-42 flip-TTA validation predictions:
   - **Global over-confidence.** All three models over-predict, and λ 1.0 more so. ECE is worse in all 30 classes. Mean probability on negatives is 0.179 for λ 1.0, against 0.160 (c) and 0.157 (ASL), and the positive-negative logit gap is 3.33 against 3.75 / 3.72.
   - **Behind on the same curve.** ASL's own mECE falls with training (0.151 → 0.121 over 4 epochs), and λ 1.0 tracks ASL about 1–2 epochs behind in mAP, val loss and mECE (`history.json`).
   - **Logit decomposition** (probe 3 A, 256 val images, SWA). The readout bias is about 0 in all models, so the shift is in h·w. λ 1.0's mean negative logit is +0.21 higher than ASL's, and +0.19 of that is the dataset-shared part of h_c·w_c. The image-specific spread is 0.94 vs 1.05.
   - **Embedding.** ‖mean h‖ is 17.8 for λ 1.0, against 30.6 for ASL and 27.0 for c; pairwise cosine is 0.77 vs 0.91.

### Correction
On 2026-10-15 I told the user that the triplet term is "a restoring force that pushes the weights back toward Stage-1". Evidence 4 contradicts that. The brake runs through the optimizer's step-size normalisation, not through the direction of the gradient.

### Implications
- **The λ 1.0 probe is not evidence about metric learning.** It measures an optimizer side effect of rare, large hinge gradients sharing AdamW state with ASL. It should not be reported as "more triplet hurts".
- **Raising λ will not make the term matter.** Any configuration with sparse but large triplet gradients will slow ASL. If the term is redesigned (see the recommended fix), keep the spikes away from ASL's step size. Options are to normalise or clip the per-step triplet gradient to at most the ASL gradient norm, or to give the triplet path its own optimizer state. A dense loss (semihard, class-conditional) would also make `v` steady rather than spiky.
- **The same mechanism explains part of the λ 0.1 picture**, and why seeded c is about 0.7% shorter than ASL. It is too small to move mAP.

## Hypothesis log
| # | Hypothesis | Test | Result |
|---|---|---|---|
| H0 | Memory path never executes / ASL dropped after the embedding layer | `triplet_stats`; code path `convnext.py:475-479`, `ml_decoder.py:398-408`, `train_2_v5.py:943-964`; probe L_cls | **Eliminated:** both losses run on the same `h` |
| H1 | Easy-mined hinge inactive → L_tri ≈ 0 | Probe 1, full bank | **Confirmed:** 0.2% / 1.2% active |
| H2 | Triplet gradient negligible vs ASL | Probe 2 M1/M2 | **Confirmed:** 0.2–10% depending on aggregation; cosine about 0 |
| H3 | Embeddings collapsed, so mining is degenerate | Probe 2 M5 | **Eliminated:** shared component is dataset-wide; residual geometry is healthy |
| H4 | Memory bank mis-populated or stale | Read-back check, every batch | **Eliminated:** 0 failures |
| H5 | Softmax-vs-binary entropy gate | Anchor Jaccard | **Refined:** 0.63–0.69 overlap. Changes which anchors are chosen, not whether the loss is live; it is a paper-vs-code confound, not the cause |
| H6 | Gradient accumulation cancels triplet gradients | Probe 2 M1 | **Eliminated:** the triplet gradient is sparse (a handful of active triplets per 384 micro-batches); accumulation cancels ASL more |
| H7 | fp16 underflow kills the small triplet gradient | Probe 2 M6 | **Eliminated:** cosine 0.99995 |
| H8 | Classifier readout is orthogonal to the triplet's push on `h` | Probe 2 M3/M4 | **Confirmed:** 0.16% readout share (random 3.6%); indirect effect is label-agnostic |
| H9 | λ 1.0 hurts because the triplet is a restoring force pulling the weights back toward Stage-1 | Probe 3: cos(−∇L_tri, θ − θ_Stage-1) per active micro-batch | **Eliminated:** −0.001 to +0.02, the same as ASL |
| H10 | λ 1.0 hurts because rare large triplet spikes inflate AdamW `v` and slow ASL | Probe 3 with warm-up: √v̂ per group; update and drift norms; full-run drift | **Confirmed:** √v̂ 1.4–5× (largest in stages 2–3); drift −22% at 600 steps, −15% in the full run (stages 2–3: −16% / −22%) |
| H11 | The calibration loss at λ 1.0 is a separate mechanism | Per-class signed ECE; logit decomposition; per-epoch history | **Eliminated:** a global, all-class over-confidence shift in the shared h·w part, matching ASL earlier in training |

**Superseded intermediate claims:**
- The 2026-10-15 smoke runs (20 batches, 80-entry bank) suggested active hinges and a large gradient ratio (0.19–1.1). They were an artefact of the small bank: fewer candidates means the closest positive is farther and the farthest negative closer.
- The earlier train-loss-gap estimate (L_tri ≈ 0.003 at λ 0.1) compared two different runs. It is the right order of magnitude for live (non-EMA, stale-bank) weights, but it is not a measurement.

## Recommended fix (for decision; nothing launched)
1. **Report the current method honestly.** As implemented (easy mining, 2048 bank, λ 0.1), Triplet Memory is close to inactive, and the entropy gate selects anchors for an almost-always-zero hinge. Any paper sentence attributing a mechanism or gain to it is unsupported. Check the abstract, method and discussion against this; see the memory note on paper honesty rules.
2. **To make the term matter, two changes are needed, not one:**
   - **Magnitude:** `--mining semihard` (flag already exists, `train_2_v5.py:1309`). This gives 100% active hinges and λ‖∇tri‖/‖∇cls‖ ≈ 0.7–1.3 at λ 0.1 (probe 1).
   - **Direction:** apply the metric loss to what the classifier reads. Use the per-class query embeddings `h_c` of the anchor's positive (non-head) classes, compared with the same classes in the memory, instead of `mean_c h_c`. This requires storing (B, 30, D) or per-class entries in `CrossBatchMemoryV2` and a class-conditional sampler. It is a code change.
   - Semihard alone is cheap to test (one about 12 h run). Probe 1 shows its gradient is still near-orthogonal to ASL (cosine 0.09–0.15), so a gain is not expected without the direction change.
3. **Seeded noise floor:** one repeat of the seeded ASL run (about 12 h) would show how much of the 0.57 weight distance and 0.983 logit correlation is GPU non-determinism. That is needed before attributing any small seeded difference to the triplet.

**Affected files** (if option 2 is chosen): `train/utils_update.py` (memory and sampler), `train/train_2_v5.py` (embedding selection), plus a parity smoke test. **Risk:** medium. It changes the method the paper describes, so it is a method change, not a bug fix.

4. **If λ is raised again**, or any triplet variant is trained alongside ASL, first bound the triplet gradient: clip its per-step norm to at most the ASL gradient norm, or give it separate optimizer state. Otherwise the result measures AdamW step-size damping, not the loss (H10).

## Regression prevention
- When probing a new triplet configuration, also log AdamW's rms √v̂ per group against an ASL-only run (`analysis/triplet_probe3.py --parts D`). A ratio well above 1 means the term is slowing ASL rather than shaping the embedding.
- Log the triplet loss and active-hinge fraction per epoch into `history["triplet_stats"]`, next to `hit_steps` and `triplets`. A future run then shows directly whether the term is live, instead of having it inferred from the train-loss gap.
- Keep `analysis/triplet_probe.py` as a check on any new triplet configuration. Healthy means an active-hinge fraction well above 1%, and a λ-ratio not near 0.
