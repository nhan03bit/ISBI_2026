# Stage 3 research plan: reducing minority-label underlearning

Revised on **2026-10-08 (Europe/London)** after an independent methodological review.

**Status (2026-10-08):** Phase A implemented after the user authorized coding. Synthetic CPU checks are available; real-data university GPU experiments are pending. See [run instructions](stage3-running.md). Phases B-E remain proposals.

## 1. Objective and research claim

Improve the existing 30-label image model, especially labels with fewer than 1% positive training examples, without materially reducing performance on other labels.

The leader's proposed mechanism is a hypothesis:

> Rare labels may receive too few useful updates. More minority-positive exposure, class-specific memory partners, and bounded minority weighting may correct that shortage.

The experiments must discover which component, if any, improves held-out performance. More triplets, larger weights, or fuller queues are mechanism measurements, not success by themselves.

The fixed minority group is: Hydropneumothorax, pneumoperitoneo, Subcutaneous Emphysema, Mass, azygos lobe, Pneumothorax, and calcified densities. Membership is computed once from the training fold using prevalence strictly below 1%.

## 2. Why the previous plan is replaced

The previous plan changed sampling, representations, memory allocation, mining, entropy selection, and weighting together. A gain would be hard to explain and a failure would reveal little.

Its 250 updates equal about 12,000 image presentations at an effective batch of 48, only about 0.12 passes through 103,305 training images. That is useful for a systems check, but weak evidence about performance. Evaluating eight arms every 50 updates would also spend disproportionate computation on validation.

The revised design asks one causal question at a time and stops weak ideas early.

## 3. Ordered hypotheses

### H1: exposure shortage

Minority positives appear too rarely for stable learning. Patient-aware minority sampling should improve tail AP with classification loss alone.

**Falsifier:** sampled classification does not beat matched natural continued fine-tuning across repeated runs.

### H2: relevant-partner shortage

Ordinary FIFO memory may contain too few fresh, different-patient positive partners for rare labels. Per-label reserved queues should improve usable coverage.

**Falsifier:** ordinary FIFO already supplies partners, or reserved queues improve coverage without improving held-out ranking/AP.

### H3: ranking shortage

Classification alone may not sufficiently rank rare positives above negatives. A target-label comparison objective may improve that ordering.

**Falsifier:** training geometry or triplet counts change but held-out ordering and AP do not.

### H4: unequal minority difficulty

After exposure and partner access are controlled, some minority labels may still need more influence.

**Falsifier:** bounded weighting adds no repeatable gain or harms overall/head performance.

The order is deliberate. Sampling must first prove the premise. A useful auxiliary objective must exist before engineering special memory for it. Weighting comes last because sampling already changes effective class importance.

## 4. Lock the baseline and evaluation

Use the verified v3 768-pixel seed-42 SWA checkpoint first. Record its exact path, hash, architecture, label order, transforms, and saved validation predictions. Recompute per-label AP from aligned predictions and explain the four validation exclusions.

Before training, freeze an experiment manifest containing data/checkpoint hashes, label order, minority membership, resolution, transforms, trainable parameters, optimizer, schedule, AMP, EMA, random seed, image budget, evaluation points, and primary metrics.

Do not select methods on the official test set. The current validation has only 7 and 9 positives for the two rarest labels and substantial patient overlap with training. Report exact support and patient-cluster bootstrap intervals; do not claim reliable per-class improvement for those two labels from this fold alone.

## 5. Progressive experiments

### Phase A: test minority exposure with no metric-learning changes

Start every arm from the same checkpoint. Freeze the backbone; train the ML-Decoder and prediction readouts. Keep all 30 ASL outputs active.

| Arm | Sampling | Loss | Question |
|---|---|---|---|
| A0 | none | none | starting checkpoint reference |
| A1 | natural | existing ASL | effect of continued fine-tuning |
| A2 | minority-aware | existing ASL | does extra minority exposure help? |

For A2, logical batches still contain four images, but only half of logical batches force one minority-positive image. The average targeted fraction is therefore 12.5%, lower than the previous 25% proposal. Other images come from the natural stream.

Choose target labels with tempered probability proportional to the square root of positive-patient count, then cycle through positive patients before repeating a patient's images. This favors the tail without presenting each of the 38 rarest examples hundreds of times merely to equalize all seven class counts. Record actual per-label, per-patient, and per-sample exposure.

All images contribute all 30 labels to ASL. This remains one shared model, not seven models.

**Gate A:** continue only if A2 improves the predefined supported-tail aggregate over A1 while satisfying overall/head guards. If it fails, investigate repetition, label quality, gradients, and learning curves before adding memory.

### Phase B: test a target-specific objective with ordinary memory

If Phase A passes, use the existing per-label decoder feature `h[i,c]`; do not average all 30 class-query features.

| Arm | Auxiliary objective | Memory | Question |
|---|---|---|---|
| B1 | none | none | selected sampling baseline |
| B2 | class-specific triplet | ordinary 2,048 FIFO | test the leader's memory hypothesis |
| B3 | pairwise class-logit ranking | ordinary FIFO/candidate buffer | test a score-aligned alternative |

For B2, a positive anchor for target `c` uses a different-patient positive for `c` and a different-patient negative for `c`. Average several eligible comparisons when available; one random triplet is too noisy. Prefer semi-hard negatives and use random valid negatives as fallback. Explicitly targeted anchors do not use the existing softmax-entropy gate.

For B3:

```text
L_rank(c) = mean softplus(score_negative,c - score_positive,c)
```

This directly penalizes positive-negative ordering errors related to AP. It is a comparator, not a predetermined replacement for triplet learning. Match partner eligibility between B2 and B3 where possible.

Begin with a small fixed auxiliary coefficient. Log classification and auxiliary gradient norms and reject settings where the auxiliary term dominates the decoder update.

**Gate B:** advance an objective only if it improves AP/held-out ordering beyond B1 and changes its named mechanism. Extra valid triplets alone do not pass.

### Phase C: test reserved minority memory

Take the better Phase B objective and compare an ordinary FIFO against a capacity-matched bank with a general FIFO region and seven reserved positive regions.

Records contain detached normalized class-query features, target label, opaque sample identity, patient group, and insertion step. Exclude same-patient partners, deduplicate candidates, and expire stale entries. Choose capacity/expiry using measured occupancy and age. The former 1,600 general plus 7 x 64 reserved layout remains a candidate setting, not a theoretical truth.

**Gate C:** retain reserved memory only if it improves both valid-partner access and score over equal-capacity FIFO. Better occupancy without better performance is insufficient.

### Phase D: test boosting last

Compare the selected unweighted method (`w_c=1`) with:

```text
w_c = min(2, sqrt(0.01 / prevalence_c))
```

Do not simultaneously modify the existing ASL weights. Report combined sampling exposure and gradient contributions because oversampling and weighting multiply each other.

**Gate D:** keep boosting only if it adds repeatable value over the complete unweighted method without breaking overall/head guards.

### Phase E: adaptation-capacity check

Frozen-backbone experiments isolate the intervention and reduce overfitting, but may cap gains if rare visual cues were never learned. After selecting the best decoder-only configuration, compare it once with gradual unfreezing of the final backbone stage at a lower learning rate. Attribute any extra gain to added capacity separately.

## 6. Compute plan

Use three levels:

1. **Synthetic/CPU checks:** sampler, eligibility, memory, loss, resume, and metric correctness. No performance claim.
2. **GPU systems check:** 5-20 optimizer updates for shapes, memory, numerical stability, and throughput.
3. **Performance screen:** a fixed budget of approximately one training-set pass, about 2,152 optimizer updates at 48 image presentations per update. Extend only if learning curves justify it.

Screen with one declared seed per arm. Any claimed winner requires at least three matched seeds, reporting mean, spread, and paired differences. Evaluate at initialization, midpoint, and endpoint. The fixed-budget endpoint is primary; best-validation results are secondary and must use the same rule for every arm.

Complete Phase A before B, B before C, and C before D. This successive elimination is more compute-efficient than fully training every combination.

## 7. Measurements

Primary performance measures:

- overall 30-label macro AP;
- seven-label minority macro AP;
- per-label AP with positive image/patient counts;
- head and medium macro AP;
- paired patient-cluster bootstrap intervals;
- across-seed variation for finalists.

ROC-AUC is supplementary because it can remain high for imbalanced labels despite poor precision.

Mechanism measures:

- positive exposure and repeats per label, patient, and sample;
- memory occupancy, age, and patient diversity;
- eligible, selected, active, and zero-loss comparisons;
- audit of possible false negatives in target-negative partners;
- classification and auxiliary gradient norms;
- held-out positive-negative ordering rate and score distributions;
- calibration and predicted-positive prevalence.

## 8. Acceptance and stopping rules

Use provisional engineering guards relative to matched continued training:

- overall macro AP loss no greater than 0.002;
- head macro AP loss no greater than 0.005;
- reported improvement in supported minority AP with uncertainty and seed variation;
- observed change in the mechanism named by the hypothesis.

Stop or simplify if sampling does not beat natural continued training, auxiliary learning changes geometry but not ranking, reserved queues change coverage but not score, weighting merely trades performance among labels, or gains vanish across seeds/patient-disjoint evaluation.

These guards are research choices, not clinical tolerances.

## 9. Generalization and final evidence

The current fold is developmental. About 77% of validation images share a patient with training, and the unseen-patient subset has no positives for the two rarest labels.

After method selection:

1. repeat the winner and strongest simple baseline with matched seeds;
2. confirm on a patient-disjoint split with adequate minority support;
3. match resolution, initialization, image budget, augmentation, EMA, and inference;
4. test TTA, checkpoint averaging, and ensembles only after establishing the single-model contribution;
5. report negative findings and keep novelty conditional on an updated closest-work review.

## 10. Recommendation

Begin with **minority-aware classification fine-tuning**. It is the cheapest direct test of the leader's premise and the component most likely to improve score.

If it succeeds, compare the requested class-specific triplet method with pairwise class-logit ranking using ordinary FIFO memory. Add reserved queues only when partner scarcity is measured. Add boosting only after the unweighted method works.

This plan does not promise improvement. It provides a defensible path to find whether the premise is correct, identify the part responsible for any gain, and avoid spending university GPU time on unsupported complexity.
