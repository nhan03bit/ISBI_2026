# Project memory and handoff

Last updated: **2026-10-10, Europe/London**.

## Read this first

**Latest priority (2026-10-10):** user requested a new branch and a clean entropy ablation for a focused four-page ISBI paper. Branch `entropy-ablation` is based on `long-tailed`. The new experiment uses Stage 2 v3, NOT minority-sampled Stage 3: classification only -> ungated FIFO triplet -> identical triplet with the shipped entropy gate. All arms start from the same pre-triplet Stage 1 checkpoint, with matched 768 resolution, schedule and seeds. See [entropy ablation protocol](docs/research/entropy-ablation.md). Markov/history is outside the main contribution. No GPU results exist for this ablation yet.

Implementation validation: 11 local tests passed (8 training/data tests, 1 ablation-report test, 2 Stage 3 submission tests), plus Bash syntax and diff checks. The three-arm CPU integration test uses a tiny model and verifies fixed endpoints, identical initialization, memory activation and group AP. University full-model training remains untested. The ablation launcher and report are ready locally; no new-branch push or job submission has occurred.

We are researching improvements to long-tailed, 30-label chest X-ray classification. The user's research leader has now specified a **Stage 3 focused on minority-related memories and class dominance**. This is the current priority, superseding the assistant's earlier broad emphasis on Markov/history extensions.

The user wants explanations that build from plain language to technical detail. They are developing their understanding of the project. Explain the purpose, mechanism, assumptions, and evidence; do not substitute a file link for an explanation when asked to discuss the research.

**Current task status:** Stage 3 Phase A is implemented in `train/train_3.py` and `train/stage3_data.py`, following the user's 2026-10-08 request to build the code. Natural versus minority-aware ASL fine-tuning is ready for university GPU preflight. Local synthetic CPU tests cover indexing, sampling, frozen backbone, export and exact resume. No real-data training or GPU jobs have run. See [run instructions](docs/research/stage3-running.md). Later memory/triplet/boosting phases remain unimplemented pending evidence.

## User requirements and boundaries

- 2026-10-10: create `entropy-ablation` and implement the three-arm study; no new remote push or training submission requested in this turn. Existing `long-tailed` code was previously pushed. `scripts/run_entropy_ablation.sh` and `analysis/entropy_ablation_report.py` launch/report this separate study.

- 2026-10-09: user requested one-line university execution with visible progress/results. `scripts/run_stage3.sh` submits CPU preflight -> GPU smoke + A1/A2 -> CPU comparison and opens a monitor. Only Phase A is automated; further phases remain evidence-dependent.

- Latest authorization: the user explicitly requested creating `long-tailed` and pushing the code there. This supersedes the earlier no-push instruction for this branch publication.

- Earlier request: pull/fetch new code on all branches, but **do not push anything**. Branches were synchronized during the initial audit. Do not assume remote refs remain current indefinitely.
- Research/design phase: understand code, saved results, and manuscripts; research primary literature; propose and challenge methods. The user confirmed university GPUs are available. The agreed plan now specifies pilot defaults, but no jobs have been launched or resource allocation committed.
- Phase A implementation and synthetic CPU training checks are authorized and performed. No real-data training jobs, commits or pushes have been performed. Existing Stage 2 implementation is unchanged.
- Latest clarification: follow the leader's memory-centered Stage 3 approach. Do not redirect the task toward patient-history modeling unless the user asks.
- Do not turn uncertain interpretations of the leader's drawings into confirmed requirements or promise that an architectural change improves results or guarantees novelty.

## Leader's instructions and drawings

The user explicitly relayed:

1. **Emphasize Minority (potential isolated training) -> Stage 3.**
2. **Batch size -> minority classes.**
3. **Boosting weight related to Minority.**

The first drawing shows an X-ray feeding groups labeled “Memories,” connected to output labels. One group is circled and marked “Minority?” Its written direction is approximately:

> On triplet-loss memory bank, find which memories are minorities, then boost the weights of that memory.

It mentions focal loss and apparently online hard-example mining (handwriting ambiguous), while noting such approaches do not fully solve long-tailed learning.

**Accepted direction:** identify minority-related memories within the learning process, then strengthen their contribution in Stage 3. This is more specific than ordinary inverse-class-frequency weighting. “Memories” refers here to triplet-learning representations, not earlier patient reports.

**Subsequently resolved:** the leader explicitly defined minorities as labels with few positive records, selected by a threshold. The user chose training prevalence strictly below 1%. This supersedes earlier interpretations based on feature clustering or memory usage. Two drawn groups do not establish a requirement for two networks. The plan operationalizes isolated training as a frozen backbone with a trainable decoder, and boosting as target-specific metric-loss weighting; these are design choices in the agreed plan, not verbatim leader requirements.

The second figure is a **metadata-to-label Cramer's V association heatmap**, rows ordered by training prevalence. It is not a class-to-class graph or direct evidence of gradient/representation dominance. Pale tail rows do not prove the model ignores those labels.

## Revised Stage 3 design — Phase A implemented, GPU validation pending

Target question: **Which minority-related memories receive insufficient relevant supervision, and can focused Stage 3 training correct that without harming other classes?**

Distinguish:

- **Dataset minority:** few positive training examples/patients.
- **Memory minority:** few usable stored examples for a finding or pattern.
- **Underused memory:** examples exist but rarely participate in relevant, active updates.

These are not equivalent. **Minority membership now uses dataset frequency only: positive training prevalence <1%.** Memory coverage and usage remain diagnostics. Do not revive the earlier proposed combined rarity/usage definition as the chosen rule.

Proposed sequence:

1. Link memory entries to labels and sample provenance; measure class coverage and age.
2. Count positive exposure, anchor selection, valid partners, active loss, and whether a positive partner actually shares the target minority finding.
3. Improve minority access through batch composition and partner selection.
4. Apply bounded, finding-specific weighting to relevant comparisons.
5. Compare with simpler sampling/weighting controls and assess all class groups.

Key design cautions:

- A multi-label image may contain common A and rare C. Boosting the whole image can boost A as well; determine which finding a comparison teaches.
- Current bank entries are detached embeddings, not independently trainable memory neurons. Weighting selection/loss is different from multiplying embedding values.
- More weight cannot rescue an absent partner or a zero-loss triplet.
- Larger random batches may still omit rare positives. Gradient accumulation does not concatenate examples for the miner.
- Oversampling and loss weighting compound each other; track actual exposure, repeat counts, and gradients.
- Minority-focused training still needs target-negative examples. Avoid positive-only training and memorization of a few patients.

The revised evidence ladder is:

1. Compare natural classification-only continued fine-tuning with minority-aware classification-only fine-tuning.
2. If sampling helps, compare class-specific triplet learning with a direct class-logit ranking comparator using ordinary FIFO memory.
3. Add reserved per-label memory only if ordinary FIFO partner coverage is a measured limitation.
4. Add bounded minority weighting only after the unweighted method improves held-out ranking/AP.
5. Compare gradual unfreezing only after selecting the decoder-only intervention.

Initial targeted sampling averages one minority-positive image per eight image presentations, rather than the earlier one in four. Labels use tempered sampling and patients are cycled to reduce severe repetition. Performance screening uses approximately one training-set pass instead of 250 updates. Finalists require matched repeated seeds. All 30 labels retain ASL supervision and inference remains image-only.

## Repository state and navigation

Last locally checked on 2026-10-06:

- Current branch: `markov`, commit `aae317c` (`aae317cc0a22f15e5ed552e57778b9b234b3ec8b`).
- `main`: `876fcc1` (`876fcc13df17a5b876d86cd2235d722479db3961`).
- Local branches matched their existing origin refs; no new fetch was performed for this memory-file request.
- Before creating these handoff files, `docs/research/` was untracked and no tracked modifications were reported. These handoff files are also local/uncommitted.
- No Stage 3-named training script was found in `train/` or `scripts/` in the latest search. Recheck before implementing.

Important files:

| Path | Purpose |
|---|---|
| `train/train_2_v3.py` | Main Stage 2 classification + entropy-triplet training |
| `train/utils_update.py` | Memory and triplet sampler, especially `sample_triplets_v13` |
| `train/ml_decoder.py` | Class-query representations |
| `train/markov_layer.py` | v4 label-graph refinement |
| `train/prior_conditioning.py`, `train/train_2_v5.py` | v5 patient-history query conditioning |
| `analysis/patient_prior.py` | Earlier-report lookup and simple temporal chain |
| `analysis/markov_hmm.py`, `analysis/markov_hmm_check.py` | Post-hoc history fusion and richer temporal experiments |
| `docs/paper/entropy/paper.tex`, `docs/paper/markov/paper.tex` | Updated manuscripts |
| `analysis/out/` | Saved predictions, results and manifests |

## Verified model mechanics

- Thirty simultaneous labels; sigmoid classification, not mutually exclusive classes.
- Custom five-stage ConvNeXt2 + ML-Decoder. Classification uses asymmetric loss (ASL).
- v3 positive classification weights: normalized log inverse frequency, clipped to [0.5, 2]. Minority weighting already exists; Stage 3 must explain what changes.
- Metric representation: normalize the mean of decoder class-query features (`head.last_h`). Base query embeddings are frozen; downstream decoder parameters train.
- FIFO memory: 2,048 detached embeddings. Current batch is enqueued after its loss calculation.
- Active v3 entropy gate: softmax entropy at least the physical batch mean. Bernoulli/gate helpers exist, but their presence does not mean v3 uses them.
- Positive partner: shares at least one label outside the five most frequent. Negative: shares no label. The sampler chooses nearest positive and farthest negative: easy mining despite variable names saying “hardest.”
- Triplet coefficient 0.1, margin 0.5 in audited setup. Logged triplet-hit rate measures valid-triplet batches, not nonzero loss or class-specific benefit.
- Audited physical batch 4, accumulation 12. Gate sees 4 examples, not 48.
- EMA 0.999; `swa-last-k` averages K best-metric EMA epochs, not necessarily the final K. Resolution, averaging, flip TTA and ensembling are separate confounds.
- Historical MoE is not active in the audited v3 constructor.

## Markov findings — background, not current research priority

The branch contains distinct paths:

- **v4:** trainable off-diagonal row-stochastic label graph; `z <- u + g * (sigmoid(z) @ P) + b`. Gate and bias start at zero. No patient history required.
- **v5:** 30 prior-vs-prevalence log-odds corrections + availability bit become decoder-query offsets. Zero prior gives zero offset, but a trained v5 with zero history is not necessarily the separately trained v3 baseline.
- **Late fusion:** add a weighted history correction to image logits. Basic temporal chain estimates finding persistence/appearance from previous report-derived states. Richer variants add gap, metadata, cross-label temporal terms and filtered image-history beliefs.
- v4 did not demonstrate consistent gains. History helped the retrospective history-eligible population. Simple calibrated late fusion was a stronger point-estimate baseline than richer temporal fusion; early-vs-late differences were not uniformly statistically conclusive. H5 selected zero added MRF weight.

The earlier assistant proposal “recognize exceptions to historical expectation” is a possible separate longitudinal project. It is **not** the current Stage 3 assignment.

## Evidence and evaluation limits

The [evidence audit](docs/research/2026-10-05-ideation/01-evidence-audit.md) and [snapshot](docs/research/2026-10-05-ideation/evidence_snapshot.json) distinguish recomputed, saved and reported results.

- Train CSV: 103,305 images; validation CSV: 17,274; aligned cached validation predictions: 17,270. Four exclusions still need an explicit ingestion record.
- Audited 18 cached prediction arrays; key/label alignment verified. Four original v5 re-dumps matched original probabilities exactly.
- 13,304/17,270 validation images (77.04%) share a patient with training; 9,186 (53.19%) share a study.
- Frontal patient-unseen subset: 3,572 images, 3,483 patients, only 28 supported labels. No positives for Hydropneumothorax or pneumoperitoneo; its tail AP covers five classes, not seven.
- Rarest training positives: Hydropneumothorax 38, pneumoperitoneo 52. Full cached validation has only 7 and 9 positives, respectively.
- Full-val flip-TTA v3 mAP: s42 .464986; s1024 .462579. v4 variants .458696–.462185; no consistent improvement.
- Saved original v5 ensemble mAP: with history .486723 / zero history .469262; repeat .479710 / .461118. Small single-run gains need repeatability evidence.
- Simple history-eligible frontal comparison (n=4,061), s42: v3 .490889; v5 early .517935; late CV-weight fusion .540175. These are not full-val or official test results.
- Later HMM analysis averages AP within patient halves across split seeds; do not mix those values with pooled full-set AP.
- Training history uses train records; validation history can use strictly earlier train AND validation reports. This is a longitudinal simulation. Same-date records are excluded. It does not establish image-only deployment gains.
- Updated manuscripts already acknowledge many old entropy/mining/resolution confounds. Do not rediscover them as unacknowledged claims.

## Theoretical conclusions and uncertainty

Established properties, not proven explanations of performance:

- Softmax entropy is invariant to a common logit shift, unlike sigmoid uncertainty; it cannot be interpreted as multi-label presence/absence uncertainty. Summed Bernoulli entropies are marginal uncertainty, not joint entropy without independence.
- Easy triplet mining minimizes the eligible hinge constraint for a fixed anchor. Valid triplets may have zero loss while other eligible pairs violate the margin.
- Pooled query features are many-to-one; information loss is possible, but its impact on rare findings is unmeasured.
- Row-normalized co-occurrence weights are graph-transition probabilities, not empirical clinical `P(label j present | label i present)`.
- One-step v4 incoming effects for a target share the sign of its gate; arbitrary opposing incoming relationships are unavailable in that formulation.
- Simple fixed-state late fusion adds the same class-specific constant within each prior-state group. It cannot improve within-group ranking/AP under a fixed monotonic calibration; richer patient-dependent corrections differ.
- Exact Bayesian image/history fusion requires appropriate posterior/prevalence and dependency assumptions. Fitted weights mitigate issues but do not prove exact inference.
- Failed metadata fusion does not prove metadata contain no conditional information.

Unverified: class-specific gate starvation, harmful shared-gradient dominance, insufficient target-specific representations, and whether any proposed Stage 3 modification improves results.

## Existing deliverables and literature

Start with [the ideation package](docs/research/2026-10-05-ideation/README.md): model audit, 22-source literature matrix, ten scored idea cards, three developed proposals, and validation blueprint. Its 2026-10-05 ranking predates the leader's clarified direction; do not treat it as the current final method choice.

Relevant prior work includes [Distribution-Balanced Loss](https://arxiv.org/abs/2007.09654), [decoupled classifier retraining](https://arxiv.org/abs/1910.09217), [SOAP](https://arxiv.org/abs/2104.08736), [SeMi](https://arxiv.org/abs/2501.06004), MulCon/MulSupCon, ws-MulSupCon, and prototype-based CXR learning. See the matrix for reading depth and implementation status. Entropy plus balanced memory, label-specific contrast, and classifier refitting are not novel by themselves.

Reproduce the aggregate audit with:

```text
python docs/research/2026-10-05-ideation/audit_extract.py
```

It reads existing data/predictions and writes the snapshot only; no training/inference. It requires materialized data and numpy/pandas/scikit-learn. It records source hashes, library versions, aggregate patient support and alignment checks, without exporting patient IDs.

## Next steps when implementation is requested

1. Use the [run guide](docs/research/stage3-running.md) to transfer local code and identify university checkpoint/shard paths. A pull from origin will not include these unpushed files.
2. Run the full shard/label preflight, review missing-image exclusions, then the five-update GPU smoke test.
3. Run matched Phase A arms when university execution is requested. No jobs have been submitted.
4. Use `analysis/stage3_compare.py` for paired patient bootstrap. Its unseen subset includes all projections and must not be called the earlier frontal-only five-tail subset.
5. Inspect Phase A results before constructing memory objectives. Sampling failure alone does not logically disprove metric learning. No empirical improvement or novelty is established.

## Continuity updates

- **2026-10-05:** Completed repository/manuscript audit, primary-source review, idea pool and evidence extraction.
- **2026-10-06:** Explained pipeline, Markov alternatives and theoretical limits. User supplied leader's drawings and clarified Stage 3 minority emphasis, batch composition and weights. User stressed memory-centered identification rather than generic class balancing.
- **2026-10-06:** Created this handoff and AGENTS.md so future sessions can resume without reconstructing the conversation. Update this record after substantive decisions; keep detailed experiment results in linked files.
- **2026-10-07:** Leader clarified frequency-based minority labels. User chose <1% prevalence, requested the completed Stage 3 plan, then asked to save/explain it. Plan Mode prevented saving during that turn.
- **2026-10-08:** Saved the agreed plan and updated this handoff in Default mode. No model implementation or training performed. User requested a detailed end-to-end explanation, including the data passed between steps.
- **2026-10-08:** Replaced the all-at-once eight-arm proposal with a progressive, score-oriented plan after independent review. No code or training was performed.
- **2026-10-08:** Implemented Phase A trainer, indexed shard access, patient-aware sampler, exposure accounting, EMA/resume, AP exports and paired-bootstrap tool. Added synthetic CPU checks and university run guide. Installed missing training dependencies locally (pip also upgraded CPU torch from 2.10.0 to 2.14.1). No GPU/data training or pushes.
- **2026-10-08:** Checked Stage 1/2 university scripts and README. Added `scripts/train_3.sh` following the existing Slurm setup: amp48, one GPU, eight CPUs, 64 GB, 36-hour limit, project `.venv`, historical colossus exclusion. Launcher checks CUDA and defaults to the seed-42 768 SWA checkpoint recorded in the prediction manifest. Bash syntax passed. Cluster paths/availability have not been verified remotely; no login or submission performed. Updated the run guide with SSH, Slurm, monitoring and CPU preflight instructions.
