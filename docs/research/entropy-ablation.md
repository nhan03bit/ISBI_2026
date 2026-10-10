# Entropy contribution: controlled Stage 2 ablation

2026-10-10. Implemented on `entropy-ablation`; GPU experiments not run.
This is a separate study from minority-sampled Stage 3.

## Question and primary contrasts

| Arm | Classification | FIFO triplet objective | Anchor selection |
|---|---|---|---|
| baseline | ASL | off | none |
| triplet | ASL | weight 0.1 | every eligible anchor |
| entropy | ASL | weight 0.1 | softmax entropy >= physical-batch mean |

`triplet - baseline` estimates the contribution of the memory/triplet package.
`entropy - triplet` estimates the incremental effect of its entropy gate.
These are matched training experiments, not comparisons with unrelated published checkpoints.

## Fixed recipe

All arms initialize strictly from the same pre-triplet Stage 1 checkpoint:
`train/checkpoint3/Model_20260119_062652/model_best.pth` by default. Override `CKPT` if
needed, but verify its lineage. Initializing from an entropy-trained Stage 2 checkpoint
would leave entropy exposure in the controls and cannot identify its original contribution.

Reuse the existing Stage 2 v3 architecture and streaming training pipeline:

- 768 pixels, batch 4, accumulation 12; ordinary sampling only.
- All backbone/decoder parameters retain the existing Stage 2 trainability policy;
  full-backbone fine-tuning, frozen base query embeddings as before.
- AdamW, LR 1e-4, weight decay .01, drop-path .1, 5% warmup.
- Exactly four epochs on the existing eight-epoch cosine horizon, identical across arms.
- Matched seeds, augmentation seed 42, worker counts, shuffle buffer, label-order-corrected
  ASL weights, augmentation and shard inputs.
- Metric: normalized mean of decoder queries, 2,048-record FIFO, margin .5.
- Preserve existing nearest-positive/farthest-negative mining (easy mining). Positives
  share a non-head label; negatives share no labels. No new mining/label-specific features.
- Memory is updated after the current batch loss. Both metric arms insert every batch,
  regardless of gate decisions. Detached partner features do not receive gradients.
- EMA .999 for all arms; primary result is `model_final.pth` after epoch four.
- No early stopping, SWA, flip TTA, ensemble, Markov layer or patient history.

Checkpoint selection is fixed, but learned weights necessarily differ between arms.
The baseline skips the metric loss and memory updates. No random selection is introduced,
so removing the gate does not consume an extra RNG stream or alter sample ordering.
Fixed-budget mode refuses decode failures rather than silently changing the sample set.

## Metrics and provenance

`history.json` includes initial metrics, every epoch's overall/per-label/head/medium/tail AP,
positive validation counts, and per-epoch anchor/selected/valid/active triplet counts.
Head = top five training-frequency labels. Tail = positive training prevalence <1%.
Medium = remaining labels. Definitions and class indices/names are saved in the manifest.
Unsupported validation labels are null per-class and excluded from supported group means.

`manifest.json` records the complete resolved configuration, checkpoint/CSV/label-info/source
hashes and shard path/size/mtime inventory. The comparison refuses unmatched configurations,
incomplete budgets, different initial evaluations or different training image counts.
The current initial-evaluation check requires exact equality; hardware-dependent numerical
differences must be investigated rather than silently accepted by the report.

`model_best.pth` remains an auxiliary training artifact. The primary table always uses
the final endpoint; selecting the best epoch separately for each arm changes the estimand.

## University commands

Once this branch is available on the university machine, from the repository root:

```bash
# Quick end-to-end tests of all three configurations; not performance evidence:
SMOKE=1 bash scripts/run_entropy_ablation.sh

# Single-seed screening:
bash scripts/run_entropy_ablation.sh

# Matched repeats for paper evidence:
SEEDS="42 86 1024" bash scripts/run_entropy_ablation.sh
```

The launcher submits one single-GPU job per arm/seed, using the existing `.venv`, amp48,
64 GB RAM and historical colossus exclusion. It prints job IDs and a unique run directory.
`TIME`, `GPU_PARTITION`, `EXCLUDE`, `TRAIN`, `VAL`, `CKPT` and `VENV_DIR` are overridable.
Checkpoint/data existence and Slurm allocation remain to be verified on the university cluster.
Do not launch both screening and repeated runs unnecessarily: seed 42 would be repeated.

Progress is the existing Stage 2 tqdm train/eval bar in each Slurm log. Inspect `jobs.tsv`,
`squeue -u "$USER"`, and `tail -f RUN/logs/*.out` for status. The Stage 2 trainer does not
implement optimizer-resume; interrupted arms must restart in fresh output directories.

When every arm for each submitted seed has completed:

```bash
.venv/bin/python analysis/entropy_ablation_report.py runs/entropy_ablation_TIMESTAMP
```

This writes `ablation.csv` and `RESULTS.txt`, including mean/sample SD across seeds and
paired seed-level overall AP differences. A single seed is explicitly marked. These are
not patient-bootstrap confidence intervals. Validation patient overlap and tiny tail support
remain limitations; this study establishes an internal ablation, not independent-population
generalization. Do not reuse Stage 3's image-only prediction files for this experiment.

## Interpretation and four-page paper focus

The shipped statistic is softmax entropy across nonexclusive labels. It is a heuristic
selection score, not a calibrated measure of multi-label clinical uncertainty. Switching to
Bernoulli entropy would be a separate experiment and is deliberately not bundled here.

The gate changes how many triplets contribute. The requested three arms isolate the gate's
net effect, but do not establish that entropy ordering is superior to any equally sparse
selection. A matched-count random gate would be a useful secondary control if evidence and
page space warrant it. Positive results, novelty and mechanism remain unproven.

Proposed four-page narrative:

1. Problem: long-tailed multi-label CXR learning; state a narrow entropy-selection hypothesis.
2. Method: one compact memory/mining/gate description and a single loss equation.
3. Experiment: this three-row table with overall/head/medium/tail AP, matched seeds and controls.
4. Analysis: entropy-vs-ungated delta plus selected/valid/active-triplet diagnostics; discuss
   easy mining, heuristic uncertainty, split overlap and limited tail support honestly.
5. Conclusion: use only what this ablation demonstrates. Keep Markov/history work outside
   the main contribution; at most brief context/future work if scientifically relevant.

No manuscript results have been invented or replaced. If entropy does not beat ungated
triplet learning consistently, the paper must not attribute the package's gain to entropy.
