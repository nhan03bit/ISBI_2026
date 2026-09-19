# Journal-extension analysis pipeline

Everything here supports the paper revision: per-class / head-medium-tail results,
the 384-vs-512 resolution control, co-occurrence-conditioned analysis, the
class-weight-order bug probe, and the A4 ablations. GPU work is Slurm; analysis is
CPU and runs on the login node.

## 0. Class metadata (no GPU)

```bash
python evaluate/class_groups.py
```

Prints the canonical 30-class list in **shard/logit order** (`label_info.pt`
`class_names`), training prevalence, and the head/medium/tail split
(CXR-LT 2024 thresholds: head >10 % = 3 classes, medium 1-10 % = 20, tail <1 % = 7;
extreme tail <0.5 % = 6). This is the single source of truth for every table.

## 1. Per-class eval of existing checkpoints  ->  A1 + A2

```bash
bash scripts/submit_eval_perclass.sh --dry-run      # inspect
bash scripts/submit_eval_perclass.sh                # 13 jobs (~30-60 min each)
```

Writes next to each checkpoint: `eval_result.<tag>.json` (macro + `group_mAP` +
`per_class`) and `preds.<tag>.npz` (`probs`, `labels`, `class_names`). Tags:
`res384` / `res512`. The converged Stage-1 is evaluated at **both** resolutions
(A2). `evaluate/evaluate.py` gained `--img-size`, `--save-preds`, `--tag`, and
per-class AP/AUROC (`average=None`); the old 4-scalar JSON fields are unchanged.

Build the tables once the jobs finish:

```bash
python analysis/per_class_report.py \
  --cond stage1_384  train/checkpoint3/Model_20260119_062652/preds.res384.npz \
  --cond stage1_512  train/checkpoint3/Model_20260119_062652/preds.res512.npz \
  --cond proposed    train/checkpoint_Triplet_2/accum_6/seed_*/Model_20260827_*/preds.res512.npz \
  --baseline stage1_512 --out analysis/out/perclass_converged
```

Outputs `perclass_converged.json` + `.tex` (frequency-group table + full 30-class
table, mean +/- seed std, bootstrap CI over val samples for the tail-measurability
caveat).

## 2. Co-occurrence analysis  ->  A3

```bash
python analysis/cooccurrence.py \
  --cond stage1   train/checkpoint3/Model_20260119_062652/preds.res512.npz \
  --cond proposed train/checkpoint_Triplet_2/accum_6/seed_86/Model_20260827_095451/preds.res512.npz \
  --out analysis/out/cooccurrence
```

Reports, per tail/medium class: AP on positives that **co-occur with another
finding** vs positives **in isolation**, plus mean percentile-rank, for each
condition and the delta. Also the training rare x head co-occurrence matrix.

## 3. B0 corrections + "measure impact first" probe

The published Stage-2 runs used softmax entropy (Eq. 1, single-label form) and
**class weights applied in the wrong order** (computed in CSV column order,
applied to shard-order logits -- e.g. Hydropneumothorax, the rarest class, was
weighted 0.58 instead of the intended 2.00). Both are now CLI flags on
`train_2_v2.py`:

- `--class-weight-order` defaults to **`shard`** (the correct mapping) -- every new
  run gets fixed weights unless it passes `csv`.
- `--entropy-mode` defaults to `softmax` (unchanged from the paper); pass `binary`
  for the multi-label-correct form. The B0 re-run scripts pass `binary`.

NB: any in-flight sweep started after this change that does not pass
`--class-weight-order` now trains with the corrected weights (a behaviour change
vs. earlier runs of the same sweep).

```bash
bash scripts/submit_classweight_probe.sh --dry-run   # 2x2: {csv,shard} x {softmax,binary}, seed 86
bash scripts/submit_classweight_probe.sh
# then per-class eval each, then compare macro + tail mAP:
python analysis/per_class_report.py \
  --cond published train/checkpoint_Triplet_2/probe/cw_csv__ent_softmax/seed_86/Model_run/preds.res512.npz \
  --cond fixed     train/checkpoint_Triplet_2/probe/cw_shard__ent_binary/seed_86/Model_run/preds.res512.npz \
  --baseline published --out analysis/out/probe
```

`cw_csv + ent_softmax` should reproduce the published ~0.414 macro mAP (a B4.1
sanity check). If the tail mAP moves materially, run the full re-run:

```bash
bash scripts/submit_stage2_fixed.sh          # 5 seeds x {converged, under-trained}
```

## 4. A4 ablations

```bash
bash scripts/submit_ablations.sh anchor      # {max, mean, all, random} anchor gate
bash scripts/submit_ablations.sh mining      # {easy, hard, semihard}
bash scripts/submit_ablations.sh memory      # {48, 96, 512, 2048}
bash scripts/submit_ablations.sh pos         # pos_min_shared {1, 2}
bash scripts/submit_ablations.sh lambda      # {0.05, 0.1, 0.2}
bash scripts/submit_ablations.sh entropy     # {softmax, binary}
SEEDS="86 123 456" bash scripts/submit_ablations.sh anchor   # key axes: >1 seed
```

Base = fixed paper recipe (converged Stage-1, binary entropy, shard weights,
mean gate, pos_min_shared 1, easy mining, lambda 0.1, memory 96). Each axis varies
one knob. Then per-class eval every `ablate/*/seed_*/Model_run/model_best.pth`.

## Notes / caveats

- The paper's "7.6 % triplet utilisation" is from a *different* config (accum 8,
  30-epoch convergence sweep, under-trained ckpt, seed 789 = job 44260). The
  Table-1 recipe logs ~13 %. `train_2_v2.py:707` logs the per-epoch hit rate.
- Tail classes have very few val positives (Hydropneumothorax ~7, pneumoperitoneo
  ~9 in 17k) -- per-class tail AP is high-variance. `per_class_report.py` reports
  `val_pos` and a bootstrap CI; consider the challenge dev set (3,020
  PadChest-GR) if reachable.
- `evaluate/evaluate.py` uses `strict=False` state-dict loading; it now logs
  missing/unexpected keys so an architecture mismatch is visible.
