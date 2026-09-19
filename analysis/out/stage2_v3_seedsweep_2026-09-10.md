# Stage-2 v3 seed sweep + Wave-2 ablations — results record

**Date:** 2026-09-10 · **Monitor:** `scripts/mon.py 'isbi2026_v3s*'` · **14 jobs** (`isbi2026_v3s2_*` + `isbi2026_v3seed_*`)
**Code:** `train/train_2_v3.py` @ 256b439-dirty · **Stage-1 init:** `checkpoint3/Model_20260119_062652/model_best.pth` (converged, same as paper Table I)
**Recipe (seed sweep = `fix_only` arm):** lr `1e-4`, **memory 2048**, `--metric-embed query`, `--head-classes 5`
(head IDs `[6,24,17,9,12]`), `--triplet-lambda 0.1`, `--class-weight-order shard` (**ordering bug fixed**),
eff. batch 48 (accum 6 × 8), 8 epochs / patience 3, monitor mAP, **EMA weights (decay 0.999) — reported
metrics are from the EMA copy**. Softmax entropy, mean-threshold anchor gate.

### This recipe diverges from the paper's described method — not just hyperparameters

| component | paper.tex | v3 seed sweep |
|---|---|---|
| memory bank size | `\|M\|=96` (§Cross-Batch Memory Bank, "12× the mini-batch") | **2048** (256×) |
| positive-overlap rule | `yₘ·yᵢ > 1` (Eq. 5) | exclude 5 head classes, then ≥1 shared non-head label |
| triplet hit rate | "only **7.6%** of steps" (§Triplet Utilization, one of the 4 contributions) | **96.6% ep1, 96.3% ep2** (from seed-1024 log) |
| weight averaging | not mentioned | **EMA, decay 0.999** |
| learning rate | `3×10⁻⁵` | `1×10⁻⁴` |
| fine-tuning length | "a single epoch", patience 1 | 8-epoch budget, patience 3, best @ ep 2–3 |
| class weights | CSV order (bug) | shard order (fixed) |

The v3 recipe essentially implements the "future work" from §Triplet Utilization Rate
("a larger memory bank, a relaxed positive threshold for rare classes") — and it works:
utilization goes 8% → 96%. That **removes the paper's 4th contribution** (the low-utilization
limitation) rather than just changing a number.

All numbers below are the **best-epoch** (early-stopping) checkpoint, read from each run's
`history.json` (last row = best epoch) and cross-checked against the SLURM `.out` epoch log.
The `ep·*` columns in the monitor are the *last* (overfit) epoch and are **not** used here.

---

## 1. Seed sweep — `checkpoint_Triplet_3/seedsweep/seed_*/Model_run/`

| seed | job   | best ep | mAP    | mF1    | mAUC   | mECE   | status |
|------|-------|---------|--------|--------|--------|--------|--------|
| 86   | 46472 | 2       | 0.4385 | 0.4461 | 0.9009 | 0.1320 | done   |
| 123  | 46473 | 3       | 0.4389 | 0.4552 | 0.8961 | 0.1155 | done   |
| 1024 | 46476 | 2       | 0.4434 | 0.4570 | 0.8999 | 0.1309 | done   |
| 42   | 46471 | —       | —      | —      | —      | —      | **OOM-killed, epoch 1** (2 `oom_kill` events, dataloader worker) |
| 456  | 46474 | —       | —      | —      | —      | —      | **OOM-killed, epoch 1** |
| 789  | 46475 | —       | —      | —      | —      | —      | **OOM-killed, epoch 1** |

**Aggregate over the 3 completed seeds (86, 123, 1024), mean ± sample sd:**

| metric | mean ± sd     | vs Stage-1 (0.385 / 0.395 / 0.875 / 0.134) |
|--------|---------------|--------------------------------------------|
| mAP    | 0.440 ± 0.003 | +0.055 |
| mF1    | 0.453 ± 0.006 | +0.058 |
| mAUC   | 0.899 ± 0.003 | +0.024 |
| mECE   | 0.126 ± 0.009 | −0.008 (calibration slightly **better**, not worse) |

Best single run: **seed 1024, mAP 0.4434 @ ep2** (job 46476).

> Note: seed 86 also ran a second time as the `fix_only` arm of the Wave-2 sweep (job 46457,
> mAP 0.4349 @ ep2). Same config, same `--seed 86`; the 0.0037 gap is run-to-run
> nondeterminism (GPU atomics + node/GPU-type differences), which bounds how tight the
> 3-seed sd can be trusted to be.

---

## 2. Wave-2 anti-overfit ablations — `checkpoint_Triplet_3/sweep2/<arm>/seed_86/` (seed 86 only)

Baseline for Δ is the `fix_only` arm (bug-fixed recipe, nothing else changed).

| arm       | job   | best ep | mAP    | Δ mAP   | mF1    | mAUC   | mECE   | what it isolates |
|-----------|-------|---------|--------|---------|--------|--------|--------|------------------|
| dp01      | 46469 | 2       | 0.4399 | +0.0050 | 0.4519 | 0.9044 | 0.1373 | ConvNeXt drop-path 0.1 |
| bug_repro | 46458 | 2       | 0.4383 | +0.0034 | 0.4326 | 0.8998 | 0.1388 | control: CSV-order weights (bug kept) |
| fix_only  | 46457 | 2       | 0.4349 |  0.0000 | 0.4524 | 0.8995 | 0.1324 | ordering-bug fix alone |
| llrd07    | 46466 | 2       | 0.4323 | −0.0026 | 0.4530 | 0.8933 | 0.1211 | layer-wise LR decay γ=0.7 |
| sched3    | 46465 | 1       | 0.4301 | −0.0048 | 0.4373 | 0.9014 | 0.1408 | cosine horizon 3 ep |
| freeze3   | 46467 | 2       | 0.4229 | −0.0120 | 0.4401 | 0.8970 | 0.1328 | freeze backbone stages ≤3 |
| combo     | 46470 | 2       | 0.4231 | −0.0118 | 0.4368 | 0.8907 | 0.1207 | sched3 + llrd07 |
| freeze4   | 46468 | 3       | 0.4130 | −0.0219 | 0.4293 | 0.8929 | 0.1363 | freeze backbone stages ≤4 |

**Reading:** none of the anti-overfit knobs beat the plain recipe on mAP by more than
run noise; `dp01` is the only one nominally ahead (+0.005, one seed). Freezing hurts.
Every arm still peaks at epoch 1–3 then overfits — the epoch-2 optimum is structural,
not fixable with these knobs. The `bug_repro` gate lands at 0.4383 (Δ vs the Wave-1
lr-1e-4 baseline of 0.435 = +0.0033) → the v3 pipeline reproduces the pre-fix result,
so the sweep is trustworthy.

**Effect of the ordering-bug fix itself** (`bug_repro` → `fix_only`, seed 86):
mAP −0.003 (noise), **mF1 +0.020**, mECE −0.006. The fix mainly moves mF1 and
(expected) per-class / tail behaviour, not aggregate mAP.

---

## 3. Reconciliation with `docs/paper.tex` (as of 2026-09-07)

The current Table I "Stage 2: Proposed" row (mAP 0.414 ± 0.002) is the **v2** measurement:
`checkpoint_Triplet_2/accum_6/seed_*/Model_20260827_*`, lr ≈ 3e-5, **1 epoch**, 5 seeds
(86/123/456/789/1024), softmax entropy, **CSV-order weights (bug present)**. Verified to
reproduce 0.414 / 0.407 / 0.889 / 0.143.

The v3 seed sweep is a **different configuration**, not just a re-run. Conflicts if it
becomes the paper's canonical result:

| paper.tex claim (location) | current text | v3 seed-sweep data |
|---|---|---|
| Table I Proposed mAP / mF1 / mAUC / mECE | 0.414 / 0.407 / 0.889 / 0.143 | **0.440 / 0.453 / 0.899 / 0.126** (3 seeds) |
| seed count (§Exp. Setup, Tab. I & II captions) | "five seeds: 86, 123, 456, 789, 1024" | 6 launched, **3 completed** (86, 123, 1024); 42/456/789 OOM |
| learning rate (§Exp. Setup) | `3×10⁻⁵` | `1×10⁻⁴` |
| "a single fine-tuning epoch" (abstract, §Intro contribs, §Results ×2, §Conclusion) | single epoch, patience 1 | **best epoch = 2–3**, patience 3 |
| calibration regression (abstract, §Intro, §Results, §Conclusion) | "small mECE regression … 0.134 → 0.143" | mECE **0.126** ± 0.009 → regression **gone / flat** |
| vs MoE framing (abstract, §Results, §Conclusion) | "matches MoE on mAP and mAUC" (+0.009 / +0.007) | mAP **+0.035**, mAUC +0.017 → "matches" understates it |
| positive-mining rule (§Method, Eq. 5) | `yₘ·yᵢ > 1` (share >1 label) | `--head-classes 5` exclusion + ≥1 shared non-head label |
| memory bank size (§Cross-Batch Memory Bank) | `\|M\|=96`, "12×" the mini-batch | **2048**, 256× |
| EMA weight averaging | not mentioned | decay 0.999; reported metrics are the EMA copy |
| §Triplet Utilization Rate (contribution #4) | "only **7.6%** of steps yield a triplet" | **~96%** (memory 2048 + head exclusion) — limitation removed |
| Table II (checkpoint maturity) | under-trained ckpt, 5 seeds | **not re-run** in this batch — v2-only, keep scoped to old config |

The mAP jump 0.414 → 0.440 is driven mostly by the **Wave-1 finding** (lr 1e-4 > 5e-5,
allow epoch 2), established before this batch; the bug fix contributes ~0 mAP. This batch's
contribution is: (a) confirms the lr-1e-4 / epoch-2 result holds across 3 seeds, (b) fixes
mF1 and calibration via the weight-ordering correction, (c) shows anti-overfit knobs don't help.

---

## 4. Edits applied to `docs/paper.tex` (2026-09-10)

Decision taken: **full swap to the v3 story now, 3 seeds**; Methods divergences flagged, not rewritten.
Paper rebuilds clean (`latexmk`, 5 pages, 0 warnings).

**Done — numbers + claims:**
- Table I "Stage 2: Proposed": `0.414/0.407/0.889/0.143 (±0.002/0.010/0.002/0.008, 5 seeds)`
  → **`0.440/0.453/0.899/0.126 (±0.003/0.006/0.003/0.009, 3 seeds)`**; mECE now bold (best row).
- §Results prose: Δ vs Stage-1 (0.055 / 0.058 / 0.024), "single epoch" → "two to three epochs",
  calibration "regression" → "not degraded"; MoE Δ (+0.035 / +0.051 / +0.017, mECE −0.004),
  dropped the "regression is a general side effect" sentence.
- Abstract: "single fine-tuning epoch" → "a few", "matches MoE on mAP and mAUC" → "surpasses
  … on mAP, mF1 and mAUC", "five seeds" → "three", "disclose a small calibration regression"
  → "without a calibration regression".
- Intro contribution #3: five→three seeds, single epoch→two-three, regression→"without degrading".
- §Experimental Setup: lr `3e-5` → `1e-4`; "single epoch / patience 1" → "8-epoch budget /
  patience 3, best within 2–3 epochs"; "five seeds: 86,123,456,789,1024" → "three seeds: 86,123,1024".
- §Sensitivity + Table II caption: scoped to "earlier configuration (lr 3e-5, single epoch)";
  the +0.029 converged gain annotated as earlier-config, cf. +0.055 now.
- Conclusion: single epoch→two-three, "comparable to MoE … slightly worse calibration … report
  plainly" → "exceed MoE on all three … without degrading calibration"; dropped "calibration
  regression" from the limitations list.

**Flagged with `% TODO(v3)` comments — need your scientific framing:**
- §Cross-Batch Memory Bank — `|M|=96` → 2048, and EMA (decay 0.999) is undocumented.
- §Hard Positive/Negative Mining, Eq. (5) — `yₘ·yᵢ > 1` → head-class-5 exclusion + ≥1 shared non-head.
- Contribution #4 + §Triplet Utilization Rate — 7.6% → ~96%; "limitation" is now "resolved".
- §Experimental Setup — seeds 42/456/789 OOM'd; rerun and restore to 5
  (`scripts/submit_stage2_v3_seedsweep.sh 42 456 789`, higher `MEM`).
- Table II — under-trained row still v2; needs a v3 rerun to match Table I.
- Abstract still promises "head-, medium- and tail-class mAP" — that per-class table is not yet
  in §Results (pre-existing; analysis pipeline in `analysis/README.md` §1).
