# Entropy-gated triplet vs MoE, against the "without entropy" Stage-1 baseline — 2026-10-08

Type: findings / provenance note and plan (not a training-result report).
Run-level metrics for every Stage-1 run: [docs/result/2026-10-08-stage1-original-baseline.md](../result/2026-10-08-stage1-original-baseline.md).
Builds on: [2026-10-02-rev4-numbers-check.md](2026-10-02-rev4-numbers-check.md), [2026-10-02-isbi2027-unified-paper-numbers-check.md](2026-10-02-isbi2027-unified-paper-numbers-check.md), [2026-10-02-047-provenance-and-paper-corrections.md](2026-10-02-047-provenance-and-paper-corrections.md).
All numbers are internal validation (17,270-image fold), not leaderboard. Numbers re-read from artifacts on 2026-10-08.

## Question
The paper (`docs/paper/isbi/paper.tex`, Table 1, label `tab1`) uses Stage 1 as the "without entropy" baseline and claims the entropy-gated triplet Stage 2 beats the MoE Stage 2. Which run produced the Stage-1 number, at what resolution, is 0.385 reproducible from the repo, and how much of "entropy over MoE" does the table actually support?

## Data and method
Read `history.json` / `eval_result*.json` of every Stage-1 run, the Slurm and run logs, `train.py`, the b12 PDF (`docs/pdf/ISBI2026.pdf`, text extracted), `train_2_v2.py` / `train_2_v3.py` flags, `scripts/submit_ablations.sh`, and `sacct` for what was ever submitted. No jobs launched, no existing file modified.

## Findings

### 1. Provenance of the Stage-1 row
- Paper row 1 (0.385 / 0.395 / 0.875 / 0.134) = b12 Table 1 stage 1 = `train/checkpoint3/Model_20260119_062652/history.json`, best epoch 6: 0.3849 / 0.3946 / 0.8746 / 0.1338. This is a training-time number at **384 px** (b12: "384x384 during training and evaluation"). The 0.505 in b12 Table 2 is the development-leaderboard score of the same Stage-1 checkpoint and is a different quantity.
- Re-measured with `evaluate/evaluate.py` at **512 px** (job 47073, 17,270 images): **0.3688 / 0.3689 / 0.8643 / 0.1408**. Head 0.7051, medium 0.3765, tail 0.2028, extreme tail 0.2298.
- Retraining `train.py` (jobs 41730, 42717, Aug 2026, 384 px) gives best 0.3771 at epoch 7, then overfits to 0.3740 at epoch 30. This does **not** reproduce 0.3849. It is the "under-trained" Stage 1 in the paper's Table 2.
- July runs (34342, 34546, 34709, labelled "buggy_posenc"): 0.298 to 0.340. Not baselines.
- So: 0.385 is traceable to a file in the repo (the history file), but not reproducible from the repo, because the January 2026 code, args and Slurm log are gone and a retrain lands 0.008 lower. The only resolution-matched re-measurement the repo holds is 0.3688 at 512 px. There is no 384 px evaluation of the checkpoint (the `res384` job in `scripts/submit_eval_perclass.sh` was never submitted; `sacct` shows only the `stage1` 512 px eval, job 47073).
- The 0.385 vs 0.3688 gap (0.016 mAP, 0.010 mAUC, 0.026 mF1, 0.007 mECE) can have three non-exclusive causes, none separable with current artifacts: (a) 384 px vs 512 px inference on a model trained at 384, (b) a different validation fold or preprocessing in the January run (b12 says about 17,000 validation images; fold identity not recorded), (c) a code-state difference in the loaded model (positional-encoding class, see the result document). The 384 px re-evaluation settles (a) vs (b)+(c).
- Peak LR of the original run was 3.9e-5 (exact fit to `history.json` `lr`), not the 4e-5 in b12.

### 2. Three-row comparison

| Row | Res | mAP | mF1 | mAUC | mECE | Source / traceability |
|---|---|---|---|---|---|---|
| Stage 1 (no entropy, no MoE), as in paper | 384 | 0.3849 | 0.3946 | 0.8746 | 0.1338 | `Model_20260119_062652/history.json` ep 6 = b12 Table 1. Traceable to a repo file |
| Stage 1, re-evaluated | 512 | 0.3688 | 0.3689 | 0.8643 | 0.1408 | `eval_result.stage1.json`, job 47073. Traceable |
| Stage 2: MoE (b12) | 384 | 0.405 | 0.402 | 0.882 | 0.130 | b12 Table 1 only (3 d.p., no seeds, no s.d.). **No MoE checkpoint, history, config or log in the repo.** `convnext3.py` (root, not imported by any `.py` or `.sh`) and the inert aux-loss hook in `train_2_v3.py` are not the MoE run |
| Stage 2: entropy-gated triplet, original impl., 5 seeds | 512 | 0.41375 +/- 0.00199 | 0.40651 +/- 0.00973 | 0.88893 +/- 0.00171 | 0.14333 +/- 0.00825 | Jobs 43047-43051 (2026-08-27), `train/checkpoint_Triplet_2/accum_6/seed_{86,123,456,789,1024}/Model_20260827_*/history.json`, best epoch 1 of 1 each (recomputed today; matches rev4 check) |

Deltas (unrounded; sample s.d. for the entropy row):

| Delta | mAP | mF1 | mAUC | mECE (negative = better) |
|---|---|---|---|---|
| MoE minus Stage 1 @384 (like-for-like resolution) | +0.020 | +0.007 | +0.007 | -0.004 |
| Entropy minus Stage 1 @384 (paper's +0.029) | +0.029 | +0.012 | +0.014 | +0.009 |
| Entropy minus Stage 1 @512 (resolution-matched evaluation) | +0.045 | +0.038 | +0.025 | +0.003 |
| Entropy minus MoE (paper's +0.009) | +0.009 | +0.005 | +0.007 | +0.013 |

### 3. What the table does and does not support
Supports:
- The entropy-triplet Stage 2 run beats its own Stage-1 initialisation in mAP, mF1 and mAUC across five seeds (all five seeds 0.4108 to 0.4156, against a single Stage-1 value). This does not require the baseline to be 0.3849 or 0.3688.

Does not support (any one of these is enough to block "entropy beats MoE"):
1. **Resolution confound, in both directions.** Stage 1 and MoE are 384 px; the entropy row is 512 px. The Stage-1 checkpoint itself loses 0.016 mAP when scored at 512 px, so the +0.029 gain over Stage 1 becomes +0.045 under a resolution-matched evaluation, and it also contains whatever 512-px adaptation the fine-tuning provides. The size and sign of the resolution effect on the MoE row is unknown.
2. **No no-entropy control.** The paper says "Stage 2" improves things, but a Stage-2 run with the same recipe and the triplet term off was never run (section 4). The +0.029 therefore bundles: one more epoch of training (lr 3e-5, constant), 512 px, class-weighted ASL (Stage 1 has no class weights; Stage 2 adds per-class positive weights, in the buggy CSV order for row 3), the memory bank/triplet term, and the entropy gate. Which of these produces the gain is unmeasured.
3. **The MoE row has no artifact, no seeds and different training budget.** It is one number from b12, trained "an additional 2 epochs" with most backbone parameters frozen, against 1 epoch full fine-tuning for the entropy row. +0.009 mAP is about 4.4 s.d. of the entropy row's seed spread (0.002), but the MoE row's own run-to-run spread is unknown, so no significance statement can be made.
4. **Row 3 is pre-fix.** Jobs 43047-43051 used CSV-order class weights (Known Bug 1: tail weights near-inverted, e.g. Hydropneumothorax 0.58 instead of 2.00), the legacy pre-`--metric-embed query` embedding (Known Bug 2), softmax entropy, the "share at least one label" positive threshold (not the paper's `>1`), bank 96, and 13-14% triplet utilisation in re-runs (the 7.6% figure comes from a different configuration). The code that ran is not preserved (`train_2_v2.py.bak2` is the nearest). The "same Stage-1 checkpoint" caption is established only indirectly for these five jobs (no checkpoint path logged; consistent in mAP level with re-runs 44274-44278 that do log it).
5. **Selection effect.** Stage-1 and MoE numbers are best-epoch-on-validation values; the entropy row has no selection (single epoch). This slightly favours the baselines, so it does not explain the gain, but it breaks symmetry.
6. **Internal-only and not patient-disjoint.** The validation fold shares patients with train (77%), so all internal gaps may differ on the frontal, patient-disjoint test set. The only test-set number in the story (0.4599, 3rd place) belongs to the MoE submission; no entropy-triplet model has been scored on the leaderboard.
7. **Head/medium/tail effect of the paper's row 3 vs Stage 1 is unmeasured.** The only per-class evaluations are of post-fix v3 runs (below).

Context, not Table 1: post-fix v3 recipe (lr 1e-4, bank 2048, 8-epoch budget, best at epoch 2, seed 86, 512 px; `eval_result.*.json`) against Stage 1 re-evaluated at 512 px, head / medium / tail / extreme-tail mAP:

| Model | mAP | head | medium | tail | extreme tail |
|---|---|---|---|---|---|
| Stage 1 @512 | 0.3688 | 0.7051 | 0.3765 | 0.2028 | 0.2298 |
| v3 `fix_only` s86 (job 46457) | 0.4349 | 0.7546 | 0.4440 | 0.2718 | 0.3025 |
| v3 `bug_repro` s86, `--class-weight-order csv` (job 46458) | 0.4383 | 0.7550 | 0.4440 | 0.2862 | 0.3187 |
| v3 seed sweep s86 | 0.4385 | 0.7556 | 0.4447 | 0.2851 | 0.3146 |

Same caveats apply (no triplet-off control; recipe differs from Table 1). Notably `bug_repro` (inverted tail weights) is not worse on the tail than `fix_only`, so the class-weight bug does not obviously matter for these runs; this is one seed.

### 4. Does a true no-entropy ablation exist?
No. Searched:
- `/home/psytp7/logs/*.out` `args:` lines: 67 occurrences of `--triplet-lambda`, all `0.1`; no `--lambda-tri 0` anywhere.
- `train/checkpoint_Triplet_2/` and `train/checkpoint_Triplet_3/` (sweep2: bug_repro, combo, dp01, fix_only, freeze3, freeze4, llrd07, sched3; reg; res_sweep; seedsweep; markov; prior; push): none disables the triplet term or the gate.
- `scripts/submit_ablations.sh` (anchor axis {max, mean, all, random}; lambda axis {0.05, 0.1, 0.2}; entropy softmax vs binary) was written for `train_2_v2.py` but never launched: no `s2ab` job in `sacct` since 2026-07-01, no `ablate` directory anywhere in the repo. Its lambda axis does not include 0 in any case.
- `analysis/out/*`: only post-hoc ensemble/Markov analyses on cached probabilities; none isolates the gate.
- `train_2_v3.py` has no anchor-rule flag (the gate is hard-coded); `train_2_v2.py` has `--anchor-rule {mean,max,all,random}` and `--lambda-tri`.
- There is no Stage-2 run at 384 px, so a resolution-matched entropy row does not exist either.
- Existing partial evidence about Stage-1 dependence: the same recipe from the 0.377 Stage-1 gives 0.393 (jobs 43998-44002), +0.016 instead of +0.029 (per the rev4 numbers check). That shows the gain depends on the initial checkpoint, not that the triplet term causes it.

### 5. Minimal runs that would be needed (not launched)
Smallest decisive set, in priority order:
1. **Triplet-off control for the Table 1 row** (answers "is the +0.029 the method?"): the same recipe as jobs 43047-43051 with `--lambda-tri 0`, seeds 86, 123, 456, 789, 1024, 1 epoch, 512 px, `--class-weight-order csv`, bank 96, lr 3e-5, from `checkpoint3/Model_20260119_062652/model_best.pth`. Code caveat: the Aug-27 `train_2_v2.py` is not preserved; HEAD has a warmup LR schedule that would not reproduce the constant LR. Nearest is `train_2_v2.py.bak2`. About 2 h per run. Compare paired by seed against the 0.4108-0.4156 values.
2. **Triplet-off control for the post-fix v3 recipe** (needed for any claim about the current method): `train/train_2_v3.py` with `--triplet-lambda 0` and otherwise the `args:` line of job 46457 (fix_only), seeds 86, 123, 1024 at least (comparators already exist in `checkpoint_Triplet_3/seedsweep`, 0.4369 to 0.4434, six seeds). About 10 h per run.
3. **Gate-off at fixed lambda** (isolates entropy from the triplet term): `train_2_v2.py --anchor-rule random` (and `all`) with `--lambda-tri 0.1`, other args as item 1. v3 would need an added flag.
4. **Resolution-matched rows**: `evaluate.py --img-size 384` on the Stage-1 checkpoint (about 35 min), and Stage-2 entropy and the item-1 control trained with `--img-size 384` (v3 has the flag; v2 hard-codes `SIZE = 512`). MoE stays untraceable regardless; a re-implementation is not available (`convnext3.py` is not wired in).

Items 1 and 4 together give the table the paper needs: Stage 1, Stage 1 + same-recipe ASL only, Stage 1 + triplet, all at the same resolution. The entropy-vs-MoE comparison still rests on b12's single MoE number.

## Recommendation
- Keep Table 1 row 1 as 0.385 only with a footnote that it is the training-time 384 px value of the checkpoint (`history.json` best epoch 6), re-measured at 0.3688 / 0.3689 / 0.8643 / 0.1408 at 512 px with the same weights. Do not describe 0.385 as reproducible; a fresh retrain gives 0.377.
- Add the 512 px Stage-1 row (or state the resolution-matched gain +0.045) so that the entropy row and the baseline are at least evaluated at the same resolution. State that this still mixes in the fine-tuning epoch and the class weights.
- Replace "entropy beats MoE" with what the evidence shows: the entropy Stage 2 reaches a higher internal mAP/mAUC than the single published MoE number, at a different resolution, epoch budget and with an untraceable MoE row, so the comparison is indicative. Do not claim the gate or the triplet is responsible until run 1 or 2 above exists.
- Fix the calibration sentence in the paper (`paper.tex` line 211): "The calibration regression observed for both Stage-2 variants relative to Stage 1" is wrong for MoE on the paper's own numbers (mECE 0.130 vs Stage-1 0.134, i.e. better by 0.004). Only the entropy row regresses (+0.009 vs the 384 px Stage-1, but only +0.003 vs the 512 px Stage-1 at 0.1408), so part of the "regression" is a resolution artefact of the baseline.

## Caveats
- The 0.3849 vs 0.3688 gap is not decomposed; its explanation (resolution) is a hypothesis.
- January code state (positional-encoding class, init, preprocessing) is unverifiable. The July "buggy_posenc" runs and the August retrain differ in more than the positional-encoding class (the MoE plumbing in `convnext.py` was also removed on 2026-08-25), so I did not attribute the 0.298-0.340 vs 0.377 gap to the positional encoding.
- The v3 per-group table is one seed per row; per-group differences below about 0.01 should not be read.
- The Stage-2 b12 MoE epoch count ("additional 2 epochs") and Stage-1 "10 epochs" are b12's text; I could not check them against any artifact.

## Not measured
- Stage-1 original at 384 px with the current evaluator; Stage-2 anything at 384 px.
- Any triplet-off or gate-off run; any MoE run.
- Per-class (head/medium/tail) results for the paper's Table 1 row 3.
- Calibration (mECE) on a patient-disjoint split.

## Notable
- The paper's Stage-1 0.385 is a history-file value from a January run whose code and log no longer exist; the repo's own re-measurement of the same weights is 0.3688 at 512 px, and a retrain reaches only 0.377. The headline "+0.029 over Stage 1" is +0.045 against the resolution-matched number, but neither isolates the entropy gate.
- The paper's sentence that mECE regressed "for both Stage-2 variants" contradicts its own Table 1: MoE improved mECE (0.134 to 0.130).
- Original Stage-1 peak LR was 3.9e-5 (from `history.json`), not 4e-5.
- `history.json` is written only on improvement, so it ends at the best epoch; epochs after it (and final-epoch values) exist only in Slurm logs, and for the January run nowhere.

## Next step
1. Submit `scripts/eval_perclass.sh` for the Stage-1 checkpoint at 384 px (tag `res384`), about 35 min: settles the resolution explanation.
2. Decide whether the paper needs the triplet-off control (items 1 and 2 above) before keeping any claim about the entropy method itself. Item 1 is cheap (5 x about 2 h).
3. Orchestrator: consider updating memory `paper-revision-confounds` with the Stage-1 provenance (0.385 = history.json at 384 px, 0.3688 at 512 px, retrain 0.377) and the MoE-calibration contradiction.
