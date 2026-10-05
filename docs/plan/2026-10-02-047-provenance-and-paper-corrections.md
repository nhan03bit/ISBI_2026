# Provenance of the 0.4722 internal-val mAP, and paper corrections — 2026-10-02

Type: findings / provenance note (not a training-result report). Numbers re-read from artifacts on 2026-10-02.
Related results: [push wave](../result/2026-09-26-push-wave.md), [Markov A/B training](../result/2026-10-01-markov-ab-training.md), [v4 Markov result](../result/2026-09-27-v4-markov-result.md), [Markov A/B verdict](2026-10-01-markov-ab-verdict.md).

## TL;DR
- 0.4722 is a real full-fold evaluation (n_val 17,270, flip TTA) of a greedy 3-model ensemble of **v3** (entropy-gated triplet + memory bank) checkpoints. It is neither a smoke result nor a Markov-layer result.
- It is optimistic, because members were chosen on the same val fold they are scored on. The fixed 4-SWA flip average, with no selection, is 0.4699. The Markov paper's baseline is a base-model number (TTA + ensemble), so every Markov method sits at or below it.
- No v3 run isolates entropy gating or the triplet term, and the v3 checkpoints are not scored on the leaderboard. The number is attributable to the v3 pipeline as a whole.

## Question
The Markov paper (`docs/paper/markov/paper.tex`) reports 0.4722 internal-val mAP. Is it a real evaluation or a smoke result, and does it come from the Markov layer, the entropy-gated method, or an earlier version?

## Provenance of the 0.4722 members
Source: `analysis/out/ens_select_w01_flip.json` gives `ensemble_mAP` = 0.472237765789032 and `n_val` = 17270. All 10 candidate arrays plus `labels.npy` in `analysis/out/probs_w01_flip/` are (17270, 30), so this is the full internal validation fold. Producer: `analysis/ens_select.py` (forward greedy selection, uniform probability average, no repeats, stop when gain <= `--min-gain`, default 1e-4; the code asserts shapes match the labels).

| Member | Job / log | Single model, flip TTA | Non-TTA own-eval (log) | Recipe (from the log `args:` line) |
|---|---|---|---|---|
| `push/img768_dp01_seed42` SWA @768 | 49000, `isbi2026_v3push_img768_dp01_s42_49000.out` | 0.4650 | SWA 0.4600 (best-epoch 0.4585) | `train_2_v3.py`, 768 px, lr 1e-4, 4 epochs, dp 0.1, accum 12 x bs 4 |
| `push/img1024_dp01_seed86` SWA @1024 | 48997, `isbi2026_v3push_img1024_dp01_s86_48997.out` | 0.4645 | SWA 0.4619 (best-epoch 0.4602) | as above, 1024 px |
| `res_sweep/img640_seed86` best epoch @640 | 47094, `isbi2026_v3res_img640_s86_47094.out` | 0.4485 | best 0.4447 (SWA 0.4458 not used) | `train_2_v3.py`, 640 px, lr 1e-4, 8 epochs, patience 3, accum 12 x bs 4 |

All three `args:` lines contain `--metric-embed query --class-weight-order shard --head-classes 5 --memory-size 2048 --triplet-lambda 0.1` and `--stage1-ckpt checkpoint3/Model_20260119_062652/model_best.pth`. All three are therefore post-fix with respect to the class-weight order and triplet-embedding bugs.

Fixed-average reference, with no selection: 4 push SWA models (768 s42, 768 s1024, 896 s86, 1024 s86), flip, from `analysis/out/tta_w1_flip.json` = **0.4699** (job 49030, COMPLETED). mAUC 0.9154, mF1 0.4819, mECE 0.1385.

## What 0.4722 is / is not
Is:
- A real, full-fold (17,270 images), flip-TTA, uniform-average ensemble of three v3 models.
- Optimistic. Selection uses the same fold that is scored, and that fold also served as the early-stopping set. The unbiased-ish fixed average is 0.4699.

Is not:
- A smoke result. The `*_smoke` directories are separate 1-epoch OOM checks.
- A Markov-layer result. The members come from `train_2_v3.py`. The Markov layer is `train/train_2_v4.py` + `train/markov_layer.py` (jobs 49063-49065, finished 2026-09-27, checkpoints under `checkpoint_Triplet_3/markov/`). v4 SWA scores: mk1 0.4551, mk2 0.4566, mk1fix 0.4569, against the matched v3 baseline of 0.4600 SWA (job 49000, no TTA). All three are below it. (An extra variant, `mk1_lr100`, has SWA 0.4575 in `history.json`. It is also below 0.4600.)
- MoE. `train/convnext.py` (`ConvNeXt2`, imported by `train_2_v3.py`) contains no `aux_loss`. The `getattr(self.model, "last_aux_loss", None)` hook at `train_2_v3.py:861` therefore returns None and is inert. A `convnext3.py` exists at the repo root, not under `train/`, and no `.py` or `.sh` file imports it.
- The leaderboard number. It is internal-val only. The v3 checkpoints have not been submitted or scored on the test leaderboard.

Triplet mechanics of v3: `utils_update.sample_triplets_v13`, anchors are samples with softmax entropy >= the batch mean (`anchor_rule="mean"`). The paper says max. Hit rates in these three runs are:

| Run | Epoch 1 | Last epoch |
|---|---|---|
| 49000 | 82.3% | 79.5% (ep 4) |
| 48997 | 82.1% | 80.2% (ep 4) |
| 47094 | 81.4% | 74.0% (ep 6) |

The 512 px seed sweep (bs 8, e.g. job 46476) logs 96.6% at epoch 1.

## Entropy-paper (`docs/paper/entropy/paper.tex`) corrections list
Table 1 "Proposed 0.414 ± 0.002" is the v2 runs of 2026-08-27, jobs 43047-43051. Seeds 86, 123, 456, 789, 1024 give Best mAP 0.4139, 0.4129, 0.4108, 0.4156, 0.4155. The mean is 0.4137 and the sample SD is 0.0020. The log config line shows lr 3e-5, memory 96, epochs 1, patience 1, accum 6, batch 8, 512 px, so these are not the current code defaults (lr default is now 1e-4, epochs 10). Differences between what ran and what the paper describes:

| # | Paper says | Runs actually did | Evidence |
|---|---|---|---|
| 1 | Decoded query embeddings | Legacy post-ReLU pooled embedding (pre `--metric-embed query` fix) | memory `stage2-triplet-embedding-miswired`; no `--metric-embed` flag in `args:` |
| 2 | Intended class weights | CSV-order weights, near-inverted on the tail. Logs show `class_weights: [0.5, 0.6528, 0.5831, ...]` | jobs 43047-51, 44260, 44274-78 |
| 3 | Anchor = max entropy (Eq. 2) | Anchor = entropy >= batch mean (`--anchor-rule` default `mean`) | `train_2_v2.py:956` |
| 4 | Positive = more than 1 shared label (Eq. 3) | >= 1 shared label (`--pos-min-shared` default 1) | `train_2_v2.py:960` |
| 5 | "Hardest" positive/negative | Easy mining (closest positive / farthest negative; `--mining` default `easy`) | `train_2_v2.py:963` |
| 6 | "Only 7.6% of steps yield a triplet" | 7.6% is from job 44260 (accum 8, 30 epochs, epoch 1: 975/12913). Re-runs of the Table-1 recipe (jobs 44274-44278, 2026-09-01, bs 8, accum 6) log 13.1-14.0% at epoch 1 (13.2, 13.1, 14.0, 13.5, 13.5). Jobs 43047-51 logged no hit rate. | logs |
| 7 | Stage-1 / MoE vs proposed | Stage-1 and MoE rows evaluated at 384 px, proposed at 512 px (resolution confound) | memory `paper-revision-confounds` |
| 8 | Softmax vs per-class sigmoid entropy | Softmax entropy over the class axis (open confound) | `--entropy-mode` default `softmax` |

Items 1, 3, 4, 5, 7 and 8 come from code defaults and the memory notes. I re-confirmed 2, 6 and the Table-1 config directly from the logs. v2 jobs 43047-51 predate the `--class-weight-order` flag, so they count as buggy tail weighting.

## v3 progression (internal val, best-epoch EMA unless SWA stated)
| Step | mAP | Source |
|---|---|---|
| v2 Table 1, 512 px, 5 seeds (buggy weights etc.) | 0.4137 (± 0.0020) | jobs 43047-51 |
| v3 512 px seed sweep, 6 seeds | 0.4394 ± 0.0025 sample SD (0.0022 population); range 0.4369-0.4434 | `seedsweep/*/history.json` |
| v3 768 px seed 86 (res_sweep) | 0.4560 best / 0.4566 SWA | `res_sweep/img768_seed86` |
| v3 768 px dp 0.1 SWA, seed 42 / seed 1024 | 0.4600 / 0.4582 | jobs 49000 / 48999 |
| v3 1024 px dp 0.1 SWA, seed 86 | 0.4619 | job 48997 |
| + flip TTA (768 s42) | 0.4650 | `ens_select_w01_flip.json` |
| fixed 4-SWA average, flip | 0.4699 | `tta_w1_flip.json` |
| greedy-3 ensemble, flip (selected on val) | 0.4722 | `ens_select_w01_flip.json` |

Approximate decomposition, with each gain computed from the rows above. Several steps compare different seeds, so each is indicative only:
- v2 to v3 (bug fixes + recipe) at 512 px: about +0.025 (0.4137 to 0.4394, mean to mean).
- 512 to 768 px: about +0.017 (seed 86: 0.4385 to 0.4560).
- Drop-path + SWA: about +0.003 to +0.005.
- Flip TTA: about +0.005 (0.4600 to 0.4650).
- Ensemble: about +0.007 (0.4650 to 0.4722). Part of this is selection on val.

## Attribution limits
- No ablation isolates entropy gating or the triplet term in v3. There is no `--triplet-lambda 0` run under `checkpoint_Triplet_3`, and the planned v2 ablation set (`scripts/submit_ablations.sh`) has no outputs: `train/checkpoint_Triplet_2/ablate` does not exist. The v2->v3 gain bundles the bug fixes, memory size (96 -> 2048), lr (3e-5 -> 1e-4), EMA, positive rule, resolution and epochs. It cannot be credited to entropy gating.
- The internal val fold is not patient-disjoint: 77.0% of val images share a patient with training, and 30% (5,234 of 17,270) are lateral views. The test set is frontal and patient-disjoint. Per Markov paper Table "subsets", on frontal unseen-patient images (n = 3,572) the 1024 vs 512 px gain shrinks from +0.020 to +0.007 (interval includes 0). The greedy-3 ensemble scores 0.4968 on frontal images and 0.4752 on frontal unseen-patient images, against 0.4722 on the full fold. Internal numbers should not be read as leaderboard predictions.
- Separately: `prior/img768_qprior_seed{42,1024}` (v5 B-early) have SWA 0.4807 / 0.4786 (re-read from `history.json`), above 0.4722. They use report-derived patient history that is unavailable at test time and are not part of the 0.4722 claim. With the prior set to zero, B-early is no better than the baseline (Markov paper).

## Paper edits made on 2026-10-02
- `docs/paper/entropy/paper.tex`: modified (mtime 17:10, 15 diff lines vs `paper.tex.pre-provenance.bak`, created 17:08). The edits add the disclosure of the v2 implementation vs method differences, the ~13% utilisation, and a v3 section.
- `docs/paper/markov/paper.tex`: edited after this note was first written (see Discrepancies, item 2, now resolved). The model description names v3 (entropy-gated triplet, no MoE, no Markov) and cites the entropy paper. The post-hoc setup defines the greedy selection, its selection bias and the 0.4699 fixed-average reference. The trainable-layer setup and Table III caption state that the single-model no-TTA numbers cannot be compared with 0.4722. The smoothing and gating captions mark the base as having no Markov component. A Discussion paragraph, "Where the 0.47 comes from", gives the gain breakdown and the missing gate ablation, and the abstract and conclusion each carry one clause on the 0.472 v3 baseline. The PDF was rebuilt with no LaTeX errors or undefined references.
- Entropy-paper follow-up: the v3 utilisation range was corrected to "74-82% at 640-1024 px, 97% at 512 px" per item 3 below.

## Discrepancies
Every numeric claim 1-7 reproduced from the artifacts. The differences are in framing and state:

1. **Path of `convnext3.py`.** Claim 2 says `train/convnext3.py`. The file is at the repo root (`/home/psytp7/ISBI_2026/convnext3.py`), not under `train/`. It is still not imported by any `.py` or `.sh` file. `train_2_v3.py` does contain an inert MoE aux-loss hook (`getattr(model, "last_aux_loss", None)`). The conclusion (no MoE in the 0.4722 members) stands.
2. **Markov paper edit not present at verification time.** The brief says both papers were corrected on 2026-10-02. At verification time `docs/paper/markov/paper.tex` was byte-identical to `paper.tex.pre-provenance.bak` (0 diff lines; mtime 2026-10-01 21:42), so no provenance edit had landed there. Its text (line 307) describes the base as "our best greedy three-model ensemble" and does not state that the ensemble was selected on the scored fold, nor that members are v3. Only the entropy paper differs from its backup. Re-check after the orchestrator's edit. **Resolved 2026-10-02 17:12:** both papers now differ from their backups and compile cleanly.
3. **Triplet hit rates are a bit lower than "~81-82%" in later epochs.** Epoch 1 is 81.4-82.3% for the 640/768/1024 px runs, but the last-epoch values are 79.5% (49000), 80.2% (48997) and 74.0% (47094, epoch 6). The 96.6% figure for the 512 px bs-8 sweep is confirmed (job 46476, epoch 1; 96.3% at epoch 2).
4. **The 0.4485 for the 640 px member is the flip-TTA single-model value.** Its non-TTA best epoch is 0.4447. "Best epoch" refers to the checkpoint, not the metric.
5. **Minor: "0.4394 ± 0.0025"** is the sample SD (n-1) over 6 seeds. The population SD is 0.0022.

## Not measured
- Leaderboard score of any v3 checkpoint or ensemble.
- Triplet-off or gating-off v3 controls.
- Nested or held-out ensemble selection. The 0.4699 fixed average is the closest unbiased reference, and it is still scored on the early-stopping fold.
- Per-class head/medium/tail breakdown of the ensemble from `class_groups.py`. Only the Markov paper's HMM analysis reports it, and that is for a different method.

## Notable
- 0.4722 does not support any claim about Markov layers, and by itself it does not support any claim about entropy gating.
- A leaderboard-comparable estimate would be the frontal, unseen-patient subset (0.4752 for the greedy-3 ensemble). Even that is still selected on the val fold.

## Next steps
- Re-check the Markov paper once the provenance edit is applied. The ensemble should be called selected-on-val, with the fixed 4-SWA 0.4699 alongside it.
- Run a v3 `--triplet-lambda 0` control at 768 px with the same seed and recipe if an entropy-gating claim is to be made.
- Score the v3 ensemble on the leaderboard once, so the internal-to-leaderboard offset is known for v3.

## Follow-up edit (2026-10-02, after review)
The Markov paper still read as if it claimed 0.47, because 0.4722 appeared in bold as "base" or "Greedy-3 ensemble". Changes:
- Every 0.4722 row (Tables smoothing, gating, subsets) is relabelled "entropy-gated v3 baseline [b25], no Markov" and un-bolded.
- The table captions say the number is the companion (entropy) paper's result.
- The abstract no longer quotes any absolute mAP.
- The Discussion and Conclusion say 0.472 is the entropy paper's result.
- The HMM emission E and the metadata base are labelled as v3 with no Markov component.
- The entropy paper now claims the 0.4722 as its own result and notes that the Markov study uses it unchanged as its baseline.

Both PDFs rebuild with 0 errors, 0 undefined references and 0 overfull boxes.
