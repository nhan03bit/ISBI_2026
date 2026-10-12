# Plan: one unified benchmark (768 px, 5 seeds, all 7 metrics) for triplet bank, entropy gate, entropy + Markov prior

## Context
The 512 px ablation (6 seeds) found no effect of the triplet/bank or of the entropy gate; the other rows of Table 1 (768 px, MoE) come from different code paths, 2 seeds and lack head/medium/tail AP, so they cannot show what the components add. The user wants ONE unified setting in which every arm has the same recipe, code path, seeds and all seven metrics (mAP, mAUC, mF1, mECE, head, medium, tail AP), covering: triplet+bank, entropy gate, entropy + Markov prior, and a search for a setting where the triplet term actually bites.

Honesty constraint (stated up front, also for the paper): the experiment measures the effects, it cannot guarantee positive ones. Arms and contrasts are fixed here before any run; every arm is reported whatever it shows; the stronger-triplet arm is exploratory. The Markov prior is expected to help when history is available (earlier +0.018 mAP, zero when withheld); the earlier diagnostics suggest the triplet/gate effects may stay inside the noise unless the stronger-triplet arm changes that.

User decisions (2026-10-12): unified setting = 768 px, 5 seeds; add pre-declared arm F (hard mining, lambda 1); cancel the whole queued resolution sweep.

## Arms (all on `train/train_2_v5.py`, push recipe, 768 px, seeds 42/123/456/789/1024)
Common args = `submit_stage2_v5_prior.sh` recipe: `--accum-steps 12 --batch-size 4 --monitor mAP --memory-size 2048 --metric-embed query --head-classes 5 --class-weight-order shard --swa-last-k 3 --lr 1e-4 --epochs 4 --patience 2 --sched-epochs 8 --drop-path-rate 0.1 --img-size 768 --stage1-ckpt checkpoint3/Model_20260119_062652/model_best.pth`, `-p amp48,ada24 -x colossus`, `--mem 96G --time 36:00:00`. PRIOR = `--prior-file analysis/out/patient_markov/priors.pt --prior-dropout 0.3 --prior-lr-mult 10`.

| Arm | Meaning | Flags (on top of common) | Existing runs reused | New runs |
|---|---|---|---|---|
| A | ASL only | `--no-triplet` | s42, s1024 (`checkpoint_MoE/asl_img768_seed*`) | s123 s456 s789 |
| B | triplet + bank, all anchors | `--triplet-lambda 0.1 --anchor-rule all` | – | 5 |
| C | entropy gate | `--triplet-lambda 0.1 --anchor-rule mean` | – (v3 runs are another code path) | 5 |
| D | entropy gate + Markov prior | C + PRIOR | s42, s1024 (`prior/img768_qprior_seed*`; the `_rerun` ones kept as extra replicates) | s123 s456 s789 |
| E | ASL + Markov prior | `--no-triplet` + PRIOR | – | 5 |
| F | stronger triplet (exploratory) | `--triplet-lambda 1.0 --anchor-rule mean --mining hard` | – | 5 |

26 new runs × ~10–14 h ≈ 310 GPU-h; amp48/ada24 caps give ≈ 8 concurrent GPUs → about 2.5–3.5 days. Existing MoE / MoE+prior (2 seeds) stay as a reference row, not part of the paired contrasts.

## Code changes (small, default behaviour unchanged, verified bit-identical)
1. `train/utils_update.py::sample_triplets_v13`: new keyword `mining="easy"`; replace the two hard-coded `argmin/argmax` lines by `_mine_pair(pos_d, neg_d, margin, mining)` (already exists, supports easy/hard/semihard). CPU unit check on random logits/labels/bank: `mining="easy"` returns exactly the old triplets; `"hard"` returns farthest positive / closest negative.
2. `train/train_2_v5.py`: add `--anchor-rule {mean,all,random}` (default mean) and `--mining {easy,hard,semihard}` (default easy), pass them (plus a CPU generator seeded `seed+1` for "random" and a `stats` dict) to `sample_triplets_v13`, log `[Epoch e] anchors/step … triplets/step …` and store `history["triplet_stats"]`, exactly as `train_2_v3.py` does. `python -c "import utils_update"` and `train_2_v5.py --help` must still work (v4/v5 callers).
3. New `scripts/submit_stage2_v5_unified.sh` (modelled on `submit_stage2_v5_prior.sh` and `submit_stage2_v3_entropy_ablation.sh`): run specs `arm:seed`, out dir `train/checkpoint_Triplet_3/unified768/<arm>/seed_<S>/Model_run`, job `isbi2026_uni_<arm>_s<seed>`, `SMOKE=1` support, and an afterok evaluation chain per run.
4. Evaluation per run: `scripts/evaluate_tta_v5.sh --batch-size 4 --dump-probs analysis/out/probs_unified_flip --out-json … <ckpt>/model_swa.pth@768` (flip TTA, SWA weights, the protocol of the existing MoE rows), with `--prior-file … --prior-mode with` and again `zero` for arms D and E (with = history available, as in a hospital archive; zero = the challenge-test condition). The dump dir is seeded with `keys.txt`/`labels.npy` like `submit_eval_moe.sh`. Existing seed 42/1024 checkpoints of A and D are re-dumped there so all rows share one provenance. Eval jobs `--time 4:00:00 -x tamar`.

## Analysis and paper
- New `analysis/unified_benchmark.py` (reuse `analysis/entropy_ablation.py`: `tm_metrics`, `group_means`, `paired`, `summarize`, `fmt`, class groups): all 7 metrics per (arm, seed, mode) from the dumped probabilities; asserts labels/keys identical; per-arm mean ± s.d.
- Pre-registered contrasts, paired by seed (n = 5; mean difference, 95% t-CI, exact sign-flip p, wins, MDE), primary metric mAP, all 7 metrics reported, Holm-adjusted p for the 7 mAP contrasts: B−A (triplet+bank), C−B (entropy gate), C−A, D−C (Markov prior on the entropy model, with history), D−C with prior withheld (test condition), D−E (entropy given the prior), E−A (prior alone), F−C (stronger triplet; exploratory).
- Outputs `analysis/out/unified_benchmark.json`, `docs/paper/isbi_submit/generated/unified_{macros,table}.tex` (table file without a closing `\\`, no `\multicolumn` right after `\input`).
- Paper: the unified table replaces the current Table 1 (rows A, B, C, D with/withheld, E with/withheld, F + MoE reference rows, all seven metric columns filled); the 512 px six-seed ablation moves to one sentence plus its macros; Method §2.5, Results, Discussion, Conclusion and abstract are rewritten from whatever the unified results show (no pre-judged wording); keep 4 pages and `check_submission.py` READY.
- Report via the `cxrlt-research-analyst` subagent → `docs/result/<date>-unified-768.md`; save this plan as `docs/plan/2026-10-12-unified-benchmark-plan.md` (pre-registration); update memory notes.

## Execution order
1. Read-only checks done; save plan copy. 2. `scancel` the 24-run resolution sweep (training 51778…51824 even, evals odd; 4 are running, 44 pending) and the held smoke leftovers. 3. Apply code changes 1–2, run CPU unit checks. 4. `SMOKE=1` for arms B, D, E, F at 768 px (40 batches): look for `anchors/step` > 0, `mining=hard` in the `v3:`-style log line, `prior:` lines, `no-triplet` for E, no Traceback. 5. Submit all 26 runs + eval chains; record job IDs in memory. 6. Write `unified_benchmark.py` and test it on a mock tree while jobs run. 7. When complete: run analysis, delegate the report, rewrite the paper sections.

## Verification
- Unit: bit-identity of `mining="easy"`; `anchor_rule="all"` gates all samples; v5 default path still equals old behaviour (compare first-step triplets on a fixed batch).
- Smoke logs per arm; after training `history.json` best_metric vs the evaluator's SWA mAP are different quantities (SWA, flip TTA), so check instead that the evaluator regression holds: re-dump seed-42 arm D and compare with the existing `ev_v5_orig_with` value (0.4797/0.4867-style numbers) within 1e-4.
- `latexmk -g -pdf paper.tex` + `python3 -I check_submission.py --allow-author` = READY, 4 pages.

## Risks
- Compute/queue: ~3 days if the cap holds; if by 19 Oct fewer than 3 seeds per arm are done, report the seeds available and say so. Prioritise seeds 42/123/456 across all arms first (submission order), seeds 789/1024 last.
- Arm F may be unstable (hard mining on report-derived labels); it stays exploratory.
- D vs C is not a like-for-like comparison at test time (history absent); the paper must show the withheld row next to the with-history row.
- Possible outcome: triplet bank and entropy gate remain inside the noise; then the paper keeps its null result and reports the Markov prior as the only measurable add-on.

## Addendum 2026-10-12 (before any unified run finished): the three systems of the paper, arm M
The paper's Table 1 is restricted to the three systems asked for: **M** sparse MoE (`--arch moe --no-triplet`, 4 experts top-2, aux 5e-3; state-of-the-art reference), **C** Triplet Memory + Entropy, **D** Triplet Memory + Entropy + Markov prior (history available and withheld). A, B, E, F stay as controls (`unified_controls_table.tex`, not in the paper unless space allows). Arm M seeds 42/1024 are the earlier `moe_img768` runs (dumps in `probs_moe_flip`), seeds 123/456/789 are new (jobs 51897/51899/51901, evals 51898/51900/51902). Added contrasts (same rules, Holm family extended): C−M, D(with)−M, D(withheld)−M. Pending control jobs were given `Nice=500` so that the C, D and M runs start first. The paper carries no result until all five seeds of M, C, D exist (`\TBD` placeholders; `check_submission.py` now requires `generated/unified_table.tex`). The earlier six-seed 512 px ablation (no gate effect, C−B −0.0014 [−0.0048,+0.0021]) stays in the record (`docs/plan/2026-10-12-why-entropy-shows-no-gain.md`); the paper discloses it in one sentence in the Discussion and must not omit it from the final text.
