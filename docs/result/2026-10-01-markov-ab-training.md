# Markov A-vs-B training comparison (B-early, A-strong) — 2026-10-01

**Directory:** train/checkpoint_Triplet_3/prior/img768_qprior_seed{42,1024}/Model_run/ and train/checkpoint_Triplet_3/markov/img768_mk1_lr100_seed42/Model_run/
**Script / recipe:** train_2_v5.py (B-early, label-query prior conditioning), train_2_v4.py (A-strong, `--markov-lr-mult 100`); 768 px push-wave recipe (49000), 4 of 8 cosine epochs, SWA last 3
**Slurm jobs:** 49375 (B-early s42), 49376 (B-early s1024), 49377 (A-strong s42); smoke runs 49373, 49374. All three: COMPLETED, exit 0:0 (sacct; elapsed 11:35:37, 11:12:20, 11:35:55). Eval dumps 49404/49405/49406; CPU comparison 49442 (COMPLETED).
**Logs:** /home/psytp7/logs/isbi2026_v5prior_qprior_s42_49375.out, ..._s1024_49376.out, isbi2026_v4markov_mk1_lr100_s42_49377.out, isbi2026_ev_ab_*_4940[4-6].out
**Companion verdict (decision rule, interpretation):** [docs/plan/2026-10-01-markov-ab-verdict.md](../plan/2026-10-01-markov-ab-verdict.md). Plan: [docs/plan/2026-09-30-markov-ab-training-plan.md](../plan/2026-09-30-markov-ab-training-plan.md). Earlier v4 runs: [2026-09-27-v4-markov-result.md](2026-09-27-v4-markov-result.md).

All metrics are INTERNAL val (17,270 rows), not leaderboard.

## Configuration (from each log's `args:` line)
| Setting | Value |
|---|---|
| Stage-1 checkpoint | checkpoint3/Model_20260119_062652/model_best.pth (all runs) |
| Image size | 768 |
| Seeds | 42, 1024 (B-early); 42 (A-strong) |
| LR / batch / accum | 1e-4 / 4 / 12; epochs 4 of sched-epochs 8, patience 2, drop-path 0.1, swa-last-k 3 |
| metric-embed / memory-size / triplet-lambda | query / 2048 / 0.1 (all runs) |
| class-weight-order | shard (fixed) in all three runs and in v3 49000 |
| B-early specifics | `--prior-file analysis/out/patient_markov/priors.pt --prior-dropout 0.3 --prior-lr-mult 10` |
| A-strong specifics | `--markov-steps 1 --markov-lr-mult 100` |

## Training-log results (Done. Best / SWA lines)
| Run | Best epoch | Best mAP | SWA mAP | SWA mAUC | SWA mF1 | SWA mECE |
|---|---|---|---|---|---|---|
| B-early s42 (49375) | 3 | 0.4793 | 0.4807 | 0.9151 | 0.4863 | 0.1342 |
| B-early s1024 (49376) | 2 (early stop triggered at epoch 4) | 0.4760 | 0.4786 | 0.9141 | 0.4903 | 0.1324 |
| A-strong mk1 lr100 s42 (49377) | 3 | 0.4576 | 0.4575 | 0.9110 | 0.4667 | 0.1354 |
| v3 baseline s42 (49000) | 3 | 0.4585 | 0.4600 | 0.9092 | 0.4685 | 0.1354 |
| v3 baseline s1024 (48999) | 3 | 0.4564 | 0.4582 | 0.9104 | 0.4741 | 0.1354 |

Caveat: the B-early log numbers are computed WITH patient-history priors for val (train ∪ val history), i.e. a longitudinal simulation (see leaderboard rule below). They are not comparable to v3 as a deployable score. No discrepancy between history and log was checked beyond reading the log lines (log lines are the source here).

Training diagnostics (log): B-early conditioner grows each epoch, s42 |u_c| mean 1.28 -> 1.61 -> 1.80 -> 1.90, |W| fro 4.64 -> 7.51; batch has_prior 0.22 (train fold; val has-prior fraction is 5759/17270 = 0.333, derived). A-strong gates stay small: |gate| mean 0.0645, max 0.302 (Mass +0.302) at last epoch, vs 0.0594 at epoch 3.

## Eval dumps (flip TTA, SWA weights, 768 px) — ENSEMBLE metrics
Row alignment checks (analysis/out/markov_ab/run.log [0]): keys.txt == reconstructed analysis/out/val_key_order.npy: **True** (17,270 rows); re-dumped v3 s42 SWA probs vs analysis/out/probs_w01_flip: **max |d| = 0.0**. This verifies, for the first time, the reconstructed row order used in all earlier post-hoc work.

| Job | Models in ensemble | mAP | mAUC | mF1 | mECE |
|---|---|---|---|---|---|
| 49404 | v3 s42 + s1024 + v4 mk1, mk2, mk1fix, mk1_lr100 (6) | 0.4662 | 0.9133 | 0.4728 | 0.1353 |
| 49405 | B-early s42 + s1024, priors "with" (2) | 0.4867 | 0.9170 | 0.4980 | 0.1332 |
| 49406 | B-early s42 + s1024, priors zeroed (2) | 0.4693 | 0.9131 | 0.4811 | 0.1358 |

These are ensembles, not per-model values. Per-model numbers below come from markov_ab_compare (job 49442).

## Per-model macro mAP by subset (flip TTA, SWA; analysis/out/markov_ab/run.log [1])
d = arm − v3 of the same seed; boot = paired patient-clustered bootstrap (300), 95% CI.

Seed 42:
| Arm | all | frontal | has_prior | has_prior & frontal (PRIMARY) | has_prior & study-not-in-train | no_prior |
|---|---|---|---|---|---|---|
| v3 | 0.4650 | 0.4875 | 0.4894 | 0.4909 | 0.4893 | 0.4529 |
| A mk1 | 0.4587 | 0.4791 | 0.4853 | 0.4839 | 0.4749 | 0.4477 |
| A mk1fix | 0.4606 | 0.4818 | 0.4857 | 0.4880 | 0.4844 | 0.4526 |
| A mk2 | 0.4622 | 0.4846 | 0.4840 | 0.4801 | 0.4780 | 0.4508 |
| A-strong lr100 | 0.4616 | 0.4903 | 0.4831 | 0.4801 | 0.4814 | 0.4526 |
| B-early | 0.4839 | 0.5055 | 0.5276 | 0.5179 | 0.5284 | 0.4562 |
| B-early0 (zero prior) | 0.4624 | 0.4858 | 0.4811 | 0.4783 | 0.4734 | 0.4562 |
| B-late w1 | 0.4721 | 0.4904 | 0.5275 | 0.5278 | 0.5457 | 0.4529 |
| B-late CV-w | 0.4882 | 0.5077 | 0.5454 | 0.5402 | 0.5538 | 0.4529 |

Seed 1024 (v3: all 0.4626, frontal 0.4862, has_prior 0.4807, primary 0.4791, study-not-in-train 0.4779, no_prior 0.4534):
| Arm | all | frontal | has_prior | PRIMARY | study-not-in-train | no_prior |
|---|---|---|---|---|---|---|
| B-early | 0.4820 | 0.5055 | 0.5265 | 0.5171 | 0.5287 | 0.4548 |
| B-early0 | 0.4663 | 0.4920 | 0.4830 | 0.4838 | 0.4807 | 0.4548 |
| B-late w1 | 0.4712 | 0.4888 | 0.5289 | 0.5259 | 0.5441 | 0.4534 |
| B-late CV-w | 0.4855 | 0.5040 | 0.5422 | 0.5354 | 0.5477 | 0.4534 |

Paired Δ vs v3 on PRIMARY (bootstrap CI): B-early s42 +0.0270 (+0.0081, +0.0497), s1024 +0.0380 (+0.0265, +0.0547); B-late CV-w s42 +0.0493 (+0.0321, +0.0715), s1024 +0.0564 (+0.0359, +0.0788); B-late w1 s42 +0.0370, s1024 +0.0468; A mk1 -0.0070, mk2 -0.0108, mk1fix -0.0029, A-strong -0.0108 (CI -0.0320, +0.0112), all CIs include 0. Full-val Δ: A arms -0.0028 to -0.0063 (all CIs include 0); A-strong frontal +0.0028 (boot +0.0009 [-0.0118, +0.0154]). B-early0 vs v3 on all: s42 -0.0026 (CI -0.0109, +0.0075), s1024 +0.0037 (CI -0.0029, +0.0106). Patient-unseen subset has n = 80 and is descriptive only; every CI spans 0.

B-late CV-w pools its two patient halves; this is safe here only because both halves selected w = 0.5 at both seeds (run.log: "w chosen [0.5, 0.5]" for s42 and s1024), so the transform applied is identical across halves. A separate redesign run found that pooling halves fitted with different transforms creates artefacts.

## Head / medium / tail ΔAP vs v3 (PRIMARY subset; run.log [4]; classes: head 3, medium 20, tail 7 per class_groups in that script)
| Arm | s | head | medium | tail |
|---|---|---|---|---|
| A-strong lr100 | 42 | +0.0003 | +0.0005 | -0.0479 (1/7 up) |
| A mk1 / mk1fix / mk2 | 42 | -0.0016 / +0.0003 / -0.0018 | +0.0026 / 0.0000 / +0.0025 | -0.0365 / -0.0126 / -0.0529 |
| B-early | 42 | +0.0175 (3/3) | +0.0360 (20/20) | +0.0055 (5/7) |
| B-early | 1024 | +0.0122 | +0.0361 (18/20) | +0.0546 (6/7) |
| B-early0 | 42 / 1024 | +0.0014 / -0.0044 | -0.0033 / -0.0029 | -0.0451 / +0.0302 |
| B-late CV-w | 42 | +0.0130 | +0.0398 | +0.0918 (6/7) |
| B-late CV-w | 1024 | +0.0133 | +0.0388 | +0.1250 (6/7) |
| B-late w1 | 42 / 1024 | -0.0008 / 0.0000 | +0.0286 / +0.0284 | +0.0770 / +0.1197 |

The head/medium/tail split in markov_ab_compare is its own (3/20/7) grouping; I did not cross-check it against evaluate/class_groups.py in this report (tail n = 7, so tail deltas are noisy).

## Comparison to baselines
- vs v3 baseline (SWA 0.4600 s42 / 0.4582 s1024, log): B-early SWA 0.4807 / 0.4786 (+0.0207 / +0.0204, derived), but only with val patient-history priors. With zeroed priors the ensemble is 0.4693 vs 6-model v3+v4 ensemble 0.4662 (different ensemble compositions, not like for like).
- A-strong SWA 0.4575 vs v3 0.4600: -0.0025 (derived). No gain from 10x larger Markov LR multiplier (tail -0.048 on the primary subset).
- vs earlier v4 Markov arms (49063-49065): see 2026-09-27-v4-markov-result.md; A-strong is in the same band as mk1/mk2/mk1fix.
- vs campaign best (memory: 0.4602, SWA 0.4619 @1024 px, job 48997): not comparable at 768 px / not re-evaluated here.

## Caveats / known-bug flags
- class-weight-order shard and metric-embed query confirmed in all three new logs and in 49000 (not pre-fix).
- **Leaderboard rule:** B-early and B-late use patient history (val priors from train ∪ val). That history does not exist for leaderboard test images, and dev/test must never be joined to the PadChest CSV. Leaderboard effect of B-early (with priors) and B-late is therefore 0; their internal numbers are a longitudinal simulation only. Leaderboard-usable arms: A arms and B-early0 (zero prior), all within noise of v3 (full-val Δ -0.0063 to +0.0037).
- Training-time priors use train-fold history only; val priors use train ∪ val (per plan). The val split is not patient-disjoint (memory: 77% of val images share a patient with train), which is exactly what the prior exploits.
- B-early0 s1024 and s42 differ in sign vs v3, i.e. zero-prior behaviour is within noise.
- Triplet hit rate printed in v5 logs is ~80-81% per epoch; this differs from the paper's 7.6% utilization figure and is another open instance of the 7.6%-vs-13% discrepancy noted in paper-revision-confounds (the quantities may be defined differently; not resolved here).
- A-strong has a single seed; seed-replication of the decision rule is n/a for A.
- Early stopping at epoch 4 for B-early s1024 (best epoch 2).

## Notable
- Row order verified: keys.txt == val_key_order.npy and v3 s42 SWA probs reproduce probs_w01_flip with max |d| = 0.0. All earlier post-hoc work built on the reconstructed order is thereby validated.
- Increasing the Markov LR multiplier to 100 does not rescue A (SWA 0.4575 < v3 0.4600).
- Feeding the prior into the network (B-early) gained +0.0207 / +0.0204 SWA mAP over v3 in simulation, but the simple post-hoc late fusion (B-late CV-w, no retraining) is as good or better (see verdict).

## Next step
- Per-class eval_result via the standard eval_perclass flow was not run for these checkpoints; per-class numbers above come from markov_ab_compare only.
- A leaderboard-usable improvement needs an approach that does not require patient history.
