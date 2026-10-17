# Unified-test Table 1 status (preliminary, n=1-2 seeds) - 2026-10-12 ~20:30

**Question:** do current results show an improvement of Triplet Memory + Entropy (C) and Entropy + Markov prior (D, history available / withheld) over the MoE baseline (M)?
**Short answer:** not established. No contrast has more than 2 seeds; the only visible effect is the history-available Markov prior, which is the leakage-prone condition. Entropy alone (C) is level with or below MoE.
Plan / pre-registration: `docs/plan/2026-10-12-unified-benchmark-plan.md`. Prior null result: `docs/plan/2026-10-12-why-entropy-shows-no-gain.md`.

## 0. Correction to the arm mapping you were given
From `analysis/unified_benchmark.py` and `scripts/submit_stage2_v5_unified.sh`:

| Arm | System | Flags |
|---|---|---|
| A | ASL only | `--no-triplet` |
| B | triplet + bank, every image an anchor | `--anchor-rule all` |
| **C** | **Triplet Memory + Entropy** (Table 1 row 2) | `--anchor-rule mean`, lambda 0.1 |
| **D** | **C + Markov prior** (Table 1 rows 3 and 4) | C + prior; scored with / zero |
| E | ASL + Markov prior (control) | `--no-triplet` + prior |
| F | stronger triplet, hard mining, lambda 1 (exploratory) | |
| **M** | **MoE** (Table 1 row 1) | `--arch moe --no-triplet`; s42/1024 are the earlier `moe_img768` runs in `probs_moe_flip` |

So in the seed-42 JSONs in `probs_unified_flip`: **a = ASL only, not MoE** (the file lives under `checkpoint_MoE/asl_img768_seed42`, which is the source of the confusion); b = no gate; c = entropy gate (system C); **e = ASL + prior, not entropy + prior**; f = exploratory. Arm D has no dump in `probs_unified_flip` yet (its s42/s1024 are older runs, re-dump pending); I read them read-only from `probs_ab_flip` (same `keys.txt`/`labels.npy`, verified identical). MoE s42 is in `probs_moe_flip`, not in the files you listed.

Settings common to all: 768 px, SWA weights, flip TTA, same row order, N=17270 val images (patient-leaky split), shard class-weight order, `--metric-embed query` (per plan recipe; the submit script is the source, I did not re-check the Slurm `args:` lines).

## 1. What exists now (seed-level, all from saved probs with `entropy_ablation.tm_metrics`/`class_groups`)
| System | seed | mAP | mAUC | mF1 | mECE | head | med | tail |
|---|---|---|---|---|---|---|---|---|
| M MoE | 42 | 0.4612 | 0.9102 | 0.4776 | 0.1318 | 0.7631 | 0.4612 | 0.3320 |
| M MoE | 1024 | 0.4630 | 0.9117 | 0.4716 | 0.1319 | 0.7639 | 0.4628 | 0.3349 |
| C Triplet + Entropy | 42 | 0.4605 | 0.9124 | 0.4670 | 0.1346 | 0.7645 | 0.4621 | 0.3257 |
| D with history | 42 | 0.4839 | 0.9162 | 0.4926 | 0.1341 | 0.7723 | 0.4799 | 0.3716 |
| D with history | 1024 | 0.4820 | 0.9153 | 0.4932 | 0.1324 | 0.7714 | 0.4807 | 0.3618 |
| D history withheld | 42 | 0.4624 | 0.9121 | 0.4742 | 0.1366 | 0.7638 | 0.4591 | 0.3423 |
| D history withheld | 1024 | 0.4663 | 0.9114 | 0.4834 | 0.1350 | 0.7628 | 0.4595 | 0.3585 |
| A ASL only (control) | 42 / 1024 | 0.4600 / 0.4599 | 0.9123 / 0.9110 | 0.4766 / 0.4728 | 0.1330 / 0.1326 | 0.7625 / 0.7632 | 0.4609 / 0.4600 | 0.3279 / 0.3293 |
| B no gate (control) | 42 | 0.4637 | 0.9116 | 0.4784 | 0.1351 | 0.7643 | 0.4620 | 0.3400 |
| E ASL + prior with / zero (control) | 42 | 0.4799 / 0.4590 | 0.9156 / 0.9117 | 0.4906 / 0.4718 | 0.1309 / 0.1335 | 0.7713 / 0.7629 | 0.4796 / 0.4584 | 0.3560 / 0.3303 |
| F hard mining, lambda 1 (control) | 42 | 0.4056 | 0.8853 | 0.4135 | 0.1559 | 0.7501 | 0.4286 | 0.1923 |

Completed seeds per Table 1 row: M 2/5 (42, 1024), C 1/5 (42; 1024 running), D 2/5 (42, 1024; the 123/456/789 runs are in training). The aggregate JSONs for a, b, c, e, f in `probs_unified_flip` match my recomputation from probs (mAP/mAUC/mF1/mECE agree to 4 dp). The M numbers differ from your a-row by design (a is ASL, not MoE).
Training logs (non-TTA, SWA) for context: uni_b_s42 0.4618, uni_c_s42 0.4578, uni_e_s42 0.4780, uni_f_s42 0.3997. Running seeds at epoch 2-3 of 4 show c_s123 0.4569, c_s456 0.4506, d_s123 0.4749 (best-epoch so far, not SWA, not TTA).
A preliminary table from `python3 -I analysis/unified_benchmark.py --allow-missing` was written to `analysis/out/unified_benchmark.json` and siblings (analysis/out only, nothing copied to `docs/paper/isbi_submit/generated/`; no tracked files changed).

## 2. Paired differences vs MoE (seed 42 only; n=1)
| Contrast (seed 42) | mAP | mAUC | mF1 | mECE | tail AP |
|---|---|---|---|---|---|
| C - M | -0.0007 | +0.0022 | -0.0106 | +0.0028 | -0.0063 |
| D(with) - M | +0.0227 | +0.0060 | +0.0150 | +0.0023 | +0.0396 |
| D(withheld) - M | +0.0012 | +0.0019 | -0.0034 | +0.0048 | +0.0103 |
| C - A (entropy vs ASL) | +0.0005 | +0.0001 | -0.0096 | +0.0016 | -0.0022 |
| C - B (gate vs no gate) | -0.0032 | +0.0008 | -0.0114 | -0.0005 | -0.0143 |

Seed 1024 (D vs M, both exist): D(with) - M +0.0190 mAP; D(withheld) - M +0.0033 mAP. The M seed-to-seed spread is 0.0018 (42 vs 1024), so differences of about 0.001-0.003 are not distinguishable from noise. With one seed (two for D vs M) there is no interval, no sign-flip test (the minimum exact p with n=2 is 0.5), and no significance claim is supportable. These are descriptions of two or three models, not estimates.

## 3. Do the data so far show improvement?
- **Entropy (C) over MoE: no sign of it.** C is level with M on mAP (-0.0007), below on mF1 and tail AP, slightly worse on mECE. C is also level with ASL-only (+0.0005) and below the no-gate arm B (-0.0032), consistent with the earlier 512 px six-seed null of the gate. Nothing here supports "Triplet Memory + Entropy beats MoE"; one seed cannot refute a small effect either.
- **Entropy + Markov prior, history available (D with): large and consistent on both existing seeds** (+0.019 to +0.023 mAP over MoE, tail AP +0.03 to +0.04). But (a) this is the condition where the val split shares patients with train (77% of val images) and the test set gives no history, so the gain is likely leakage-inflated and is 0 on the leaderboard; (b) the same gain appears with ASL-only + prior (E with 0.4799 vs A 0.4600), so it is the prior, not the triplet/entropy part; D(with) - E(with) at seed 42 is +0.0040, one seed. Do not describe it as an entropy+Markov improvement.
- **Entropy + Markov prior, history withheld (the test-like row): +0.0012 (s42) and +0.0033 (s1024) over MoE**, both inside the M seed spread; E zero is 0.4590 vs A 0.4600 (no effect). Not distinguishable from MoE with the data so far.
- Honest summary: the current data show no demonstrated improvement of either system over MoE under test-like conditions; the only visible gain is the history-available prior, which is not an expected test-set effect. Whether the withheld D row ends slightly above M needs all five seeds.
- Other confounds still apply: single internal val fold (leaky, includes lateral views, absolute values optimistic), entropy is a softmax over binary labels, and D's s42/s1024 runs predate the unified script (same recipe by plan, different launch; reruns `_rerun` of them also exist and were not used).

## 4. Table 1 / paper.tex TBDs
Fillable now: none. The plan and `check_submission.py` require `generated/unified_table.tex`, which `unified_benchmark.py` only publishes when M, C, D have all five seeds (missing: M 123/456/789, C 123/456/789/1024, D 123/456/789 training plus the re-dumps of D s42/s1024 and all D with/zero evals). Every TBD (abstract results, the results-text paragraph at ~L210, Discussion ~L214, Conclusion ~L218) depends on paired 5-seed contrasts, so all must wait. The fixed, already-true Discussion sentences (512 px null, leakage, withheld-is-test-like, softmax entropy) can stay; the abstract/conclusion wording should not claim improvement unless the five-seed contrasts show it.

## 5. Jobs / ETA (`scripts/mon.py 'isbi2026_uni*'`, 20:30)
- Running (7): c_s123 51843 and d_s123 51845 (epoch 3/4, ~7 h in), c_s456 51857 (3/4), d_s456 51859, c_s789 51871, d_s789 51873 (2/4, ~4 h in), c_s1024 51883 (1/4, ~1 h in). Runs take ~11-13 h, so the first four finish within about 2-4 h of wall-clock, the 2/4 ones around 7-9 h, c_s1024 about 11-12 h, plus the afterok eval chains (a few hours each, with/zero for D).
- Pending, reason QOS-gpu (18): A 123/456/789, B 123/456/789/1024, E 123/456/789/1024, F 123/456/789/1024, M 123/456/789 (51897/51899/51901). Controls carry Nice=500 so C/D/M start first, but M's three runs are still queued behind the running ones.
- Rough estimate (a guess from the QOS cap of about 7 concurrent jobs, not a scheduler promise): M s123/456/789 start as slots free in the next hours and need ~11-13 h plus eval, so the 5-seed M, C, D set is likely complete around late 13 Oct to early 14 Oct, and the 12 control runs after that, 1-2 further days. Verify with `squeue --me --start` before relying on it.

## Not measured
Significance of any contrast; D re-dumped under the unified protocol; M seeds 123/456/789; extreme-tail breakdown reported (ignored here).

## Notable
- The arm mapping in the request was wrong (a = ASL, e = ASL+prior); the "e prior-with 0.4799" is the control, not entropy+Markov. D is 0.4839 / 0.4820 with history, 0.4624 / 0.4663 withheld.
- The leak-free (withheld) Markov row is only +0.001 to +0.003 over MoE so far, matching the earlier "0 on leaderboard" finding.

## Next steps
Wait for M, C, D five seeds; run `python -I analysis/unified_benchmark.py` (no `--allow-missing`) for the Holm-adjusted contrasts; then fill the TBDs. No extra jobs needed for the Table 1 rows; the controls only matter for the discussion.
