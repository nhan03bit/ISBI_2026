# Matched-recipe MoE / ASL / MoE+Prior baselines: flip-TTA evaluation — 2026-10-11

**Directory:** `train/checkpoint_MoE/{moe,asl,moeprior}_img*_seed*/Model_run/` (training); `analysis/out/probs_moe_flip/` (eval dumps)
**Script / recipe:** `train/train_2_v5.py` with `--no-triplet`, `--arch {moe,convnext2}`, v3 recipe; eval `evaluate/evaluate_tta_v5.py` (flip TTA, single scale); comparison `analysis/moe_compare.py`
**Slurm job(s):**
- Training: 51151-51159 (all COMPLETED 0:0), logs `/home/psytp7/logs/isbi2026_{moe,asl,moeprior}_img*_511{51..59}.out`
- Flip-TTA eval: 51736-51746 (all COMPLETED 0:0, 25-46 min each), logs `/home/psytp7/logs/isbi2026_eval_*_517{36..46}.out`
- The first eval chain (51178-51189) failed with `RuntimeError: row order differs from the existing keys.txt` (`--batch-size 4` in `scripts/submit_eval_moe.sh`, reference dump built with 8; with 2 val shards the row order depends on batch size). `evaluate_tta_v5.py` now permutes rows into the `keys.txt` order and still requires the same row set. All 11 reruns loaded with `(missing=0 unexpected=0)` and `n_rows = 17270`.

Plan and design: [`docs/plan/2026-10-08-moe-matched-baseline-plan.md`](../plan/2026-10-08-moe-matched-baseline-plan.md). Related: [`docs/plan/2026-10-08-entropy-vs-moe-baseline.md`](../plan/2026-10-08-entropy-vs-moe-baseline.md), [`docs/plan/2026-10-08-moe-anchored-entropy-markov-vs-gcn.md`](../plan/2026-10-08-moe-anchored-entropy-markov-vs-gcn.md).

All numbers below are **internal validation (val_fold_1, 17,270 images), flip TTA, single scale, SWA checkpoints** unless stated. They are not leaderboard or test numbers. The old published MoE 0.4599 (b12) is a test number from a different recipe and is not comparable.

## Configuration
| Setting | Value |
|---|---|
| Stage-1 checkpoint | `checkpoint3/Model_20260119_062652/model_best.pth` (all 9 MoE-track runs and all Triplet Memory twins; read from the `args:` lines and `history.json`) |
| Image size / seeds | MoE: 640 s86, 768 s42, 768 s1024, 896 s86, 1024 s86. ASL: 768 s42, 768 s1024. MoE+prior: 768 s42, 768 s1024 |
| LR / batch / accum | lr 1e-4, batch 4, accum 12 (effective 48), `--swa-last-k 3`, `--monitor mAP`; same as twins |
| Schedule | 768/896/1024: `--epochs 4 --patience 2 --sched-epochs 8 --drop-path-rate 0.1`. 640: `--epochs 8 --patience 3`, no drop-path/sched-epochs (this matches the Triplet Memory `res_sweep` 640 twin, job 47094) |
| Module | MoE arm: `--arch moe`, 4 experts top-2, aux weight 5e-3 (log: `moe_aux_weight=0.005`). ASL arm: `--arch convnext2`, aux 0. MoE+prior: `--arch moe` plus `--prior-file analysis/out/patient_markov/priors.pt --prior-dropout 0.3 --prior-lr-mult 10` (same prior args as the Triplet Memory + Markov twins 49375/49855) |
| metric-embed / memory-size / triplet-lambda | not applicable: all 9 logs print `no-triplet: triplet loss, memory bank and metric embedding disabled`; per-epoch `ASL-only steps = 25822`. (`memory_size: 4096` appears in the logged default config dict only; no bank is built.) |
| Triplet Memory twins | `--metric-embed query --memory-size 2048 --triplet-lambda 0.1 --head-classes 5` (jobs 48997-49000, 47094, 49375/49855) |
| class-weight-order | `shard` (fixed) in all 9 MoE-track runs and in all Triplet Memory twins whose `args:` I read (49000, 48999, 48998, 48997, 47094, 49375, 49855) |
| Eval | `evaluate_tta_v5.py --batch-size 4`, flip TTA, scale = training size, strict loads for MoE (`ConvNeXt2MoE` / `ConvNeXt2MoEPrior`, missing=0, unexpected=0) |

## Results

### 1. Per run (flip TTA, SWA; 640 row uses `model_best`, see note)
Triplet Memory (TM) = entropy-gated triplet, v3. Values read from `analysis/out/probs_moe_flip/eval_*.json` and `analysis/out/moe_compare.json`; the two sources agree to 1e-6 for every single-run arm.

| Config | Arm | mAP | mAUC | mF1 | mECE |
|---|---|---|---|---|---|
| 768 s42 | TM | 0.4650 | 0.9106 | 0.4748 | 0.1352 |
| 768 s42 | MoE | 0.4612 | 0.9102 | 0.4776 | 0.1318 |
| 768 s42 | ASL | 0.4600 | 0.9123 | 0.4766 | 0.1330 |
| 768 s1024 | TM | 0.4626 | 0.9117 | 0.4788 | 0.1352 |
| 768 s1024 | MoE | 0.4630 | 0.9117 | 0.4716 | 0.1319 |
| 768 s1024 | ASL | 0.4599 | 0.9110 | 0.4728 | 0.1326 |
| 896 s86 | TM | 0.4578 | 0.9130 | 0.4712 | 0.1397 |
| 896 s86 | MoE | 0.4634 | 0.9130 | 0.4675 | 0.1370 |
| 1024 s86 | TM | 0.4645 | 0.9145 | 0.4675 | 0.1438 |
| 1024 s86 | MoE | 0.4636 | 0.9140 | 0.4693 | 0.1414 |
| 640 s86 (best) | TM | 0.4485 | 0.9009 | 0.4597 | 0.1223 |
| 640 s86 (best) | MoE | 0.4492 | 0.9078 | 0.4610 | 0.1330 |

Note on 640: `eval_moe_img640_s86.json` reports 0.4516 / 0.9082 / 0.4631 / 0.1316, but that is the average of **two** checkpoints (`model_swa` + `model_best`, `n_models = 2`), not a single run. The table row is the `model_best` single checkpoint as used in `moe_compare.json` (same as the Triplet Memory 640 twin, which also uses `model_best` from `res_sweep`). A 640 SWA-only single was not computed.

TM minus MoE mAP per config: +0.0038 (768 s42), -0.0005 (768 s1024), -0.0056 (896), +0.0009 (1024), -0.0007 (640). The sign is mixed and the mean is -0.0004. MoE minus ASL mAP: +0.0012 (s42), +0.0032 (s1024).

MoE + Markov prior (patient-history prior into the label queries), 768 px, flip TTA, SWA, next to the Triplet Memory + Markov twins (orig = jobs 49375/49376, rerun = 49855/49856). "zero" evaluates the same checkpoint with the prior zeroed.

| Config | Arm | prior | mAP | mAUC | mF1 | mECE |
|---|---|---|---|---|---|---|
| 768 s42 | MoE+prior | with | 0.4805 | 0.9125 | 0.4921 | 0.1298 |
| 768 s1024 | MoE+prior | with | 0.4809 | 0.9147 | 0.4982 | 0.1301 |
| 768 s42 | MoE+prior | zero | 0.4592 | 0.9099 | 0.4711 | 0.1323 |
| 768 s1024 | MoE+prior | zero | 0.4651 | 0.9099 | 0.4825 | 0.1328 |
| 768 s42 | TM+M orig | with | 0.4839 | 0.9162 | 0.4926 | 0.1341 |
| 768 s1024 | TM+M orig | with | 0.4820 | 0.9153 | 0.4932 | 0.1324 |
| 768 s42 | TM+M rerun | with | 0.4760 | 0.9144 | 0.4885 | 0.1326 |
| 768 s1024 | TM+M rerun | with | 0.4786 | 0.9165 | 0.4904 | 0.1329 |
| 768 s42 | TM+M orig | zero | 0.4624 | 0.9121 | 0.4742 | 0.1366 |
| 768 s1024 | TM+M orig | zero | 0.4663 | 0.9114 | 0.4834 | 0.1350 |
| 768 s42 | TM+M rerun | zero | 0.4560 | 0.9106 | 0.4694 | 0.1350 |
| 768 s1024 | TM+M rerun | zero | 0.4622 | 0.9120 | 0.4742 | 0.1355 |

### 2. Aggregates (probability averages, fixed members, no re-selection on val)
Members: 2-seed = 768 s42 + s1024; fixed 4-model = 768 s42, 768 s1024, 896 s86, 1024 s86; trio = 768 s42 + 1024 s86 + 640 s86 (best), the members of the greedy Triplet Memory 0.4722 ensemble (so the trio is selected for TM on val; the MoE trio uses the same members, which slightly favours TM).

| Aggregate | mAP | mAUC | mF1 | mECE |
|---|---|---|---|---|
| TM 2-seed 768 | 0.4659 | 0.9126 | 0.4817 | 0.1352 |
| MoE 2-seed 768 | 0.4668 | 0.9125 | 0.4788 | 0.1318 |
| ASL 2-seed 768 | 0.4625 | 0.9131 | 0.4730 | 0.1328 |
| TM fixed 4-model | 0.4699 | 0.9154 | 0.4819 | 0.1385 |
| MoE fixed 4-model | 0.4716 | 0.9155 | 0.4758 | 0.1355 |
| TM trio | 0.4722 | 0.9144 | 0.4769 | 0.1337 |
| MoE trio (same members) | 0.4697 | 0.9148 | 0.4767 | 0.1354 |
| MoE+prior 2-seed, with | 0.4835 | 0.9153 | 0.5000 | 0.1300 |
| MoE+prior 2-seed, zero | 0.4663 | 0.9115 | 0.4825 | 0.1325 |
| TM+M orig 2-seed, with / zero | 0.4867 / 0.4693 | 0.9170 / 0.9131 | 0.4980 / 0.4811 | 0.1332 / 0.1358 |
| TM+M rerun 2-seed, with / zero | 0.4797 / 0.4611 | 0.9169 / 0.9128 | 0.4904 / 0.4706 | 0.1328 / 0.1352 |

### 3. Paired image-level bootstrap (B = 2000, from `analysis/out/moe_compare.json`)
Point difference [95% CI]. Bold = CI excludes 0.

| Contrast (A - B) | mAP | mAUC | mF1 | mECE |
|---|---|---|---|---|
| TM - MoE, 2-seed 768 | -0.0008 [-0.0065, +0.0043] | +0.0001 [-0.0009, +0.0012] | +0.0029 [-0.0090, +0.0171] | **+0.0034 [+0.0033, +0.0035]** |
| TM - ASL, 2-seed 768 | +0.0035 [-0.0018, +0.0101] | -0.0005 [-0.0015, +0.0005] | +0.0087 [-0.0031, +0.0221] | **+0.0024 [+0.0023, +0.0024]** |
| MoE - ASL, 2-seed 768 | +0.0043 [-0.0034, +0.0123] | -0.0006 [-0.0019, +0.0005] | +0.0057 [-0.0060, +0.0203] | **-0.0010 [-0.0011, -0.0009]** |
| TM - MoE, fixed 4-model | -0.0016 [-0.0051, +0.0023] | -0.0001 [-0.0010, +0.0009] | +0.0061 [-0.0042, +0.0199] | **+0.0030 [+0.0029, +0.0030]** |
| TM - MoE, trio | +0.0026 [-0.0056, +0.0104] | -0.0005 [-0.0016, +0.0007] | +0.0002 [-0.0050, +0.0053] | **-0.0016 [-0.0017, -0.0016]** |
| MoE+prior with - MoE, 2-seed | **+0.0168 [+0.0097, +0.0241]** | **+0.0029 [+0.0001, +0.0053]** | **+0.0212 [+0.0086, +0.0364]** | **-0.0019 [-0.0020, -0.0018]** |
| MoE+prior zero - MoE, 2-seed | -0.0005 [-0.0045, +0.0048] | -0.0009 [-0.0020, +0.0001] | +0.0037 [-0.0081, +0.0180] | **+0.0007 [+0.0006, +0.0008]** |
| TM+M orig with - TM, 2-seed | **+0.0208 [+0.0130, +0.0295]** | **+0.0044 [+0.0019, +0.0066]** | **+0.0163 [+0.0083, +0.0243]** | **-0.0020 [-0.0021, -0.0019]** |
| TM+M rerun with - TM, 2-seed | **+0.0138 [+0.0047, +0.0217]** | **+0.0043 [+0.0022, +0.0064]** | +0.0087 [-0.0059, +0.0223] | **-0.0025 [-0.0026, -0.0024]** |
| TM+M orig with - MoE+prior with | +0.0032 [-0.0008, +0.0085] | **+0.0017 [+0.0002, +0.0032]** | -0.0020 [-0.0080, +0.0038] | **+0.0033 [+0.0032, +0.0034]** |
| TM+M rerun with - MoE+prior with | -0.0038 [-0.0099, +0.0010] | +0.0016 [-0.0000, +0.0035] | -0.0095 [-0.0232, +0.0026] | **+0.0028 [+0.0027, +0.0029]** |
| TM+M orig zero - MoE+prior zero | +0.0030 [-0.0015, +0.0091] | **+0.0015 [+0.0002, +0.0030]** | -0.0014 [-0.0064, +0.0046] | **+0.0033 [+0.0032, +0.0034]** |
| TM+M rerun zero - MoE+prior zero | -0.0052 [-0.0126, +0.0017] | **+0.0013 [+0.0002, +0.0025]** | -0.0119 [-0.0256, +0.0005] | **+0.0027 [+0.0026, +0.0028]** |

### Cross-check against training histories
`history.json` `swa_metrics` are no-TTA single-scale numbers on the same val fold, so they sit about 0.002-0.006 mAP below the flip-TTA values (e.g. MoE 768 s42: 0.4560 no-TTA vs 0.4612 flip; ASL 768 s1024: 0.4574 vs 0.4599; TM 768 s42: 0.4600 vs 0.4650). Each log's `Done. Best epoch ... | Best mAP` and `SWA weights -> ... mAP=` lines agree with `history.json` (best_metric / swa_metrics) for all 9 runs. History-level (no-TTA) SWA mAP, MoE / ASL / TM: 768 s42 0.4560 / 0.4594 / 0.4600; 768 s1024 0.4585 / 0.4574 / 0.4582. The ranking of arms at no-TTA is therefore not the same as at flip-TTA (ASL > MoE at s42 without TTA), which again says the arms are not separable at this noise level. The evaluator regression job 51178 (8 Oct, TM push 768 s42 SWA through the modified evaluator) gave mAP 0.464986 vs 0.464986 in `moe_compare.json`, i.e. TM loads were not changed by the MoE edits (the raw arrays are in a different row order from `probs_w01_flip`, so a literal max|delta| = 0 check was not possible; I compared metrics only).

## Comparison to baselines
- vs Stage-1 backbone (mAP 0.385 internal): all arms are far above it (0.448-0.465 single-model flip TTA; note the 0.385 is a different protocol, no flip TTA and original resolution).
- vs MoE Stage-2 published (internal 0.405 / test 0.4599): not comparable. The published MoE is at 384/512 px, a different recipe, no seeds or logs in the repo. The rebuilt MoE at matched recipe reaches 0.4612-0.4636 internal at 768-1024 px, i.e. resolution and recipe, not the MoE module, account for most of the old-to-new internal gap (ASL alone gets 0.4600/0.4599 at 768).
- vs paper's entropy-triplet 5-seed result (mAP 0.414 +/- 0.002, 512 px, no flip TTA): different resolution and protocol; not comparable.
- vs Triplet Memory twins (this report, matched recipe): see conclusion below.

## Conclusion (honest reading)
1. **mAP: Triplet Memory, MoE and ASL are not separable.** TM - MoE on the 2-seed 768 pair is -0.0008 [-0.0065, +0.0043]; fixed 4-model -0.0016 [-0.0051, +0.0023]; trio +0.0026 [-0.0056, +0.0104]. TM - ASL +0.0035 [-0.0018, +0.0101]; MoE - ASL +0.0043 [-0.0034, +0.0123]. Every mAP, mAUC and mF1 CI includes zero. The per-config TM - MoE sign flips (+0.0038, -0.0005, -0.0056, +0.0009, -0.0007). The observed 0.003-0.004 point gaps of both TM and MoE over ASL are in the same range as the seed-to-seed spread inside one arm (TM 768: 0.4650 vs 0.4626), so no ordering among the three can be claimed from this data. This report does **not** support "Entropy beats MoE" at matched recipe; it also does not support the opposite.
2. **Calibration: TM has the higher mECE in most, not all, matched comparisons.** TM - MoE mECE is +0.0034 (2-seed 768) and +0.0030 (4-model), TM - ASL +0.0024, and per config TM is higher than MoE at 768 s42, 768 s1024, 896 and 1024 (0.1352 vs 0.1318, 0.1352 vs 0.1319, 0.1397 vs 0.1370, 0.1438 vs 0.1414). It reverses at 640 (TM 0.1223 vs MoE 0.1330) and in the trio (TM 0.1337 vs MoE 0.1354, CI [-0.0017, -0.0016]). MoE has a marginally lower mECE than ASL (-0.0010). The effect is small in absolute terms (about 0.003 on a base of 0.13). The very narrow mECE CIs come from resampling images only; they ignore seed variance and only 2 seeds per arm exist, so "TM is worse calibrated" is a consistent but small and not seed-validated observation. It is consistent with the paper's own finding that the entropy-triplet run regresses mECE vs Stage-1.
3. **Markov prior (patient-history prior).** MoE+prior with-prior: 0.4805 / 0.4809 (2-seed 0.4835), zero-prior 0.4592 / 0.4651 (2-seed 0.4663). The prior adds +0.0168 [+0.0097, +0.0241] mAP over the plain MoE 2-seed, but zeroing the prior returns the plain-MoE level (-0.0005 [-0.0045, +0.0048]), so the gain is carried entirely by the prior input at inference. The effect is the same size on both backbones (TM+M orig +0.0208, rerun +0.0138, MoE+M +0.0168), and MoE+M vs TM+M with-prior is inside the CI on mAP (+0.0032 [-0.0008, +0.0085] orig, -0.0038 [-0.0099, +0.0010] rerun). TM+M leads on mAUC by about 0.0016 (CI excludes 0 for orig and for both zero comparisons; the rerun with-prior CI touches 0), while mF1 leans slightly toward MoE+M; neither is a mAP effect. So the prior's benefit does not depend on the triplet module.
4. **Class-weight and wiring bugs.** All 9 MoE-track runs and the 7 Triplet Memory twin logs I checked use `--class-weight-order shard` (fixed side). Triplet Memory twins use `--metric-embed query` (fixed side). No run here falls on the buggy side.

## Caveats / known-bug flags
- **Internal-val only; split is not patient-disjoint** (project memory: 77% of val images share a patient with train, 53% share their StudyID). The bootstrap resamples images, which are not independent within a patient/study, so CIs are somewhat too narrow, not too wide; the "within CI" conclusions in item 1 would only get weaker. Absolute numbers (about 0.46-0.48) are not leaderboard numbers.
- **The with-prior gains are internal-val numbers.** Per project memory the patient-history prior gave 0 on the leaderboard (2026-09-30), because the leakage above supplies patient labels that the leaderboard split does not. Do not present 0.48 as a test result or compare it with the 0.4722 no-history ensemble. I did not re-verify the leaderboard claim in this task.
- **Row alignment.** MoE/ASL/MoE+prior dumps are permuted into `keys.txt` order by the fixed evaluator. The Triplet Memory dumps in `probs_w01_flip` have no `keys.txt`; `moe_compare.py` aligns them only through the `labels.npy` equality assertion. Metric-level agreement (TM 768 s42 flip-TTA mAP identical to 6 decimals in the 8 Oct regression eval) supports the alignment, but a row swap between images with identical label vectors would not be caught by the assertion. Not checked beyond this.
- **Unequal selection.** The trio is the greedy-selected TM member set (selected on val for TM); the same members are used for MoE. The 640 row uses `model_best` (selected by val mAP) while the other rows use fixed-window SWA; this holds for both arms.
- **Only 2 seeds** at 768 and 1 seed at 640/896/1024 per arm; no 5-seed estimate for MoE or ASL.
- **Paper confounds touched** (project memory `paper-revision-confounds`): the 384-vs-512 resolution confound is avoided here (all arms matched at 640-1024); the softmax-vs-sigmoid entropy choice and easy-vs-hard mining are properties of the TM twins and not re-examined; the 7.6%-vs-13% triplet-utilization discrepancy is not touched; the tail-analysis gap is **not** closed (see below).
- The logs contain `libpng error: PNG input buffer is incomplete` (one truncated validation image). It appears in the reference TM eval logs as well, and `n_rows = 17270` in every run, so I treat it as benign but did not trace which file it is.

## Not measured
- No head/medium/tail per-class breakdown. No per-class `eval_result*.json` exists for these checkpoints, and `moe_compare.json` holds macro metrics only; `evaluate/class_groups.py` was not applied. The question "does TM/MoE help the tail" is open.
- No MoE routing diagnostics (expert load, collapse) in this report; the per-epoch aux/load log lines exist in the training logs but were not analysed.
- No 640 SWA-only single, no multi-scale TTA, no test/leaderboard evaluation.

## Notable
- First matched-recipe MoE and ASL controls: **at matched recipe, resolution and flip TTA, the MoE module, the entropy-gated triplet loss and plain ASL are statistically indistinguishable on internal mAP** (all paired CIs include 0). This contradicts any reading of Table III that attributes the old-to-new gain to the Triplet Memory module; resolution and SWA/flip-TTA account for most of it. It also weakens a "Triplet Memory beats MoE" claim for the paper unless more seeds or a different metric (calibration/tail) separate them, and on calibration TM is the slightly worse arm.
- The Markov-prior gain (about +0.014 to +0.021 mAP with-prior, 0 with zeroed prior) is backbone-independent (TM, MoE) but is an internal-val effect that uses the leaky split.

## Next step
- Per-class head/medium/tail eval with `evaluate/class_groups.py` on the 12 dumps already in `analysis/out/probs_moe_flip/` (cheap, no GPU): this is the only place TM could still differ from MoE/ASL (tail).
- More seeds for the 768 arms (at least 5 each for TM, MoE, ASL) with a seed-level (not only image-level) interval, if the paper wants to claim either an mAP or an mECE difference.
- Use a patient-cluster bootstrap for the CIs.
- Paper: use the matched MoE and ASL rows in Table III (replacing the mislabeled rows noted in the plan) and report the TM-vs-MoE comparison as "no significant difference in mAP; TM slightly worse calibrated".
