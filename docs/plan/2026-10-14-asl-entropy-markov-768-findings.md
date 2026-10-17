# Findings: ASL only vs Entropy vs Entropy + Markov prior at 768 px — 2026-10-14

**Metrics, configs, jobs:** [docs/result/2026-10-14-asl-entropy-markov-768.md](../result/2026-10-14-asl-entropy-markov-768.md). Plan: [2026-10-13-asl-v0-vs-entropy-v5-plan.md](2026-10-13-asl-v0-vs-entropy-v5-plan.md). Parity: [2026-10-13-v0-v5-parity-validation.md](2026-10-13-v0-v5-parity-validation.md). The older [2026-10-12-unified-table1-status.md](2026-10-12-unified-table1-status.md) (MoE-era arm mapping) is superseded.

## Question
On one recipe (job 49375: 768 px, Stage-1 `checkpoint3/Model_20260119_062652`, 4 of 8 cosine epochs, SWA, flip TTA), does the entropy-gated triplet loss, or the patient-history Markov prior added to it, beat plain ConvNeXt + ASL on the internal validation split?

## Data and method
Five seeds (42, 123, 456, 789, 1024) per system, internal val (17,270 rows), SWA weights, flip TTA, one row order. Row A: `train/train_2_v0.py` (ASL only). Rows C (Triplet Memory + Entropy) and D (+ Markov prior, scored with history available and withheld): `train/train_2_v5.py`. The two scripts are never pooled. Paired-by-seed contrasts: mean difference, 95% t-CI, exact sign-flip p, wins, Holm over five mAP p-values, paired-t MDE. Residual script difference: v0 still computes and discards ConvNeXt2's legacy embedding output (shared `convnext.py`, no parameters, no effect on loss); the webdataset shuffle buffer is unseeded in both scripts, so same-seed reruns differ.

## Findings
1. Recipe parity holds: the five v0 `args:` lines match job 49375 on every shared setting (only the triplet/memory/prior arguments are absent), class weights are in shard order (identical line in the logs compared), Stage-1 loads with missing=0, all five v0 runs finished 4 epochs with best epoch 3 and SWA [2,3,4], and all ten Slurm jobs COMPLETED 0:0. C and D runs match 49375 apart from the prior arguments (C) and an explicit `--anchor-rule mean` equal to the v5 default. No known bug applies (shard weights, `--metric-embed query` in C/D).
2. Entropy triplet alone adds nothing over ASL: C − A = −0.0005 mAP [−0.0049, +0.0038], 2/5 seeds. mAUC and mF1 are also flat. Calibration is slightly worse (mECE +0.0024 [+0.0012, +0.0035], 5/5 seeds), head AP +0.0012 [+0.0004, +0.0021], medium and tail flat (tail +/−0.014 CI). With MDE 0.0058, an effect larger than about 0.005 mAP is unlikely; smaller ones cannot be excluded. This agrees with the earlier 512 px ablation (gate null) in [2026-10-12-why-entropy-shows-no-gain.md](2026-10-12-why-entropy-shows-no-gain.md).
3. The Markov prior helps only when history exists. With history available: +0.0187 mAP over ASL [+0.0149, +0.0225], 5/5 seeds, and +0.0193 over C [+0.0151, +0.0234], 5/5 seeds; mAUC +0.004, mF1 +0.020, head +0.008, medium +0.019, tail +0.024 (tail CI [+0.010, +0.037]); mECE equal to ASL's (+0.0004 [−0.0002, +0.0010]). With history withheld (the test-like condition, since the challenge test patients have no history): −0.0006 vs ASL [−0.0066, +0.0054], −0.0001 vs C [−0.0053, +0.0050]; mECE +0.0029 vs ASL (5/5 worse), medium AP −0.0024 (0/5 wins), tail flat (large CI). The same five models give +0.0194 mAP with versus without history (5/5).
4. Statistics at n = 5: the smallest attainable two-sided exact sign-flip p is 0.0625, so even 5/5 wins (both Dwith contrasts) give p = 0.0625, Holm 0.3125. No contrast is "significant at 0.05" under the pre-set test; the CIs exclude zero and the effect is about 3.5 times the MDE (0.0051-0.0056). Report CI and wins, not p, for the D(with) claim.
5. Noise floor: same-seed reruns of the D runs differ by 0.0053 (s42: 0.4793 vs 0.4740 best EMA) and 0.0001 (s1024: 0.4760 vs 0.4761); in flip-TTA single-model D(with) mAP the difference is 0.0079 (s42) and 0.0034 (s1024). The row-D s42 point is the original run (the high one); using its rerun would lower the D(with) mean by about 0.0016 and not change any conclusion.
6. Tail AP is a seven-class mean with sd 0.012-0.015 across seeds, dominated by Hydropneumothorax (38 training positives; A 0.297, C 0.271, Dwith 0.360, Dzero 0.377 mean class AP) and Mass (0.08). Tail claims are the least certain numbers in the table.

## What the data do and do not show
- Show: ASL ≈ Entropy ≈ Entropy+Markov-without-history within about 0.005 mAP; Markov prior with history +0.019 mAP, consistently across seeds.
- Do not show: that the entropy gate or the triplet term helps (it is null by C − A and by the earlier ablation); that the prior helps on the challenge test set (the withheld row is the test-like row and is null; its mECE and medium AP are slightly worse); any tail-class benefit that survives seed noise except under history.
- The history-available row is leakage-prone: the validation split is not patient-disjoint (77% of val images share a patient with train, 53% share their StudyID), and only 33% of validation images have a prior (`has_prior` 0.334 in the 49375 log), so the +0.019 is an average of a larger gain on those images and zero elsewhere; it is an upper bound for a hospital archive deployment, not a test-set claim.
- The numbers are internal validation, not leaderboard. The 384-vs-512 resolution confound does not arise (all 768 px).

## Recommendation
Table 1 can be filled with `generated/unified_table.tex` as is. In prose, call the Entropy gain null, and present D(with) as "when a prior report exists", D(zero) as the test-like row with no measurable change; do not quote 0.4807 (job 49375 SWA, no TTA) as an expected value. Report n = 5 contrasts as CI and wins, and say the minimum exact p is 0.0625. If the paper claims anything about calibration, note that C and D(zero) are about 0.002-0.003 worse in mECE than ASL (5/5 seeds).

## Caveats
Same as the result file: C/D ran on the pre-cleanup v5 (tag `v5-pre-moe-removal`); the shuffle buffer is unseeded (noise floor); one validation fold; the C/D versus A comparison crosses two scripts that are source-identical on shared code and bit-identical in a 40-batch smoke at `--shuffle-buf 1`, not bit-identical at shuffle 400. The paper's own text must still be reconciled (see below).

## Not measured
Test-set/leaderboard scores for these rows; a patient-disjoint split; ASL + Markov prior (arm dropped); an ensemble row; a per-class table from an `eval_result*.json`; the discarded-embedding effect of ConvNeXt2 in v0 (argued zero, not measured beyond the 40-batch parity).

## Paper housekeeping (read, not edited)
- `docs/paper/isbi_submit/paper.tex` row labels: the generated rows are "ASL only (asymmetric loss)", "Triplet Memory + Entropy", "... + Markov prior, history available", "... history withheld". `\IfFileExists{generated/unified_table.tex}` will use them; the file is in place. The fallback block (used only if the file is missing) still starts with "MoE (sparse mixture of experts)" and is stale. The caption's "Empty cells await the runs" and the TBD results/discussion/conclusion text remain, as do the \AUTHOR items.
- `python3 -I check_submission.py --no-build` (read-only; `paper.pdf` is dated 2026-10-12, before the table existed): 4 pages; fails with 4 `\TBD` and 3 `\AUTHOR` items (ethics, funding, author order). The build check was not run (it calls latexmk and rewrites the PDF); the table must be rebuilt before the page layout check means anything.
- The Discussion says "Markov prior ... only its withheld rows are test-like": consistent with the numbers.

## Notable
- Contradicts any residual expectation that the entropy gate helps: C − A is null at 768 px, same as the 512 px ablation.
- New best internal result on this recipe family: D(with) mean 0.4805 ± 0.0027 over 5 seeds (flip TTA, SWA), but history-dependent and leakage-prone; D(zero) 0.4611 does not beat ASL 0.4617.
- The "0.4848" in the pasted job-status line for 49375 is the epoch-4 mF1, not mAUC.

## Next steps
1. Fill the paper's \TBD text from the contrasts above (author decision); rebuild with latexmk, then run `check_submission.py` (without `--no-build`).
2. Remove the stale MoE fallback row in the table and the "Empty cells" caption sentence.
3. Optionally strengthen the Dzero conclusion with a patient-disjoint or test-like re-split, and consider extra seeds only if a statement needing p < 0.05 is wanted (n = 5 cannot reach it with sign-flip).
