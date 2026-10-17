# Findings: seeded rerun, does Entropy differ from ASL only at n = 1? — 2026-10-15

**Metrics, configs, jobs, per-class table:** [docs/result/2026-10-15-seeded-asl-entropy-768.md](../result/2026-10-15-seeded-asl-entropy-768.md)
**Earlier 5-seed study:** [2026-10-14-asl-entropy-markov-768-findings.md](2026-10-14-asl-entropy-markov-768-findings.md), [../result/2026-10-14-asl-entropy-markov-768.md](../result/2026-10-14-asl-entropy-markov-768.md); why the gate is null: [2026-10-12-why-entropy-shows-no-gain.md](2026-10-12-why-entropy-shows-no-gain.md)

## Question
After the shuffle seeding fix (both `train_2_v0.py` and `train_2_v5.py` now seed the sample-buffer shuffle from `--seed`), does a clean single-seed (42, 768 px) rerun separate Triplet Memory + Entropy from ASL only on mAP, mAUC, mF1, mECE and head/medium/tail AP? Does the lambda-1.0 probe show the memory path can change the model?

## Data and method
Four seeded retrains (A, C, D, and probe P), one seed each, same recipe as job 49375. Scored on the internal validation split (17,270 rows) with SWA weights and flip TTA, same row order as the unseeded dumps. Group AP and per-class AP are recomputed from the probability dumps. A paired bootstrap over validation rows quantifies validation-sampling noise only. Seed-to-seed noise comes from the five unseeded seeds.

## Findings
1. **Headline (flip TTA, SWA, internal val, seed 42, n = 1).** ASL 0.4605 / mAUC 0.9103 / mF1 0.4715 / mECE 0.1327 / head 0.7627 / medium 0.4620 / tail 0.3269. Entropy 0.4625 / 0.9104 / 0.4753 / 0.1356 / 0.7628 / 0.4617 / 0.3362. Entropy + Markov, history available 0.4807 / 0.9120 / 0.4931 / 0.1338 / 0.7701 / 0.4819 / 0.3532. Entropy + Markov, withheld 0.4619 / 0.9089 / 0.4727 / 0.1362 / 0.7617 / 0.4603 / 0.3378.
2. **Entropy vs ASL, seeded: +0.0020 mAP** (mAUC +0.0001, mF1 +0.0038, mECE +0.0029, head +0.0001, medium −0.0003, tail +0.0093). The unseeded seed-42 pair was **−0.0051 mAP** (mAUC +0.0031, mF1 −0.0137, mECE +0.0011, tail −0.0219); the 5-seed mean is −0.0005 with a paired sd of 0.0035. The sign flipped; both numbers are inside about 1.5 sd of the paired noise. The row-bootstrap CI of the seeded delta is [−0.0012, +0.0080] (unseeded pair [−0.0148, +0.0033]).
3. **Entropy is not differentiated from ASL at n = 1, and cannot be claimed to be, in either direction.** A single pair carries a noise floor of about ±0.004 mAP (paired seed sd 0.0035; unseeded same-seed reruns differed by up to 0.0079), so +0.0020 is not a detectable effect. The medium and head AP differences are ≤ 0.0003 in magnitude; the mAUC difference is 0.0001. The n = 5 conclusion (C − A = −0.0005 [−0.0049, +0.0038]) is neither supported nor contradicted more than before; if the seeded +0.0020 replaced the unseeded seed-42 difference (−0.0051), the 5-seed mean C − A would be about +0.0009, still null.
4. **The tail delta is two classes.** Seeded tail C − A = +0.0093 comes from Hydropneumothorax (7 validation positives, +0.0523) and pneumoperitoneo (9 positives, +0.0385); the other five tail classes average −0.0052 (Subcutaneous Emphysema −0.0227, Pneumothorax −0.0090). In the unseeded pair the same two classes moved −0.1055 and −0.0765. A swing of 0.14 AP in Hydropneumothorax between two ASL runs that differ only in shuffle order (0.372 unseeded, 0.231 seeded) shows these two classes measure sampling noise, not method. Do not write tail claims from this table.
5. **Only calibration is consistent.** mECE for Entropy is worse than ASL by +0.0029 here, +0.0011 unseeded s42, and +0.0024 [+0.0012, +0.0035] in 5/5 unseeded seeds. Seeded D(zero) is also worse (+0.0036). It is a small effect (about 0.002-0.003 on 0.133) but the only one that repeats.
6. **Seeding did not make runs reproduce the earlier ones.** Seeded seed-42 results differ from unseeded seed-42 results by −0.0051 (A), +0.0020 (C), −0.0032 (D with), −0.0005 (D withheld), all within the 5-seed range. The seeded stream is a new stream, so this is expected. Seeding does make A and C more alike (mean logit correlation 0.983 versus 0.969 for the unseeded pair), consistent with a shared data order, which should shrink the data-order part of the paired noise. How much the paired noise actually shrank is unmeasured (needs two seeded runs of one config).
7. **Markov prior: unchanged story.** With history +0.0201 mAP over ASL (unseeded s42 +0.0183; 5-seed +0.0187), mAUC +0.0017, mF1 +0.0216, head/medium/tail +0.007/+0.020/+0.026; same model without history +0.0013 mAP over ASL (5-seed −0.0006). The history gain is +0.0188 on the same model (5-seed range +0.0158 to +0.0215). Leakage caveat: 77% of val images share a patient with train, 53% share their StudyID, so D(with) is not test-realistic. D(zero) is the test-like row and it is null within noise.
8. **Probe P (lambda 1.0, job 52655, 11.8 h, trained; flip-TTA eval 52656 still running).** Training-time SWA, no TTA: mAP 0.4535 vs C 0.4590, mF1 0.4648 vs 0.4695, mAUC 0.9105 vs 0.9091, mECE **0.1543 vs 0.1356**. So a 10-fold stronger triplet term does move the model: calibration worsens clearly (mECE +0.019, at every epoch +0.010 to +0.016), mAP drops 0.0055 (within noise at n = 1), no metric improves meaningfully (mAUC +0.0014 is inside seed sd 0.0011-0.0017). The memory path is therefore live and its effect, when amplified, is negative or null. The "14 h probe" question is answered at the training-time level: lambda 1.0 shows no hint of a benefit. Both weights see the same triplet yield (80.6% of steps with at least one triplet vs 80.0%), so the larger lambda is not a utilisation effect but a loss-scale effect. This does not test hard mining or the paper's max-entropy anchor (the runs use `--anchor-rule mean`, `easy` mining).

## What the data do and do not show
- Show: ASL ≈ Entropy within noise on mAP, mAUC, mF1, head and medium AP; Entropy slightly worse on mECE; D(with) +0.02 mAP over both, D(zero) ≈ ASL; stronger lambda hurts calibration.
- Do not show: that Entropy beats or loses to ASL on mAP (single seed, noise ±0.004); anything about tail classes; whether seeded runs are deterministic; flip-TTA numbers for P; test-set or leaderboard behaviour (internal validation only).

## Recommendation
- If Table 1 uses seed 42 alone, the seeded set above is the clean one (post shuffle fix, both pair members on the same data order). The ASL-vs-Entropy statement must stay "no measurable difference": do not word it as Entropy gains +0.002 or loses 0.005. The unseeded 5-seed table remains the better estimate of the means (ASL 0.4617 ± 0.0031, Entropy 0.4612 ± 0.0028, Markov with history 0.4805, withheld 0.4611); the seeded seed-42 row sits within 0.0013 of each of those means.
- Prefer reporting the 5-seed table with CIs for the paper, and use the seeded run only as a deterministic seed-42 single row; do not mix the two sources in one row.
- Report the mECE increase for Entropy and Markov-withheld (0.002-0.004) as the one consistent difference. Report the probe as "lambda 1.0: mAP −0.006, mECE +0.019, training-time, no TTA" once its eval is in, or state that its eval is pending.

## Caveats
- n = 1; Entropy runs use `--anchor-rule mean` and easy mining; the 7.6%/13%/80% utilisation figures are not reconciled (here 80% of steps carry at least one triplet); known paper confounds (softmax vs per-class sigmoid entropy, tail-analysis gap) unchanged. Post-fix runs for the class-weight order and `--metric-embed query` (shard / query in all four logs).
- Patient leakage in the validation split inflates D(with).
- The working-tree `train_2_v5.py` (MoE removed, seeded shuffle) and `train_2_v0.py` are uncommitted; the unseeded C/D runs used the pre-cleanup v5.

## Not measured
Flip-TTA eval of the probe (job 52656 running); a second seeded run of the same config (seeded reproducibility and seeded paired noise); seeded runs at other seeds; confidence intervals over seeds for the seeded configuration; test-set scores; a patient-disjoint split.

## Notable
- The ASL-vs-Entropy gap changed sign between unseeded and seeded seed-42 pairs (−0.0051 to +0.0020 mAP) with the group-level tail swing −0.0219 to +0.0093: a single-seed table row cannot carry an ASL-vs-Entropy claim.
- The lambda-1.0 probe shows the triplet path affects calibration (+0.019 mECE) but not accuracy for the better.
- Hydropneumothorax (7 val positives) AP moved 0.14 between two ASL runs that differ only in sample-shuffle order.

## Next steps
1. When job 52656 finishes, compute P's flip-TTA row with the same script (dump to `analysis/out/probs_unified_flip_seeded/p__seed_42__model_swa@768_flip.npy`, `eval_p_s42.json`) and append to the result file.
2. If a seeded paired claim is wanted, run at least one more seeded seed (or a duplicate seed-42 seeded A) to measure the seeded paired noise; this analysis launched no jobs.
3. Keep the paper statements at "no measurable difference" for Entropy vs ASL and "mECE slightly worse"; keep Markov claims limited to the history-available condition with the leakage caveat.
