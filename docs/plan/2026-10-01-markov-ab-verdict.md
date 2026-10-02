# Verdict: A (label-graph Markov layer) vs B (patient-history chain) — 2026-10-01

Training/eval numbers: [docs/result/2026-10-01-markov-ab-training.md](../result/2026-10-01-markov-ab-training.md). Plan and fixed decision rule: [2026-09-30-markov-ab-training-plan.md](2026-09-30-markov-ab-training-plan.md). Background: [2026-09-30-patient-markov-research.md](2026-09-30-patient-markov-research.md). Source of every number: analysis/out/markov_ab/run.log (job 49442). INTERNAL val only; not leaderboard.

## Question
Which is more effective: A, a trainable label-graph Markov layer, or B, a per-class patient-history chain prior, injected either into the network label queries (B-early) or post hoc on the probabilities (B-late)?

## Data and method
Flip-TTA SWA probabilities of v3 (49000, 48999), v4 A arms (mk1, mk2, mk1fix, A-strong lr100), B-early (s42, s1024, priors with / zeroed), plus B-late fusion on v3 probs (w = 1 and CV-w). Primary subset = has-prior & frontal (n 4061 of 17270). Paired patient-clustered bootstrap, 300 resamples. Row alignment verified: keys.txt == val_key_order.npy and v3 s42 re-dump matches probs_w01_flip with max |d| 0.0. B-late CV-w pools two patient halves; safe because both halves chose w = 0.5 at both seeds ("w chosen [0.5, 0.5]"); pooling halves with different transforms creates artefacts (separate redesign run).

## Findings
1. **Internal / longitudinal winner (decision rule [5]): B-late CV-w.** Primary Δ vs v3: +0.0493 (CI +0.0321, +0.0715) at s42 and +0.0564 at s1024 (CI +0.0359, +0.0788). B-early (+0.0270 [+0.0081, +0.0497] s42; +0.0380 s1024) and B-late w1 (+0.0370; +0.0468) also qualify. No A arm qualifies (primary Δ -0.0029 to -0.0108, all CIs include 0; A has one seed).
2. **B-early vs B-late (paired, [2]):** B-early − CV-w on primary: -0.0222 (CI -0.0385, +0.0014) s42, -0.0184 (CI -0.0372, +0.0070) s1024; on has_prior: -0.0177 (CI -0.0354, +0.0002) and -0.0157 (CI -0.0382, +0.0004). B-early − B-late w1 on primary: -0.0099 and -0.0088 (CIs span 0). So B-early is not better than late fusion; versus CV-w it is lower at both seeds with CIs just touching 0.
3. **Guards (B-early):** Guard 1 (no-prior images not worse than baseline) passes at both seeds (no_prior Δ +0.0033 [-0.0060, +0.0286] s42; +0.0014 [-0.0046, +0.0112] s1024). Guard 2 (new-finding stratum not worse) passes at both seeds.
4. **Copy check (stratified AP, [3]):** Prior-negative stratum (new vs absent, 28 classes): v3 0.3626 (s42), B-early 0.3684, d +0.0057 (boot +0.0064 [+0.0011, +0.0138]); s1024 +0.0012 (CI -0.0065, +0.0071). B-late leaves it exactly unchanged (d 0.0000, by construction). Prior-positive stratum (persistent vs resolved, 26 classes): B-early d -0.0059 (CI -0.0123..+0.0041) s42, -0.0017 s1024; all within noise. So there is no evidence B-early gains by copying the prior; its within-stratum ranking is unchanged to slightly better for new findings, and its gain is mainly between-stratum (as in late fusion). A significant new-finding gain exists at s42 only; do not over-read it.
5. **Head/medium/tail (primary subset, [4]; this script's 3/20/7 split, tail n = 7):** B-early gains in all groups (s42 head +0.0175, medium +0.0360 [20/20 up], tail +0.0055; s1024 +0.0122, +0.0361, +0.0546). B-late CV-w gains more in the tail (+0.0918 s42, +0.1250 s1024) with head +0.013 and medium +0.039. A arms: head and medium ≈ 0, tail negative (-0.013 to -0.053), A-strong -0.0479. The prior helps medium and tail classes most (persistent findings), whereas the Markov layer hurts the tail. Tail n is small; treat as directional.
6. **A-strong:** 100x multiplier did not help. SWA 0.4575 vs v3 0.4600 (logs); full-val Δ -0.0034 (boot [-0.0115, +0.0044]); primary -0.0108.

## Leaderboard verdict
B-early (with priors) and B-late rely on patient history (val priors from train ∪ val). That history does not exist for leaderboard test images, and dev/test must never be joined to the PadChest CSV. Their leaderboard effect is **0**; their internal numbers are a longitudinal simulation only. Leaderboard-usable arms are the A arms and B-early0 (zero prior), all within noise of v3 (full-val Δ: mk1 -0.0063, mk2 -0.0028, mk1fix -0.0044, A-strong -0.0034, B-early0 -0.0026 s42 / +0.0037 s1024; frontal Δ for A-strong +0.0028 [-0.0118, +0.0154]). No usable arm clears the single-model > 0.42 internal gate in a meaningful way over v3, and none is a detectable improvement.

## What it means
- The simple post-hoc late fusion (no retraining, 2 fitted parameters per class plus w) is at least as good as feeding the prior into the network (B-early) on the primary subset (CV-w better by ~0.02 at both seeds), at zero GPU cost. Training the prior into the decoder queries bought nothing beyond late fusion.
- Neither the label-graph layer nor the prior-in-network arm gives a leaderboard gain. The gains are entirely attributable to patient history, unavailable at test time.
- Post-prediction HMM redesign: a variant is being evaluated in analysis/out/markov_hmm/ (job 49445, not finished). Numbers are intentionally not reported here.

## Recommendation
Do not spend further GPU on B-early or on larger A multipliers. If longitudinal history is ever available in a deployment setting, use late fusion with CV-w. For the leaderboard, focus on ensemble/TTA and resolution levers that do not need patient metadata.

## Caveats
- Internal val is not patient-disjoint (77% of val images share a patient with train, 53% their StudyID), which inflates what the prior can do; this is a simulation.
- A has one seed; B-early two seeds; bootstrap n = 300.
- Tail group n = 7; patient-unseen subset n = 80 (CIs span 0).
- Group split is from markov_ab_compare, not re-verified against evaluate/class_groups.py here.
- Known confounds in paper.tex (384-vs-512 resolution, softmax entropy, 7.6%-vs-13% utilization) not touched; v5 logs show ~80% triplet hit rate, which is a differently defined quantity from the paper's 7.6%.

## Not measured
Leaderboard effect of anything; B-late on other base models; calibration comparison per arm beyond ensemble mECE (0.1332 B-early with priors vs 0.1358 zeroed vs 0.1353 6-model v3/v4).

## Notable
- Row order of the reconstructed val_key_order.npy is verified.
- A is dead: larger LR does not help. B's internal gain is unusable on the leaderboard and late fusion matches or beats B-early.

## Next steps
Await the HMM redesign (49445) before any further B-side decisions; update memory campaign note with the above.
