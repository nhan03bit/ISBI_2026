# ISBI 2027 paper: author rewrite fact sheets (2026-10-11)

Source: `docs/paper/isbi_submit/paper.tex`. Produced by the `isbi-ai-minimization` skill.
Authors confirmed (2026-10-11) that **no section's prose was written by them**, so all prose below counts as AI-drafted.

How to use: write each section's sentences yourselves from the bullets, then paste them over the matching
text in `paper.tex`. Keep `\label`, `\cite` keys and `\Macro` names. This sheet holds facts only, not sentences
to copy. Never type a result number: use the macros from `generated/*.tex`. If you cannot verify a bullet, delete it.

Until you have replaced a section, it stays AI-drafted and the disclosure must say so (see the end).

| Section | Words now | Status |
|---|---|---|
| Introduction | 347 | to rewrite |
| Method | 387 | to rewrite |
| Experimental Setup | 421 | to rewrite |
| Results | 211 | to rewrite, and mostly unwritten (TBD) |
| Discussion | 75 | to rewrite, TBD |
| Conclusion | 0 | unwritten, TBD |
| Ethics (43) and Acknowledgments (57) | | authors' own statements; confirm and unwrap `\AUTHOR{}` |

Abstract (not counted above): one TBD sentence on the primary result; the rest is AI-drafted.

## 1. Introduction
Facts a reviewer must find:
- CXR-LT 2026 Task 1, 30 findings, built on PadChest (`dong2026cxrlt`, `bustos2020padchest`).
- Long tail: 3 findings are present in >10% of training images, 7 in <1%; the rarest has 38 positives in 103,303 images.
- Rare findings seldom occur alone, so their gradient is entangled with the common findings they co-occur with.
- Imbalance-aware losses reweight per-label terms and do not act on the shared representation: ASL (`ridnik2021asl`), distribution-balanced loss (`wu2020dbloss`).
- Prior entry (`pham2026efficient`): ConvNeXt (`liu2022convnext`) + ML-Decoder (`ridnik2021mldecoder`), ASL, then a Stage-2 sparse MoE in the top stages and decoder FFNs. Cost: expert and router parameters at inference.
- This paper: regularise the shared embedding at fine-tuning only. Triplet loss (`schroff2015facenet`, `hermans2017defense`) against a cross-batch memory bank (`wang2020xbm`), which supplies positives/negatives for rare findings that a batch of 8 rarely holds.
- Anchors = images with predictive entropy at or above the batch mean (uncertainty criterion from active learning, `settles2009active`).
- Why an ablation: gate, triplet loss, bank and the fine-tuning itself are confounded in the full model's accuracy.
- Contributions (three): (1) the entropy-gated anchor rule, no inference parameters; (2) the like-for-like ablation, arms A/B/B′/C, six seeds, checkpoint/resolution/schedule/augmentation fixed; (3) head/medium/tail AP per arm with seed-level and image-level uncertainty.

## 2. Method
- Model: ConvNeXt-B with a fifth stage of two blocks (total stride 64) + ML-Decoder. Feature map gets 2D positional encoding, projected to 768 channels; 30 **frozen** label queries; two decoder layers (cross-attention then self-attention); group-wise linear head gives logit z_ic. (Verified: queries are frozen; older papers said learnable.)
- Stage 1: end-to-end with ASL at 384 px. Stage 2 (this paper): fine-tunes all trainable weights.
- Embedding e_i = mean of the 30 decoded query embeddings of image i, ℓ2-normalised, 768-d (`train_2_v3.py:840`).
- Bank: FIFO, 2,048 entries (256 batches of 8) of detached embeddings and multi-hot labels, updated after every step; no extra forward cost.
- Gate: Shannon entropy of the **softmax over the 30 logits** (Eq. entropy); anchors = H_i ≥ batch mean (Eq. gate). Known limitation: softmax treats separate binary labels as one categorical; per-label Bernoulli entropy is an untested alternative.
- Measured anchors per step: TBD (from gate rerun logs; smoke runs gave about 2.2-2.3 of 8).
- Positives: bank entries sharing ≥1 finding with the anchor other than the five most frequent (Normal, pleural effusion, cardiomegaly, support devices, aortic elongation); D = 0 at those five. Negatives: share no finding.
- Mining: closest positive, farthest negative (Euclidean on ℓ2-normalised embeddings, same ranking as cosine). These are the *easiest* pairs, not the hardest (`hermans2017defense`); harder mining was not compared. Fig. xbm's caption currently calls them "hardest"; the caption itself says that is the usual mismatch.
- Loss: hinge, margin α = 0.5, averaged over anchors with non-empty P and N; total = ASL + λ·triplet, λ = 0.1. ASL: γ− = 2, γ+ = 0, probability margin 0.05, positive-term class weights ∝ log(1/f_c), normalised to mean 1, clipped to [0.5, 2].
- Bank, gate and triplet term are training-only.

## 3. Experimental setup
- Data: 30-finding CXR-LT 2026 Task 1 training data; 103,303 train / 17,270 validation images; image-level split (77% of val images share a patient with train; 30% lateral). All results on this validation fold; not scored on the challenge test set.
- Training (identical for all arms): from the Stage-1 checkpoint; 512×512; AdamW lr 1e-4, wd 0.01; 5% linear warm-up, cosine over 8 epochs, stepped per update; batch 8, gradient accumulation 6; mixed precision; EMA decay 0.999 (EMA weights are evaluated); stop after 3 epochs without val-mAP gain, keep best epoch.
- Augmentation (random resized crop, flip, affine, distortion, brightness/contrast, noise/blur) is derived deterministically from image identity + epoch; seed sets data order. Six seeds (42, 86, 123, 456, 789, 1024); same RNG stream across arms at a given seed.
- Arms: A = ASL only (λ = 0; bank and mining run, no gradient). B = triplet+bank, every image an anchor. B′ = triplet+bank, |A| random anchors, |A| = the count the gate would pick on that batch, separate generator. C = the method.
- Metrics: macro mAP (primary, challenge metric), mAUC, mF1 at 0.5, mECE (15 bins, all image-finding pairs). Head >10% (3), medium 1-10% (20), tail <1% (7) by training prevalence, following CXR-LT 2024 (`lin2025cxrlt`); mean AP per group. Tail val positives: 7 to 140 per finding.
- Statistics: per-run best-epoch EMA model, selected on val in the same way for all arms; mean ± s.d. over seeds; paired difference over seeds with 95% t-interval and exact two-sided sign-flip test; primary contrast fixed before the runs: C − B in mAP; image-level bootstrap (2,000 resamples) of seed-averaged probabilities as a check.
- Pre-registration: `docs/plan/2026-10-08-entropy-ablation-isbi-submit-plan.md`.

## 4. Results (cannot be written until the ablation finishes)
Numbers come from `generated/entropy_ablation_{table,macros}.tex`; none exist yet.
Needed, in the pre-registered order: (1) B − A; (2) C − B with 95% CI, sign-flip p, wins; C − B′; (3) head/medium/tail; (4) calibration; (5) anchors and triplets per step; (6) reproducibility of C (reruns s86, s1024); (7) frontal and unseen-patient subsets, one sentence; (8) paired-difference table if space allows.
Already available (C arm, 6 seeds, 2026-10-08): mAP 0.4394 ± 0.0025; head 0.7544, medium 0.4439, tail 0.2919 ± 0.0103.
Higher-resolution paragraph facts: 768/1024 px single models 0.460/0.462 mAP; 3-model ensemble (640-1024 px) with flip TTA 0.472, members chosen on val (optimistic); all used the entropy gate and were not ablated.
MoE paragraph facts: MoE rebuilt with the same recipe as the high-resolution runs (same Stage-1 checkpoint, seeds, schedule, weight averaging, flip TTA; 4 experts, top-2, ASL + load balancing, no triplet) plus an ASL-only control at 768 px. Mean differences and CIs are the `\Moe…` macros: triplet−MoE mAP, triplet−ASL, MoE−ASL, mECE. Table `tab:moe`.

## 5. Discussion and limitations (TBD interpretation)
Facts to state: single internal val fold shares patients with train and includes lateral views, challenge test is frontal and patient-disjoint, so absolute values are optimistic for every arm alike; tail estimates rest on few positives; softmax entropy over binary labels; no harder-mining comparison; one dataset. Interpretation follows the pre-registered outcome and cannot be drafted before results.

## 6. Conclusion
Unwritten; depends on the outcome of C − B.

## Disclosure (Acknowledgments), state today
Everything above is still AI-drafted, so the disclosure must keep naming the text drafting. Proposed wording for the authors to edit and confirm:

> Generative AI (Claude, Anthropic) was used to write the training and analysis code of the ablation and to draft and edit the text of this paper. The authors designed the study, ran and verified the experiments, and reviewed and take responsibility for all content.

Put it in its own sentence, apart from the funding/COI statement, and remove the `\AUTHOR{}` wrapper only after confirming.
After you replace sections with your own prose, change the text part to name only the sections that remain AI-drafted, or to "AI-assisted grammar editing" if that is all that is left (never covering the reference list). The code part stays.
