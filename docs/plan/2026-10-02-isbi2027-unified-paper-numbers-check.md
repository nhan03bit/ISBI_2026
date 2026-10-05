# Numbers check of the unified ISBI 2027 paper (paper.tex + supplement.tex) — 2026-10-02

**Type:** findings note (offline cross-check, not a training run).
**Files checked:** `docs/paper/isbi/paper.tex` (main, 4 pages), `docs/paper/isbi/supplement.tex` (13 pages).
**Source drafts:** `docs/paper/entropy/paper.tex`, `docs/paper/markov/paper.tex`.
**Related:** [047 provenance and paper corrections](2026-10-02-047-provenance-and-paper-corrections.md), [push wave](../result/2026-09-26-push-wave.md), [v4 Markov result](../result/2026-09-27-v4-markov-result.md), [Markov A/B training](../result/2026-10-01-markov-ab-training.md), [Markov applicability](2026-09-26-markov-applicability.md).
**Slurm jobs read:** 43047-43051 (original Table I), 43998-44002 (under-trained Stage-1), 44274-44278 (hit-rate re-runs), 46471-46476 and 47097-47099 (v3 512 px), 47094-47096 (640/768 px), 48997-49000 (push wave), 49063-49065 (Markov layer). Logs in `/home/psytp7/logs/`.

## Question
Does every numeric claim in the unified main paper match (a) the two source drafts and (b) the primary artifacts (Slurm logs, `history.json`, `analysis/out/*.json`, the published challenge paper)? Also: is the Table I mF1/mAUC column order right, do the "Supp. S#" pointers name the right supplement section, and are the honesty caveats present?

## Data and method
- Logs and histories: Slurm logs (`Done. Best`, `SWA weights ->`, `triplet hit rate`, `args:`); `history.json` under `train/checkpoint3`, `checkpoint_Triplet_2/accum_6`, `checkpoint_Triplet_3/{seedsweep,push,res_sweep,markov}`.
- Analysis artifacts: `analysis/out/ens_select_w01_flip.json`, `tta_w1_flip.json`, `markov_hmm/results.json`, `metadata_eda/eda.json`.
- Challenge paper: `docs/pdf/ISBI2026.pdf` (b12) for the Stage-1, MoE and test numbers.
- Re-runs: `analysis/markov_check.py` end to end (smoothing, gating, patient counts, validation subsets; output identical to the paper); a 300-resample patient-clustered bootstrap of the 1024 px vs 512 px difference; per-class gating counts; Normal/abnormal co-occurrence count.
- Code read for method claims: `train/train_2_v2.py`, `train_2_v3.py`, `utils_update.py`, and the Aug-26 snapshot `train_2_v2.py.bak2`.
- Token check: every numeric token in paper.tex (outside the bibliography) was searched in the entropy draft, the Markov draft and the supplement. All occur in at least one of them. The only unmatched tokens are artefacts (`13--14%` is a rounding of `13.1--14.0%`; `2,048.` and `768,` carry punctuation; `208` is a LaTeX `trim` option). This cannot catch a number in the wrong context, so every table cell and prose claim was also compared with the artifacts below.
- Supplement HMM table (59 numeric cells) compared programmatically against `markov_hmm/results.json`: no differences.
- Status key: **OK** = matches source and artifact. **MISMATCH** = wrong value or wrong statement. **CAVEAT** = number is right but wording or scope needs care.

## Findings

### Summary
**6 MISMATCHes** (4 in paper.tex, 2 in supplement.tex; the paper.tex s.d. error also repeats in supplement Table S1). 5 of the 6 are inherited from the drafts; only the S8 job range is new in the supplement. No transcription error between drafts and main paper.

| # | Location | Issue |
|---|---|---|
| M1 | paper.tex L93 | "(without ℓ2 normalization)". The code ℓ2-normalises the mean-pooled embedding (`train_2_v3.py:840`, `train_2_v2.py:672`), re-normalises anchor, positive and negative before the loss (`train_2_v3.py:852-854`), and the bank stores the normalised vectors. Inherited from entropy draft L103. Method statement, not a number. |
| M2 | paper.tex L151 | Original recipe "cosine schedule". All five Aug-27 `history.json` files have `lr = [0.0]` after one epoch, i.e. a `CosineAnnealingLR(T_max=1)` stepped once per epoch (`train_2_v2.py.bak2:547,711`). The LR was a constant 3e-5 during the epoch. Inherited from entropy draft L166. |
| M3 | paper.tex L174 (and Table S1 L95) | v3 six-seed mAP "±0.003". Sample s.d. is 0.00246 (population 0.00225); both round to 0.002. Inherited from entropy draft Table III. |
| M4 | paper.tex L213 | Metadata stacking "lowers mAP by 0.003 to 0.012". True only for the all-image cells (−0.0026 to −0.0123). The 16 cells span 0.0026 to 0.0158 (ensemble, frontal, all three groups). Inherited from the Markov draft abstract. |
| M5 | supplement L147 (S3-B) | Later-epoch hit rate "11.9–12.6%". Logs give 11.9–12.8% (seed 456, epoch 2: 1657/12912 = 12.8%). Inherited from entropy draft L239. |
| M6 | supplement L690 (S8) | "Table S2: jobs 44000–44002". The five under-trained seeds are jobs 43998–44002. Newly written in the supplement. |

No mF1/mAUC swap in Table I (all 9 rows checked value by value). All 6 "Supp. S#" pointers in paper.tex name the right supplement section. The headline numbers (Tables I and II, the validation proxy) otherwise reproduce from artifacts.

### Claim-by-claim table: claim | location | value in paper | source value | status

#### Table I (paper.tex L162-184)
| Claim | Location | Paper | Source | Status |
|---|---|---|---|---|
| Stage 1 @384 (mAP / mF1 / mAUC / mECE) | Tab I row 1 | 0.385 / 0.395 / 0.875 / 0.134 | b12 Table 1 (columns mAP, mF1, mAUC, mECE): same. `checkpoint3/Model_20260119_062652/history.json`, best epoch 6: 0.3849 / 0.3946 / 0.8746 / 0.1338 | OK |
| MoE @384 | Tab I row 2 | 0.405 / 0.402 / 0.882 / 0.130 | b12 Table 1 stage 2: same. No MoE checkpoint in the repo, so b12 is the only source | OK |
| Original code, 5 seeds, mAP | Tab I row 3 | 0.414 ± 0.002 | jobs 43047-51 `Done. Best mAP`: 0.4139, 0.4129, 0.4108, 0.4156, 0.4155; mean 0.41375, sample s.d. 0.0020 | OK |
| Original, mF1 / mAUC / mECE | Tab I row 3 | 0.407 / 0.889 / 0.143 | per-seed mF1 0.4195, 0.4113, 0.3932, 0.4045, 0.4041 (mean 0.4065); mAUC mean 0.8889; mECE mean 0.1433 (s.d. 0.0083) | OK |
| Original: 512 px, batch 8, accum 6, lr 3e-5, wd 0.01, 96 slots | Tab I, L96, L151 | as stated | log config dict (lr 3e-5, wd 0.01, batch 8, accum 6, memory 96, epochs 1); `SIZE = 512` in `train_2_v2.py` | OK |
| v3 512 px, 6 seeds, mAP mean | Tab I row 4 | 0.439 | seedsweep best epochs 0.4434, 0.4389, 0.4414, 0.4375, 0.4369, 0.4385; mean 0.43945 | OK |
| v3 512 px, mAP s.d. | Tab I row 4 | ±0.003 | sample s.d. 0.00246 (population 0.00225) | **MISMATCH (M3)** |
| v3 512 px, mF1 / mAUC / mECE | Tab I row 4 | 0.454 / 0.900 / 0.128 | 0.4537 / 0.89995 / 0.1282 | OK |
| v3 768 px SWA seed 42 | Tab I row 5 | 0.4600 / 0.4685 / 0.9092 / 0.1354 | `push/img768_dp01_seed42` `swa_metrics` 0.45997 / 0.46851 / 0.90924 / 0.13538; log 49000 SWA line identical | OK |
| v3 1024 px SWA seed 86 | Tab I row 6 | 0.4619 / 0.4679 / 0.9138 / 0.1438 | `push/img1024_dp01_seed86` `swa_metrics` 0.46189 / 0.46793 / 0.91383 / 0.14384; log 48997 | OK |
| 768 px seed-42 SWA + flip | Tab I row 7 | 0.4650 | `ens_select_w01_flip.json` single: 0.46499 | OK |
| Fixed 4-model average, flip | Tab I row 8 | 0.4699 / 0.4819 / 0.9154 / 0.1385 | `tta_w1_flip.json`: 0.46993 / 0.48191 / 0.91539 / 0.13848; checkpoints 1024 s86, 896 s86, 768 s42, 768 s1024; flip true | OK |
| Greedy 3-model ensemble, flip | Tab I row 9 | 0.4722 (others "--") | `ens_select_w01_flip.json`: ensemble_mAP 0.472238, n_val 17,270; members 768 s42 SWA, 1024 s86 SWA, 640 s86 best epoch; no other metrics stored | OK |
| Column order mAP, mF1, mAUC, mECE | Tab I header and all 9 rows | mF1 before mAUC | each row's two values compared with its artifact; the entropy draft's Table III order (mAP, mAUC, mF1, mECE) was translated correctly in every row | OK (no swap) |
| px column | Tab I | 384, 384, 512, 512, 768, 1024, 768, mixed, mixed | args `--img-size`; 4-model = 768/768/896/1024; greedy = 768/1024/640 | OK |
| Validation size | Tab I caption, L149 | 17,270 | `labels.npy` 17,270 (`val_fold_1.csv` has 17,274 rows; 4 images absent from the shards) | OK |

#### Abstract, introduction, conclusion
| Claim | Location | Paper | Source | Status |
|---|---|---|---|---|
| Test mAP and third place | Abstract, Intro | 0.4599, third | ISBI2026.pdf Table 3 (Stage 2, test): 0.4599; "securing third place" | OK |
| 0.385 → 0.414, five seeds | Abstract | as stated | Stage-1 0.3849; original 0.41375 | OK |
| Five differences | Abstract, L157 | five | S1 lists five; code confirms each | OK |
| 0.460 single / 0.472 ensemble | Abstract, contributions, Conclusion | 0.460 / 0.472 | 0.45997 (768 px s42 SWA, no TTA) / 0.47224 | OK |
| "after correcting them" (all five) | Abstract L55, Conclusion L220 | all five corrected | L157 says "v3 fixes the first two"; anchor rule, positive threshold (now non-head) and easy mining are unchanged in the code, and the text was rewritten to describe them | CAVEAT |
| "0.460 as a single model" | Abstract, contributions | 0.460 | 0.460 is the 768 px seed-42 model (the Markov baseline). The best single model in Table I is 0.4619 (1024 px). Conservative, not an overstatement | CAVEAT |
| Which mAP is 0.460 / 0.472 | Abstract | unlabelled | only the 0.385 → 0.414 sentence says "internal-validation"; the 0.460/0.472 sentence has no label and follows "test mAP 0.4599" directly, so a test score and an internal score that differ by 0.0001 sit side by side. The internal-to-leaderboard offset is large (Stage-1: 0.385 internal, 0.505 development leaderboard, per b12) | CAVEAT |

#### Method and setup (paper.tex L84-157)
| Claim | Location | Paper | Source | Status |
|---|---|---|---|---|
| Bank size, original / corrected | L96 | 96 / 2,048 | log config `memory_size 96`; v3 args `--memory-size 2048` | OK |
| e ∈ R^768, "without ℓ2 normalization" | L93 | no ℓ2 norm | `train_2_v3.py:840` `F.normalize(embeddings, dim=1)` after the mean pool; L852-854 normalise anchor/pos/neg again; `memory.update(embeddings.detach(), …)` stores normalised vectors; same in `train_2_v2.py:672,688-690` | **MISMATCH (M1)** |
| Softmax entropy over 30 logits; anchors at or above batch mean | L99-109 | as stated | `calculate_entropy_from_logits` default `softmax`; `if entropy[i] < entropy_thr: continue` | OK |
| Positives share ≥1 label outside the five most frequent; negatives share none | L113-118 | as stated | `sample_triplets_v13`; `--head-classes 5` in all v3 args | OK |
| Easy mining (closest positive, farthest negative) | L120-123 | as stated | `argmin(pos_d)`, `argmax(neg_d)` | OK |
| λ = 0.1, margin α = 0.5 | L131 | 0.1, 0.5 | `--triplet-lambda 0.1`; cfg margin 0.5; `TripletMarginLoss(p=2)` | OK |
| ASL γneg 2, γpos 0, clip 0.05, weights in [0.5, 2] | L131 | as stated | cfg gamma_neg 2.0 / gamma_pos 0.0; `clip=0.05`; `np.clip(class_weights, 0.5, 2.0)`; log 49000 (shard order): Hydropneumothorax 2.000, Normal 0.500 | OK |
| Training fold size | L149 | 103,303 | `data/CXRLT_2026_training_filtered.csv` has 103,303 rows; `data/train_fold_1.csv` has 103,305 rows, of which 103,303 are in the filtered file; batches per epoch at batch 8 are 12,912–12,914 (≈103.3k images). `--train-size 103300` in the scripts is only a constant for the LR-schedule step count | OK |
| Original recipe: AdamW, lr 3e-5, wd 0.01, batch 8, accum 6, 512 px, five seeds | L151 | as stated | job names, args, config dict | OK |
| Original recipe: "cosine schedule" | L151 | cosine | all five Aug-27 histories: `lr = [0.0]`, 1 epoch, from `CosineAnnealingLR(T_max=1)` stepped per epoch; LR constant at 3e-5 during training. Logs 43047 and 44274 have no `LR schedule: warmup` line (v3 log 46472 does). The exact Aug-27 source file is not preserved; the history `lr` is the direct evidence | **MISMATCH (M2)** |
| v3: lr 1e-4, bank 2,048, EMA 0.999, 8-epoch cosine, early stopping | L153 | as stated | args `--lr 1e-4 --memory-size 2048 --sched-epochs 8 --patience 2/3`; `--ema-decay` default 0.999. v3 uses a per-step cosine with about 5% linear warmup (log 46472: warmup 860 of 17,217 steps), which the paper does not mention | OK |
| v3 at 768/1024 px: batch 4, accum 12, four of eight epochs, drop-path 0.1, SWA over last three EMA epochs | L153 | as stated | args `--batch-size 4 --accum-steps 12 --epochs 4 --sched-epochs 8 --drop-path-rate 0.1 --swa-last-k 3`; `swa_epochs [2,3,4]` | OK |
| Stage-1 and MoE at 384 px, ours at 512 px | L157 | 384 vs 512 | ISBI2026.pdf "384x384 during training and evaluation"; `SIZE = 512` | OK |
| v3 fixes items 1 and 2 | L157 | embedding, class-weight order | every v3 args line has `--metric-embed query --class-weight-order shard` | OK |
| All Table I rows start from the same Stage-1 checkpoint | Tab I caption | same | v3 and re-run histories record `checkpoint3/Model_20260119_062652/model_best.pth`. The Aug-27 original runs record no checkpoint (no `stage1_ckpt` key, no "Stage-1 init" log line); the Aug-26 code snapshot loads that file and the mAP level matches that lineage | CAVEAT (indirect for the original runs) |

#### Results prose, entropy stage (paper.tex L186-188)
| Claim | Location | Paper | Source | Status |
|---|---|---|---|---|
| Gain over Stage 1 | L186 | +0.029 | 0.41375 − 0.3849 = +0.0289 | OK |
| Gain over MoE | L186 | +0.009 | 0.41375 − 0.405 = +0.0088 | OK |
| Seed s.d. | L186 | 0.002 | 0.0020 | OK |
| mECE | L186 | 0.134 → 0.143 | 0.1338 → 0.1433 | OK |
| Less converged Stage-1: gain halves | L186 | +0.016 | jobs 43998-44002 mean 0.3931 vs `stage1_full30` 0.3771 = +0.0159 (vs +0.0289 for the converged one) | OK |
| 96-slot bank: valid triplet in 13–14% of steps | L186 | 13–14% | jobs 44274-78 epoch 1: 13.2, 13.1, 14.0, 13.5, 13.5 (13.1–14.0%) | OK |
| Provenance of that rate | L186 | not stated | Table I jobs 43047-51 logged no hit rate. The rate comes from re-runs 44274-78, which use `--epochs 3 --patience 2` (not the 1-epoch recipe) and log 11.9–12.8% in epochs 2-3 | CAVEAT |
| v3 hit rate | L186 | 74–97% | v3 logs (smoke runs excluded): 640–1024 px, batch 4: 74.0% (47094 ep 6) to 82.3% (49000 ep 1); 512 px, batch 8: 91.2% (46473 ep 6) to 96.6% (46472 and 46476 ep 1). Union 74.0–96.6% | OK |
| The two ranges compared | L186 | "13–14% … raises this to 74–97%" | 13–14% is epoch 1 only; 74–97% spans all epochs. Epoch-1-only v3 range is 81.2–96.6% | CAVEAT |
| v3 mean mAP at 512 px | L188 | 0.439 | 0.43945 | OK |
| Resolution gain at 768 px | L188 | about +0.017 | seed 86: 0.4560 (`res_sweep/img768_seed86`) − 0.4385 (`seedsweep/seed_86`) = +0.0174 | OK |
| Stochastic depth with SWA | L188 | +0.003 to +0.005 | no seed-matched 768 px pair exists. 768 px s86 no-dp SWA 0.4566 vs dp SWA s42 0.4600 (+0.0034) and s1024 0.4582 (+0.0016); +0.0053 uses the 1024 px model, so resolution is mixed in. The one clean same-seed A/B is 512 px s86, +0.0050 best epoch (push-wave note) | CAVEAT |
| Flip TTA | L188 | +0.005 | 0.46499 − 0.45997 = +0.0050 | OK |
| Greedy ensemble over the flip single model | L188 | +0.007 | 0.47224 − 0.46499 = +0.0073 | OK |
| Fixed average | L188 | 0.4699 | 0.46993 | OK |

#### Markov (paper.tex Table II L192-211, prose L213)
| Claim | Location | Paper | Source | Status |
|---|---|---|---|---|
| Smoothing α_s 0.02 / 0.20 | Tab II | −0.0006 / −0.0053 | `markov_check.py` re-run: 0.4716 (−0.0006) / 0.4669 (−0.0053) from 0.4722 | OK |
| Normal-gating α_ng 0.25 / 0.5 | Tab II | −0.0103 / −0.0239 | re-run: 0.4619 (−0.0103) / 0.4483 (−0.0239); α = 1.0 gives −0.0459 | OK |
| Trainable layer mk1 / mk2 / mk1fix, SWA | Tab II | −0.0049 / −0.0033 / −0.0031 | `swa_metrics` 0.45510 / 0.45664 / 0.45691 vs baseline 0.45997 | OK |
| Baseline for the layers | Tab II caption | matched run, SWA 0.4600, 768 px s42, no TTA | job 49000; layer args identical apart from `--markov-*` flags and `--markov-lr-mult 10` | OK |
| Patient HMM, frontal with prior study | Tab II | +0.0375 [+0.023, +0.051] | `markov_hmm/results.json` s42 H4 primary: 0.03752 [0.02283, 0.05125]; n = 4,061 | OK |
| Same, image-only history | Tab II | +0.0092 [−0.001, +0.018] | H4-F s42: 0.0092 [−0.0013, 0.0181] | OK |
| Leaderboard row | Tab II | 0 | by construction (test patients have no history); not a measurement | OK |
| Smoothing loss at α = 0.05 mainly on head classes | L213 | head | per-class dAP head-10 −0.0021, tail-10 −0.0001 | OK |
| Normal-gating lowers 28 of 30 classes | L213 | 28 of 30 | re-computed: α = 0.5 and 1.0 give 28 worse, 1 better, 1 unchanged (Normal); α = 0.25 gives 26 worse. The text does not say which α | CAVEAT |
| Normal co-occurs with an abnormal label in 26 of 17,270 | L213 | 26 | 26 | OK |
| Layer gates stay near zero | L213 | mean abs gate ≤ 0.013 | final epoch: mk1 0.0129, mk2 0.0111, mk1fix 0.0131 (max 0.0613 / 0.0466 / 0.0613) | OK |
| Shortfall within seed noise; baseline seeds differ by about 0.002 | L213 | as stated | s42 vs s1024 at 768 px: best 0.4585 vs 0.4564 (0.0021), SWA 0.4600 vs 0.4582 (0.0018). The shortfalls (0.0031–0.0049) are 1.5–2.5 times that gap. "Within noise" holds against the 512 px six-seed range (0.0065, s.d. 0.0025). The supplement words it more carefully ("cannot tell no effect from a slight harm") | CAVEAT |
| Metadata alone, held-out macro AUROC | L213 | 0.76 | 0.7600 (`metadata_eda/run.log`) | OK |
| Stacking metadata lowers mAP | L213 | by 0.003 to 0.012 | 16 cells (`eda.json` `incremental`): all-image −0.0026 to −0.0123; s42 frontal −0.0028 to −0.0106; ensemble frontal −0.0044 to −0.0158; 15 of 16 intervals exclude 0 | **MISMATCH (M4)** |

#### Validation proxy and conclusion (paper.tex L217, L220)
| Claim | Location | Paper | Source | Status |
|---|---|---|---|---|
| Validation images sharing a patient with training | L217 | 77% | 0.770 (10,978 shared patients; `markov_check.py`) | OK |
| Lateral share | L217 | 30% | 5,234 / 17,270 = 30.3% | OK |
| Ensemble: frontal-unseen / all frontal / lateral | L217 | 0.4752 / 0.4968 / 0.4058 | re-run: 0.4752 / 0.4968 / 0.4058 (n 3,572 / 11,973 / 5,234) | OK |
| 1024 vs 512 px gain, seen → unseen; significance | L217 | +0.020 → +0.007, not significant | bootstrap re-run (300 resamples, seed 0): +0.0196 [+0.0047, +0.0358] seen; +0.0075 [−0.0201, +0.0282] unseen (documented: +0.0074 [−0.0200, +0.0269]; the interval includes 0) | OK |
| Point estimate vs bootstrap mean | L217 | +0.020 | table point estimates: 0.4930 − 0.4750 = +0.0180 (seen); 0.4719 − 0.4649 = +0.0070 (unseen). +0.020 is the bootstrap mean | CAVEAT |
| What is compared | L217 | "1024 over 512 px" | 1024 px s86 SWA with drop-path 0.1 vs 512 px s1024 best epoch without drop-path or SWA; seed, regulariser and weight averaging differ along with resolution | CAVEAT |
| Conclusion 0.460 / 0.472 | L220 | as stated | as above | OK |

#### "Supp. S#" pointers in paper.tex (checked against the compiled `supplement.aux`)
| Pointer | Location | Intended content | Compiled section | Status |
|---|---|---|---|---|
| Supp. S4 | L145 | patient HMM full method | S4 (HMM is S4-F, label `sec:hmm`) | OK |
| Supp. S1 | L157 | corrections, job identifiers | S1 (jobs 43047-51 in the text) | OK |
| Supp. S3 | L186 | Stage-1 maturity and utilisation | S3 (S3-A maturity, S3-B utilisation) | OK |
| Supp. S2 | L188 | v3 progression | S2 | OK |
| Supp. S6 | L213 | metadata | S6-E (`sec:metaeda`) | OK |
| Supp. S6 | L217 | validation subsets, bootstrap | S6-F (`sec:audit`) | OK |

Section map from `supplement.aux`: S1 corrections, S2 v3, S3 maturity/utilisation, S4 Markov full method, S5 Markov setup, S6 Markov results (S6-D HMM, S6-E metadata, S6-F validation subsets), S7 discussion, S8 reproducibility. Both PDFs compile with no undefined references.

#### Supplement items checked (focus was the main paper; not exhaustive)
| Claim | Location | Value | Source | Status |
|---|---|---|---|---|
| HMM table, 59 numeric cells (E, H0–H5, H4-acq, H4-F, B-early, B-early0; seeds 42 and 1024; full, primary, no_prior) | L491-502 | as printed | `markov_hmm/results.json`, compared programmatically | OK |
| Smoothing, gating and subsets tables | S6 | as printed | `markov_check.py` re-run, identical | OK |
| Under-trained Stage 2 | Tab S2 | 0.393 / 0.379 / 0.886 / 0.151, mF1 s.d. 0.014 | jobs 43998-44002: 0.3931 / 0.3792 / 0.8858 / 0.1511; mF1 s.d. 0.0137 | OK |
| Gains vs under-trained Stage-1 | S3-A | mAP +0.016, mF1 +0.003, mAUC +0.007, mECE +0.003 | +0.0159, +0.0032, +0.0066, +0.0035 | OK (mECE sits on the 0.003/0.004 boundary) |
| Later-epoch hit rate, 96-slot bank | S3-B L147 | 11.9–12.6% | 11.9–12.8% (epoch 2: 12.5, 12.6, 12.8, 12.5, 12.6; epoch 3: 11.9, 11.9, 12.4, 12.0, 12.2) | **MISMATCH (M5)** |
| v3 hit rates | S3-B | 74–82% at 640–1024 px; 97% at 512 px | 74.0–82.3%; 512 px epoch 1 is 96.4–96.6% (91.2% by epoch 6) | OK |
| Under-trained job IDs | S8 L690 | 44000–44002 | 43998–44002 (five seeds) | **MISMATCH (M6)** |
| v3 six-seed s.d. | Tab S1 L95 | ±0.003 / 0.002 / 0.004 / 0.006 | 0.00246 / 0.00199 / 0.0042 / 0.00623 | **MISMATCH** (mAP s.d. only; same as M3) |
| Stage-1: 0.385 internal vs 0.505 development leaderboard | S6-F | as stated | ISBI2026.pdf: "mAP of 0.505 on the Development set … the mAP is around 0.385" | OK |
| "the temporal work in Section II" | S6-D L465 | main-paper Section II | the main paper's Related Work has no temporal-CXR paragraph, so the pointer lands on nothing | CAVEAT (non-numeric) |
| Bibliography b11 author list | paper.tex vs entropy draft | "Xuan Zhong Feng" vs "Fengnian Zhao" | main paper follows the Markov draft; the two drafts disagree | CAVEAT (non-numeric) |

### Honesty caveats in paper.tex
| Caveat | Present | Weak or absent |
|---|---|---|
| 0.4722 is selection-biased; 0.4699 is the reference | Setup L153; Table I caption; Results L188 (explicit 0.4699) | Abstract, contribution bullet and Conclusion quote 0.472 with no selection note. In Table I the bold marks the selection-biased 0.4722, not 0.4699 |
| The fixed 4-model average is "not" selected | Table I caption | True for ensemble selection, but the same fold drove early stopping, best-epoch choice and resolution/drop-path choices, so 0.4699 is less optimistic, not unbiased (047 note says the same) |
| No gate ablation | Abstract; Results L188; Conclusion | none (not in the contribution bullets; acceptable) |
| Not leaderboard-scored | Results L188; Conclusion | Abstract never says it, and does not label 0.460/0.472 as internal (see CAVEAT above) |
| Validation fold shares patients with training | Abstract; Sec V-C; Conclusion | none |
| Patient HMM unusable on the test set | Contributions; Table II caption; Results L213; Conclusion | Abstract says "only in an internal simulation that reads the previous report" without "unusable on the test set" |
| 384 vs 512 px confound | Setup L157; Results L186; px column | none |
| Easy (not hard) mining; softmax entropy | Method L109, L123 | none |
| Original-code row predates both known bug fixes (class-weight order, embedding) | Setup L157 lists both | Table I caption does not mark the original-code row as pre-fix |

## Recommendation
Edit paper.tex, and the matching lines in the entropy draft and supplement:
1. L93: replace "(without ℓ2 normalization)" with a statement that e is ℓ2-normalised before it goes to the bank and the loss (entropy draft L103 too). The margin 0.5 then lives on the unit sphere.
2. L151: replace "cosine schedule" for the original recipe with "constant learning rate 3×10⁻⁵ (a cosine scheduler was configured but stepped once per epoch, so it had no effect inside the single epoch)"; say that v3 uses a per-step cosine with about 5% warmup (entropy draft L166 too).
3. L174, supplement Table S1 L95 and entropy draft Table III: ±0.003 → ±0.002 for the v3 six-seed mAP s.d.
4. L213 and the Markov draft abstract: "by 0.003 to 0.016", or restrict to "on all validation images (0.003 to 0.012)".
5. Supplement L147: "11.9–12.8%". Supplement L690: "jobs 43998–44002".
6. Abstract and Conclusion: replace "after correcting them" with "after fixing two (triplet embedding, class-weight order) and re-describing three".
7. Abstract: label 0.460 and 0.472 as internal-validation mAP, add "selected on that fold; fixed-average reference 0.4699" and "not yet scored on the leaderboard".
8. L186: say the 13–14% comes from 3-epoch re-runs, and compare epoch with epoch (v3 epoch 1: 81–97%) or state "74–97% over all epochs".
9. L188: reword the stochastic-depth gain as "+0.002 to +0.005, confounded with seed and resolution".
10. L213: reword "within seed noise" as in the supplement; name α for "28 of 30" (α ≥ 0.5).
11. Table I caption: mark the original-code row as pre-fix; say that 0.4699 is also scored on the early-stopping fold; consider un-bolding 0.4722.
12. L217: give "bootstrap mean +0.020 (point estimate +0.018)", and say that the two models also differ in drop-path, SWA and seed.

## Caveats
- The MoE row cannot be traced to an artifact in this repo; the published challenge paper is the only source.
- The Stage-1 checkpoint of the Aug-27 original runs is not recorded in their logs or histories. The Aug-26 code snapshot loads `Model_20260119_062652` and the 0.414 level is consistent with it, but the evidence is indirect.
- The exact Aug-27 source file of `train_2_v2.py` is not preserved. M2 rests on the five `history.json` `lr = [0.0]` values and the Aug-26 snapshot.
- The 13–14% hit rate comes from 3-epoch re-runs, so it only approximates the 1-epoch Table I recipe.
- The bootstrap was re-run with a different RNG stream; means and lower bounds agree with the documented values to 0.0001–0.0002, the unseen upper bound differs by 0.0013. The paper only claims that the interval includes 0.
- External literature claims (the Task-1 winner's gating, DB loss, class-aware sampling, challenge de-duplication rules) were not checked against b14 and b15.

## Not measured
- Leaderboard score of any v3 model or ensemble.
- A triplet-off or gate-off v3 control.
- Per-class head/medium/tail results for v3.
- Hit rate of the original Table I jobs (never logged).

## Notable
- Two method statements in both the main paper and the entropy draft contradict the code (ℓ2 normalisation, cosine schedule); both survived the earlier correction pass. The constant-LR finding for the original recipe is new.
- The abstract's unlabelled 0.460/0.472 sits next to the test mAP 0.4599. Internal and leaderboard scores differ by a large offset (Stage-1: 0.385 vs 0.505), so this juxtaposition invites a wrong comparison.
- The headline numbers (Tables I and II, the validation proxy) all reproduce from artifacts.

## Next steps
- Apply the edits above, rebuild both PDFs, and re-run this check on the changed lines.
- Run a v3 `--triplet-lambda 0` control at 768 px if an entropy-gate claim is wanted.
- Score the v3 ensemble on the leaderboard once, so the v3 internal-to-leaderboard offset is known.
