# Stage-2 v4 Markov label layer (mk1 / mk2 / mk1fix) — 2026-09-27

**Directory:** `/home/psytp7/ISBI_2026/train/checkpoint_Triplet_3/markov/img768_{mk1,mk2,mk1fix}_seed42/Model_run/` (the `*_smoke` siblings are ignored)
**Script / recipe:** `train/train_2_v4.py` + `train/markov_layer.py`: the v3 push-wave recipe (768 px, dp 0.1) plus the Markov head
**Slurm jobs:**
- 49063 mk1 (`isbi2026_v4markov_mk1_s42_49063.out`)
- 49064 mk2 (`..._mk2_s42_49064.out`)
- 49065 mk1fix (`..._mk1fix_s42_49065.out`)
- Baseline: 49000 (`isbi2026_v3push_img768_dp01_s42_49000.out`, also in `docs/result/2026-09-26-push-wave.md`)

All logs are in `/home/psytp7/logs/`. `L<n>` means line n of the named `.out` log.

All metrics are internal-val only. The val split is not patient-disjoint (see project memory). Nothing here is leaderboard or test.

## Configuration
Each run's `args:` line is L2 of its log.

| Setting | Value (all four runs) |
|---|---|
| Stage-1 checkpoint | checkpoint3/Model_20260119_062652/model_best.pth |
| Image size | 768 |
| Seed | 42 |
| LR / batch / accum | 1e-4 / 4 / 12 (effective batch 48), epochs 4, patience 2, sched-epochs 8, swa-last-k 3 |
| Drop-path | 0.1 |
| metric-embed / memory-size / triplet-lambda | query / 2048 / 0.1 |
| class-weight-order | shard (fixed) |
| Markov flags (v4 only) | `--markov-lr-mult 10`; mk1: `--markov-steps 1`; mk2: `--markov-steps 2`; mk1fix: `--markov-steps 1 --markov-fixed-transition` |

## Results

### Per-epoch val metrics (EMA weights per epoch)
Each cell is mAP / mAUC / mF1 / mECE. The Epoch 01–04 lines are at:
- mk1: L108 / 154 / 200 / 246
- mk2 and mk1fix: L108 / 154 / 200 / 245
- 49000: L106 / 151 / 196 / 241

| Ep | 49000 (v3) | mk1 | mk2 | mk1fix |
|---|---|---|---|---|
| 1 | .4370/.9057/.4344/.1536 | .4365/.9063/.4511/.1537 | .4388/.9066/.4381/.1531 | .4397/.9055/.4355/.1543 |
| 2 | .4573/.9087/.4637/.1466 | .4519/.9095/.4642/.1471 | **.4585**/.9100/.4709/.1467 | **.4569**/.9100/.4715/.1463 |
| 3 | **.4585**/.9088/.4696/.1363 | **.4541**/.9092/.4652/.1364 | .4565/.9090/.4698/.1361 | .4546/.9075/.4673/.1352 |
| 4 | .4554/.9044/.4665/.1229 | .4504/.9045/.4626/.1238 | .4500/.9038/.4609/.1226 | .4486/.9024/.4585/.1234 |

Bold marks each run's best epoch. The `history.json` mAP lists match the logs to 4 decimal places for all four runs.

### Best-epoch EMA vs SWA, with Δ vs 49000
- **Best-epoch EMA** = `model_best.pth` (from the `Done. Best` line).
- **SWA** = the average of the EMA weights of epochs [2, 3, 4] (`model_swa.pth`). SWA lines are at L254 for the v4 runs and L249 for 49000.

| Run | Best ep | Best mAP (EMA) | Δ best vs 49000 | SWA mAP | Δ SWA vs 49000 | SWA mAUC | SWA mF1 | SWA mECE |
|---|---|---|---|---|---|---|---|---|
| 49000 v3 | 3 | 0.4585 | – | 0.4600 | – | 0.9092 | 0.4685 | 0.1354 |
| mk1 | 3 | 0.4541 | −0.0044 | 0.4551 | −0.0049 | 0.9096 | 0.4661 | 0.1360 |
| mk2 | 2 | 0.4585 | −0.0000 (0.45848 vs 0.45850) | 0.4566 | −0.0033 | 0.9097 | 0.4719 | 0.1354 |
| mk1fix | 2 | 0.4569 | −0.0016 | 0.4569 | −0.0031 | 0.9088 | 0.4642 | 0.1353 |

Raw values:
- Best mAP: 0.4541016 (mk1, L247), 0.4584792 (mk2, L247), 0.4569111 (mk1fix, L247), 0.4584990 (49000, L242).
- SWA mAP: 0.45508 (mk1), 0.45664 (mk2), 0.45688 (mk1fix), 0.45997 (49000).
- SWA val_loss: 0.0395 (mk1), 0.0394 (mk2), 0.0397 (mk1fix), 0.0395 (49000).

Early stopping:
- mk2 and mk1fix both log "Early stopping triggered at epoch 4. Best mAP=… at epoch 2." (L246).
- mk1 and 49000 (best epoch 3 in both) ran all 4 epochs with no early-stop line.

The SWA pool was [2, 3, 4] in every run.

## Markov gate statistics
The script logs `markov: steps=K |gate| mean/max | |bias| mean | strongest gates` once per epoch, at about L107/153/199/245 (L244 in mk2 and mk1fix). At initialisation g = b = 0, so the layer is a no-op. The "strongest gates" field lists the top-5 classes by |g|.

| Run | Ep1 \|g\| mean/max, \|b\| mean | Ep2 | Ep3 | Ep4 |
|---|---|---|---|---|
| mk1 | 0.0055/0.0274, 0.0067 | 0.0090/0.0445, 0.0109 | 0.0124/0.0540, 0.0156 | 0.0129/0.0613, 0.0173 |
| mk2 | 0.0056/0.0235, 0.0061 | 0.0090/0.0456, 0.0111 | 0.0118/0.0522, 0.0154 | 0.0111/0.0466, 0.0169 |
| mk1fix | 0.0052/0.0289, 0.0056 | 0.0095/0.0462, 0.0107 | 0.0127/0.0580, 0.0145 | 0.0131/0.0613, 0.0164 |

- **Layer usage:** the gates stay tiny and grow slowly and monotonically. By epoch 4 the mean |g| is about 0.011–0.013 and the max about 0.05–0.06. The optimiser used the layer a little, not substantially.
- **Learnable vs frozen P:** the gate and bias statistics are essentially identical across mk1, mk2 and mk1fix. There is no evidence that a learnable P changes how much the layer is used.
- **Strongest gates:**
  - Mass is positive in every run and epoch (+0.027 to +0.061).
  - Pneumothorax, pneumoperitoneo and Emphysema (negative) recur across runs.
  - Pneumothorax's gate is larger in mk1fix (+0.037 to +0.044) than in mk1 (+0.015 to +0.020).
  - Normal appears among the strongest gates in mk2.
  - No interpretation is claimed for these.
- **P drift:** the script does not log it. It was estimated by comparing `label_refine.A` in each learnable run against mk1fix's frozen A, on the assumption that all three share the same initialisation (same seed and training labels).
  - Best-EMA checkpoints: mean off-diagonal |ΔA| is 0.039 (mk1) and 0.026 (mk2); max 0.49 and 0.32.
  - SWA checkpoints: mean 0.038 (mk1) and 0.036 (mk2); max 0.47 and 0.42.
  - A has a standard deviation of about 5.04, so P moved very little, consistent with the small gates.

## Comparison to baselines
- **vs 49000 (matched v3 baseline, same seed 42):** all three v4 arms are at or below it.
  - Best-epoch mAP Δ: −0.0044 (mk1), −0.0000 (mk2), −0.0016 (mk1fix).
  - SWA mAP Δ: −0.0049, −0.0033, −0.0031.
  - SWA mAUC and mECE are within ±0.001 of 49000. SWA mF1 ranges from −0.0043 to +0.0034, with mixed signs.
- **vs the Stage-1 backbone (internal mAP 0.385):** all arms are far above it, but that gain comes from the Stage-2 recipe, not from the Markov layer.
- **vs the MoE Stage-2 (internal 0.405 / test 0.4599) and the paper's 5-seed entropy-triplet result (0.414 ± 0.002, internal, 512 px):** not like-for-like, because these runs use 768 px and dp 0.1.
- **vs the campaign best (0.4602, SWA 0.4619, 1024 px, job 48997):** none of these runs gets close.

## `args:` diff vs 49000
The `args:` lines were compared token by token. Flag order differs; flag sets differ only in the Markov flags.
- **Identical in all four runs:** seed 42, accum-steps 12, batch-size 4, monitor mAP, memory-size 2048, metric-embed query, head-classes 5, triplet-lambda 0.1, class-weight-order shard, swa-last-k 3, lr 1e-4, epochs 4, patience 2, sched-epochs 8, drop-path-rate 0.1, img-size 768, stage1-ckpt.
- **Added in v4:** `--markov-lr-mult 10` (also the script default) and `--markov-steps {1,2,1}`; mk1fix adds `--markov-fixed-transition`.
- **Startup log (L42):** "Markov label-refine layer: steps=K learn_transition=True/False lr_mult=10.0 (identity at init)". mk1fix shows `learn_transition=False`.
- **When and where:** jobs 49063–49065 ran from 2026-09-26 20:00 to 2026-09-27 06:14 on petra. 49000 ran on 2026-09-25.

## Known-bug checks
- **Class-weight-order bug:** not present. `--class-weight-order shard` is set in every run, and `train_2_v4.py` raises an error if a Markov run uses a non-shard order.
- **Metric-embed wiring:** `--metric-embed query` is set in every run, so these runs are post-fix.
- **Paper-revision confounds:** none applies, because both arms run at 768 px in a matched comparison.
  - Triplet utilisation was not measured.
  - Softmax vs sigmoid entropy is unchanged between arms.
- No known bug invalidates the comparison.

## Noise context
All v4 arms are single-seed (42), and their only paired reference is 49000 (seed 42). Available spread from other runs:
- **512 px seedsweep** (6 seeds): best-epoch mAP 0.4434, 0.4389, 0.4414, 0.4375, 0.4369, 0.4385. Range 0.0065, SD ≈ 0.0031. These histories have no SWA metrics.
- **640 px res_sweep:** range 0.0047.
- **768 px dp 0.1, two seeds:** seed 42 gives best 0.4585 / SWA 0.4600; seed 1024 gives 0.4564 / 0.4582. The gap is 0.0021 (best) and 0.0018 (SWA).

What this means for the Markov runs:
- The best-epoch Δs (−0.0044, −0.0000, −0.0016) are within about 1.4 seed SD and inside the observed range of 0.0047–0.0065.
- The SWA Δs (−0.0049, −0.0033, −0.0031) are all negative and similar in size. That is a small, consistently signed shortfall, but one seed cannot establish it as real.
- **Safe statement:** no arm shows a detectable improvement over v3. The data do not distinguish "no effect" from "slightly harmful".
- The differences between the arms themselves are also within noise, so there is no basis for ranking K = 1 vs K = 2, or learnable vs frozen P.

## Not measured
- Flip-TTA
- Frontal-only subset
- Unseen-patient subset
- Per-class or head/medium/tail breakdown (no tail-class claim either way)
- Ensemble
- Triplet utilisation

No job was launched.

## Caveats
- Single seed and a single run per arm, so the Δs cannot be separated from seed noise.
- Checkpoint selection used internal-val mAP on a split that is not patient-disjoint. This biases every arm upward by the same mechanism.
- "Best" means the EMA weights at the best epoch; "SWA" means the mean of EMA epochs 2–4. Do not mix the two.
- The P-drift estimate relies on mk1fix's A being unchanged from its initialisation.

## Notable
- **Negative result:** all three Markov arms are at or below the matched v3 run on best-epoch EMA and SWA mAP, with SWA about 0.003–0.005 lower. The shortfall is within seed noise but consistently signed.
- **The layer barely contributed:** the gates stayed near zero (mean |g| ≈ 0.013). The lack of effect may reflect an under-weighted layer rather than a verdict on label-graph information (speculation, not measured). A larger `markov-lr-mult` or a longer run would test it.
- No new best, and no bug invalidates the comparison.

## Next step (suggestion only)
- Multi-seed repeat of 49000 and one v4 arm (e.g. seeds 86 and 1024).
- Per-class eval (`eval_perclass.sh`) of the v4 arms and 49000, to check the tail classes.

---

## Main-session notes (added 2026-09-30)
- **P initialisation is train-fold only (verified).** `cooccurrence_transition(labels_np)` (`train_2_v4.py:1436`) uses `labels_np` from `train/CXRLT_2026_training_filtered.csv` (`train_2_v4.py:1340–1354`). That file has 103,303 rows; 99.998% of `data/train_fold_1.csv` images are in it and 0 of the 17,274 `data/val_fold_1.csv` images are. No val labels leak into the prior.
- **No strict logit bound.** A draft of this report bounded the per-class logit correction at |g| ≤ 0.06 on the grounds that σP ≤ 1. That bound is not strict: m_j = Σ_i q_i P_ij is bounded by column j's sum, and column sums of a row-stochastic P can exceed 1. With about 1.3–1.6 positive labels per image, m_j is typically at most about 1, so the correction is still small. The paper states only the measured gate magnitudes.
- **Delivery.** This report was returned as text by the `cxrlt-research-analyst` subagent and saved here by the main session. The spot checks against the logs matched: the 49000 SWA line, and all 12 `markov: steps` gate lines of 49063–49065.
