# Plan: rebuild Mixture of Experts and run it under the Triplet Memory setting (640–1024 px)

> Copy of the plan-mode plan approved on 2026-10-08 (docs policy: approved plans are kept in `docs/plan/`).
> Status at copy time: the 9 training runs (jobs 51151-51159) and the flip-TTA eval chain (51178-51189,
> `scripts/submit_eval_moe.sh`) were submitted. On 2026-10-08 the 7 still-pending runs (51153-51159) were put on
> hold so that the entropy ablation (`2026-10-08-entropy-ablation-isbi-submit-plan.md`) gets the GPUs first;
> they are released once every ablation run has started.

## Context
Table III of `docs/paper/isbi/paper.tex` compares Triplet Memory (entropy-gated triplet, code v3) and Triplet Memory + Markov Prior (v5) at 640–1024 px. The only MoE numbers are the published ISBI 2026 values (b12, `docs/pdf/ISBI2026.pdf`; `Entropy.pdf` p.9 shows 0.392 @384 / 0.405 @512). Those are at lower resolution, from a different recipe, with no checkpoint, seeds or logs in the repo. A 384 px comparison proves nothing.

To claim that Entropy beats MoE, and to measure what the Markov Prior adds (as the basis for the CheXGCN comparison), MoE has to be re-run with **everything identical to Triplet Memory except the module**:
- same Stage-1 checkpoint, resolutions, seeds, batch/accum, LR schedule, epochs, EMA, SWA, drop-path, class weights, data order and flip-TTA evaluation.

**User decisions:**
- No Triplet Memory involvement in MoE: no triplet loss, no memory bank, no metric embedding.
- Full fine-tune, the same trainable parameters as Triplet Memory.
- Cancel pending job 51100 (384 px MoE smoke on the old `train_moe.py` recipe) and keep 51099 (512 px no-triplet control).
- Add a plain ASL-only control at 768 px.

**Exploration facts:**
- `train_moe.py` is an untracked reconstruction from today.
- The MoE plumbing lives only in root `convnext3.py` + `moe.py`. That stack returns logits only and uses root `ml_decoder.py`, which has no `last_h` and no `query_offset`.
- `train/train_2_v3.py` and `v5` always build ConvNeXt2. Their aux hook (`0.005 * model.last_aux_loss`, v3 L861) is inert.
- Both evaluators load with `strict=False`, so they would silently drop the expert weights.

## Arms (all on the internal val fold, 17,270 images)
| Arm | Loss | Module |
|---|---|---|
| A. ASL-only control | ASL (shard-order class weights) | none |
| B. MoE | ASL + 5e-3 × load-balance aux | MoE in stages 3–4 MLPs + decoder cross-attention FFN, 4 experts, top-2 (b12 design, `moe.py`) |
| C. MoE + Markov Prior | B + patient prior into label queries | B + `PriorQueryConditioner` (v5) |
| existing | Triplet Memory / Triplet Memory + Markov | already trained |

## Implementation
1. **`train/moe.py`**: copy of root `moe.py`, unchanged, so `train/` stays self-contained.

2. **`train/convnext_moe.py`**: `ConvNeXt2MoE(ConvNeXt2)`, built from `train/convnext.py`'s ConvNeXt2 so it keeps `forward -> (logits, emb)` and the train/ `ml_decoder.Decoder` (`last_h`, `query_offset`).
   - Port `BlockMoE` from root `convnext3.py:13-55` for stage 3 and 4 blocks. Expert 0 is the original `pwconv1/pwconv2`, so Stage-1 keys load unchanged.
   - Port `DecoderLayer_CASA_MoE` (`convnext3.py:58-160`) onto `train/ml_decoder.DecoderLayer_CASA`. Only the FFN after cross-attention is replaced.
   - Keep `drop_path` working in `BlockMoE`.
   - `forward` aggregates aux exactly as `convnext3.py:328-349` into `self.last_aux_loss`.
   - Expert init: copies of expert 0 + N(0, (0.01·std)²) noise, seeded. Reuse the init function from `train_moe.py:47-71`.
   - Add `ConvNeXt2MoEPrior` for arm C (MoE model + `prior_cond`, reusing `train/prior_conditioning.py`).

3. **`train/train_2_v5.py`** (one trainer for A, B and C; v5 without `--prior-file`/`--markov-steps` is the v3 recipe). New flags:
   - `--arch {convnext2,moe}`, `--moe-n-experts 4`, `--moe-aux-weight 5e-3`, `--expert-noise 0.01`.
   - `--no-triplet`: do not build `CrossBatchMemoryV2`, skip mining and the `last_h` metric-embed path, loss = ASL (+ aux).
   - `--arch moe` requires `--no-triplet`. This enforces "no Triplet Memory involvement".
   - `create_model` picks ConvNeXt2 / ConvNeXt2MoE / ConvNeXt2MoEPrior, and asserts that the only missing Stage-1 keys are `extra_experts`/`router`/`prior_cond`.
   - Replace the hard-coded `0.005` aux weight with the flag.
   - Log the aux loss and per-expert load per epoch, to check for collapse.
   - SWA/EMA must average the expert and router weights too (they are ordinary parameters, so check that the SWA state-dict keys include them).

4. **`evaluate/evaluate_tta_v5.py`**: detect MoE checkpoints by `extra_experts`/`.router.` keys and build `ConvNeXt2MoE`/`ConvNeXt2MoEPrior`. For any MoE checkpoint, load with `strict=True` and fail loudly on missing keys. Prior with/zero modes are unchanged.

5. **`scripts/submit_stage2_moe.sh`**: modelled on `scripts/submit_stage2_v3_push.sh` and `submit_stage2_v5_prior.sh`. It uses the same SBATCH block (`-p amp48,ada24 -x colossus -c 8 --mem=96G --time=36:00:00`), has a `SMOKE=1` mode, and writes to `train/checkpoint_MoE/<cfg>/Model_run`.

## Runs
Every run gets the exact v3 args of its Triplet Memory twin (job 49000 `args:` line): `--accum-steps 12 --batch-size 4 --lr 1e-4 --epochs 4 --sched-epochs 8 --patience 2 --drop-path-rate 0.1 --swa-last-k 3 --class-weight-order shard --stage1-ckpt checkpoint3/Model_20260119_062652/model_best.pth`.

| Twin of | Arm B MoE | Arm A ASL | Arm C MoE+Markov |
|---|---|---|---|
| push 768 s42 / s1024 | ✓ / ✓ | ✓ / ✓ | ✓ / ✓ (`--prior-file analysis/out/patient_markov/priors.pt --prior-dropout 0.3 --prior-lr-mult 10`) |
| push 896 s86, 1024 s86 | ✓, ✓ | – | – |
| res_sweep 640 s86 (`--epochs 8 --patience 3`, no dp, no sched-epochs) | ✓ | – | – |

- That is 9 training runs of about 10–16 h each, in about 2 waves given the 4-GPU-per-partition and CPU QOS caps (about 1.5–2 days).
- First, `scancel 51100`, then a smoke test (`SMOKE=1`, 1024 px, limited batches) of arms B and C to check memory, the strict loads and the aux values.

## Evaluation and analysis
- **Flip-TTA dumps** for every SWA checkpoint go to `analysis/out/probs_moe_flip/` via `scripts/evaluate_tta_v5.sh --dump-probs`. Arm C runs in both `--prior-mode with` and `zero`.
- **`analysis/moe_compare.py`**: computes mAP/mAUC/mF1/mECE with the `evaluate.py` definitions. The rows mirror Table III with fixed members and no re-selection:
  - singles (SWA);
  - fixed 4-model average (768 s42, 768 s1024, 896 s86, 1024 s86);
  - the greedy-member trio (768 s42 + 1024 s86 + 640 s86, the same members as Triplet Memory's 0.4722);
  - 2-seed MoE + Markov with/zero, against 2-seed MoE and against Triplet Memory + Markov orig2/rerun2.
- **Paired image-level bootstrap CIs** (2,000 reps) on Triplet Memory − MoE, Triplet Memory − ASL, MoE − ASL, and Markov increments on each backbone. Reuse the analyst's `boot.py` (scratchpad `work/`), archived to `analysis/`.
- **Per-class head/medium/tail split**: `evaluate/class_groups.py`.

## Docs and paper
- Copy the analyst's MoE-anchored comparison (in `/home/psytp7/.claude/plans/purring-wandering-quiche-agent-a30f6248a77ceb836.md`) to `docs/plan/2026-10-08-moe-anchored-entropy-markov-vs-gcn.md`.
- **Results**: `docs/result/<date>-moe-rebuild-matched.md`, written by the cxrlt-research-analyst. **Findings**: `docs/plan/<date>-entropy-vs-moe-vs-gcn-findings.md`.
- **Fix two mislabeled Table III rows now.** They currently claim MoE or Markov but contain neither:
  - "Mixture of Experts 768 px seed-42 SWA, flip" is a Triplet Memory model; rename it to "Triplet Memory 768 px seed-42 SWA, flip".
  - "Triplet Memory + Markov Prior, flip, 0.4722" is the greedy 3-model Triplet Memory ensemble with no Markov.
  - The real Markov rows (0.4847 / 0.4667) need to be restored.
- **After results**: add "Mixture of Experts (rebuilt, same recipe)" and "ASL-only" rows at matched px, plus MoE + Markov rows, and update the GCN discussion with the Markov increment on both backbones.

## Verification
- Unit check before submitting: build `ConvNeXt2MoE`, load the Stage-1 checkpoint, and confirm that only expert/router keys are missing.
- **Equivalence check**: with the routers forced to expert 0 only, the output equals ConvNeXt2's on a random batch.
- With `--arch moe`, no `CrossBatchMemoryV2` is built; grep the log for `triplet` and `memory_size`.
- Smoke job: the aux loss is finite and close to 2 at near-uniform routing; per-expert load is non-degenerate; peak GPU memory is logged at 1024 px.
- Strict-load eval of a smoke checkpoint through `evaluate_tta_v5.py`.
- Reproduce an existing Triplet Memory dump with the modified evaluator to confirm no regression: `push/img768_dp01_seed42@768` should give max |Δ| = 0 vs `probs_w01_flip`.
- **Final**: the `moe_compare.py` table numbers cross-check against each run's `history.json` `swa_metrics`.
