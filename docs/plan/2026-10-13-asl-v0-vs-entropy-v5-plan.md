# Plan: pure ConvNeXt + ASL `train_2_v0.py` vs Entropy / Entropy+Markov `train_2_v5.py`, all on the job-49375 recipe

## Context
Three systems only: **ASL only**, **Triplet Memory + Entropy**, **Triplet Memory + Entropy + Markov prior** (history available /
withheld). MoE, "every image an anchor", "ASL + Markov prior" and the strong-triplet arm are dropped.
- ASL trains **only** on `train/train_2_v0.py`; Entropy and Entropy+Markov train **only** on `train/train_2_v5.py`.
- Reference config = job 49375 (`v5prior_qprior_s42`, mAP 0.4735 / SWA 0.4807). From its log: seed 42, aug_base_seed 42,
  768 px, shuffle_buf 400, rrc_scale_min 0.9, bs 4 x accum 12 (eff. 48), 4 of 8 cosine epochs, patience 2, warmup 0.05,
  gamma_neg 2.0 / gamma_pos 0, clip 0.05, EMA 0.999, SWA of top-3 EMA epochs, drop-path 0.1, lr 1e-4, wd 0.01, workers 8/2,
  strict_repro, Stage-1 `checkpoint3/Model_20260119_062652/model_best.pth`, class weights in shard order.
- Table rows: ASL only (v0) / Entropy / Entropy+Markov (history available) / Entropy+Markov (history withheld);
  columns n, mAP, mAUC, mF1, mECE, head, medium, tail AP. c and d already exist (5 seeds each); only ASL trains.

## Step 0 - freeze the old state (answers risk 2)
Before editing v5: record `git rev-parse HEAD` (v5 is committed and clean), `sha256sum train/train_2_v5.py train/convnext_moe.py
scripts/submit_stage2_moe.sh scripts/submit_stage2_v5_unified.sh`, and the list of run dirs made with the removed flags
(`unified768/{a,b,e,f}`, `checkpoint_MoE/*`) into `docs/plan/<date>-v5-cleanup-manifest.md`; add a local, non-pushed
git tag `v5-pre-moe-removal` at HEAD. Old results then stay reproducible with `git checkout v5-pre-moe-removal -- train/train_2_v5.py`;
no copy of the old script is left in the active tree. (No commit is made without being asked.)

## Step 1 - write `train_2_v0.py` as a clean, minimal script (replaces the stripped-v2 file)
Rewrite, not a patch: ConvNeXt2 + ASL + the shared data pipeline, nothing else. Remove every v2 leftover:
`--disc` and the "Factors" code, `AUG_LOG`/`ENABLE_AUG_LOG`, albumentations/`Image`/`F`/`TF`/`normalize`/`rgb_to_grayscale`/
`CosineAnnealingLR`/`model_urls`/`Path`/`io` imports, commented-out code (`# set_params`, `# _normalize`, `#trainer = ...`),
the `checkpoint_Triplet_2` default out_dir, deprecated `torch.cuda.amp.autocast/GradScaler` (use `torch.amp.autocast("cuda")` /
`GradScaler("cuda")` exactly as v5 already does), the moe-aux hook, the `(logits, embeddings)` unpack (take `out[0]`), the triplet/
memory args, the v2 ASL defaults (gamma 4.0/1.0 -> 2.0/0.0), the 1000 shuffle buffer. Flags kept (all with v5's defaults):
`--seed --aug-base-seed --out-dir --stage1-ckpt --num-classes --lr --weight-decay --epochs --patience --min-delta --warmup-frac
--sched-epochs --train-size --gamma-neg --gamma-pos --monitor --batch-size --accum-steps --no-amp --num-workers --val-num-workers
--log-every --class-weight-order --img-size --drop-path-rate --shuffle-buf --rrc-scale-min --ema-decay --swa-last-k --resume
--limit-train-batches --limit-val-batches --train-shards-dir --val-shards-dir --no-strict-repro`.
Pieces copied verbatim from v5 (no import): `stable_seed_from_key`, `deterministic_random_resized_crop`,
`deterministic_brightness_contrast`, `deterministic_aug`, `deterministic_val_resize`, `decode_and_transform` /
`DecodeAndTransform` (minus the prior lookup), `make_webdataset` (minus prior), `seed_everything`, `seed_worker`, `AsymmetricLoss`
(already identical to v5), `EarlyStopping`, `preprocess_batch` (already identical), the LR schedule (per-step warmup+cosine over
`sched_epochs`), single-group `AdamW(model.parameters())` (what v5's `_build_optimizer` returns by default), EMA, top-k EMA SWA,
`_evaluate_impl` metrics. Model: `ConvNeXt2(depths=[3,3,27,3,2], dims=[128,256,512,1024,1024], drop_path_rate=...)`, Stage-1 load
`strict=False` as v5, but log and **assert `missing_keys == []`** (no new modules exist, so any missing key is a bug).
No entropy, triplet, memory bank, MoE, Markov or prior code anywhere; `grep` check below enforces it.

## Step 2 - risk 1 workaround: make "tiny differences" measurable and zero
Parity tests, none of which run a v5 training for the reported results:
a. **Source equality** (`ast`, scratchpad script): the functions/classes listed as "verbatim" above have identical normalized
   source in v0 and v5; argparse defaults identical for every shared flag (today's diff shows only v0's old accum/epochs/gamma_neg/
   patience/warmup differ, which Step 1 resets).
b. **Data-pipeline equality** (CPU, num_workers 0): from `make_webdataset` of each script with seed 42, img 768, shuffle 400, the first
   ~256 sample keys and a hash of every transformed tensor are identical.
c. **One-off numerical parity, before v5 is edited**: run the *current* v5 with `--no-triplet` and v0 with the same seed/args for
   `--limit-train-batches 40` and compare the per-step loss and the first EMA/eval numbers (fp tolerance). It is a validation of v0
   only; its output is saved in docs and is never put in a results table.
d. **End-to-end sanity**: v0 seed 42 val best/SWA mAP lands in the range of the existing v5 ASL run `checkpoint_MoE/asl_img768_seed42`
   (0.4567 best) and its same-seed noise (below), otherwise investigate before launching the other seeds.
Residual difference after a-d, stated in the write-up: ConvNeXt2 still computes (and discards) its legacy embedding output because
`convnext.py` is shared with v5/eval and is not edited; it has no parameters and no effect on loss.

## Step 3 - clean `train_2_v5.py`
Delete `--arch --moe-n-experts --moe-aux-weight --expert-noise --no-triplet`, their cfg keys, the `convnext_moe` import, the MoE
branches of `create_model`, `self.is_moe/moe_aux_weight/no_triplet` and every `if self.no_triplet ... else`, the `moe_aux` hooks and
history keys (l.45, 468-512, 653-695, 721, 925-1050, 1348-1360, 1473-1489), and the `checkpoint_MoE` out_dir branch. Entropy path
(`--triplet-lambda`, `--anchor-rule`, `--prior-file`, ...) stays unchanged. `submit_stage2_v5_unified.sh` is cut to arms c and d;
`scripts/submit_stage2_moe.sh` deleted. `evaluate/evaluate_tta_v5.py` and `train/convnext_moe.py` untouched (read old checkpoints only).
Check: `grep -n "moe\|no_triplet\|no-triplet\|expert" train/train_2_v5.py` empty; `SMOKE=1 submit_stage2_v5_unified.sh c:42 d:42` starts.

## Step 4 - risk 3 workaround: cheap, resumable ASL runs
- v0 writes `last.pth` every epoch (model, EMA, optimizer, GradScaler, scheduler, EarlyStopping state, SWA pool, history,
  `g_torch` state, epoch) and `--resume` continues from it, so a timeout, colossus failure or manual cancel costs at most one epoch
  (the 2026-10-13 cancel destroyed several 11 h jobs). Augmentation is keyed on (sample, epoch) and shard order on the seed, so
  a resumed epoch sees the same data.
- `scripts/submit_stage2_v0.sh`: 49375 flags above, MEM 96G, TIME 36h, partitions amp48,ada24, colossus excluded, out_dir
  `checkpoint_Triplet_3/v0_768/seed_S/Model_run`, flip-TTA dump of `model_swa.pth@768` to `analysis/out/probs_v0_flip`
  (keys.txt/labels.npy copied from `probs_unified_flip`), eval job `afterok`. Smoke first.
- Wave order 42 -> 123, 1024 -> 456, 789: an early 3-seed paired read arrives after the first wave; all five are still run (n=5 fixed
  in advance). Check `squeue`/fairshare before each wave; no new jobs while others of the user are queued.

## Step 5 - risk 4 workaround: report noise honestly
- Report mean +/- sd over n=5 and paired-by-seed differences with 95% t-CI, exact sign-flip p and wins (existing helpers in
  `analysis/entropy_ablation.py`), plus the paired-t MDE already in `unified_benchmark.py`.
- Same-seed repeat variance as noise floor, already on disk: `prior/img768_qprior_seed42` vs `_rerun` (val best 0.4793 vs 0.4740)
  and seed1024 vs `_rerun` (0.4760 vs 0.4761). State it next to every contrast; do not quote 0.4807 as an expected value.
- The Entropy+Markov s42 row point is the original 49375 run; the rerun is mentioned, not substituted.
- Enable `torch.backends.cudnn.benchmark = False` explicitly in v0 (v5 sets only `deterministic = True`); no
  `use_deterministic_algorithms` (it would slow v0 relative to v5 and change kernels); AMP/atomic-add nondeterminism is shared by both
  and is exactly what the same-seed repeat measures.

## Step 6 - table and reports
`analysis/unified_benchmark.py`: `ROWS = A (v0, reads probs_v0_flip), C, Dwith, Dzero`, `MAIN_ROWS = (A, C, Dwith, Dzero)`,
contrasts C-A, Dwith-A, Dzero-A, Dwith-C, Dzero-C; remove the B/E/F/M rows, controls table and macros; update docstring.
`python -I analysis/unified_benchmark.py` -> `unified_table.tex`. Write-up through the `cxrlt-research-analyst` agent: metrics to
`docs/result/<date>-asl-entropy-markov-768.md`, findings to `docs/plan/<date>-asl-entropy-markov-768-findings.md`, naming the script
behind each row and the residual-difference statement from Step 2.

## Verification
1. `grep -n -i "triplet\|memory_bank\|memory\b\|anchor\|entropy\|moe\|prior\|markov\|disc\|embedding" train/train_2_v0.py` has no code hits;
   `py_compile` passes for v0 and v5.
2. Step 2a-c pass; the 40-batch v0 smoke log shows img_size 768, shuffle_buf 400, aug_base_seed 42, EMA/SWA on, Stage-1 missing=0.
3. Kill and `--resume` a smoke job after epoch 1: epoch-2 metrics equal an uninterrupted run (within fp tolerance).
4. Finished v0 history has `swa_metrics`; `unified_benchmark.py` prints n=5 for all four rows and writes the table.
