# Why ASL (`train_2_v0.py`) matches Triplet Memory + Entropy (`train_2_v5.py`) at 768 px

2026-10-14. Finding, read-only audit; no training launched. Related: `docs/plan/2026-10-14-asl-entropy-markov-768-findings.md`,
`docs/plan/2026-10-13-v0-v5-parity-validation.md`. Run metrics live in the unified-benchmark table.

## Verdict
No pipeline fault found. The two systems are genuinely close because the triplet term is a small additive loss
(`loss = ASL + 0.1 * triplet`), and the seed-to-seed noise is larger than any effect it has.

## Evidence
| Suspicion | Check | Result |
|---|---|---|
| v0 runs v5 code | `scripts/train_2_v0.sh` -> `python train_2_v0.py`; v0 imports nothing from v5; `scripts/check_v0_v5_parity.py` | `PARITY OK` (9 functions/classes, 7 Trainer methods, 33 shared flags) |
| v0 loads a v5/Stage-2 checkpoint | job 52402 log: `Stage-1 init: .../checkpoint3/Model_20260119_062652/model_best.pth (missing=0 unexpected=0)`, `'resume': False`; same Stage-1 line in c job 51832 | Both start from the same Stage-1 file only; separate out-dirs |
| Row A dumps come from another checkpoint | `unified_benchmark.json` runs.A files = `probs_v0_flip/v0_768__seed_*__model_swa@768_flip.npy`, all 5 seeds | OK. `eval_a_s42.json` (0.4600) points at `checkpoint_MoE/asl_img768_seed42`, a different run, and is not used by the table |
| v5 entropy / memory off | v5 has no `--entropy`; gate = `--anchor-rule mean`, memory always built. Job 51832: `anchor_rule=mean memory_size=2048 triplet_lambda=0.1 metric_embed=query`; `history.json` triplet_stats: 56-60k gated anchors and 31-33k mined triplets per epoch, hit rate 80-82% of 25,822 micro-steps | Both active |
| GPU parallelism | no DataParallel/DDP/`device_count` in either script; every job `-G 1` | Nothing to disable |
| Different effective batch | both `--batch-size 4 --accum-steps 12` (48), 768 px, same lr/schedule/SWA | Identical recipe |

## Why they are close
- Triplet term size: seed-42 train loss per epoch is v0 0.0406/0.0380/0.0353/0.0320 vs c 0.0408/0.0383/0.0357/0.0323. The extra
  term adds only about 2-4e-4 to a loss of about 0.04 (about 1%). Its gradient flows through the decoded query embeddings, which
  the ASL loss already shapes.
- Noise floor: webdataset `.shuffle()` is unseeded, so same-seed reruns differ by about 0.005 mAP.
- Paired C - A mAP over the 5 seeds (42, 123, 456, 789, 1024 order: 1024 +0.0019, 123 +0.0039, 42 -0.0051, 456 -0.0012, 789 -0.0021):
  mean -0.0005, s.d. 0.0035, exact sign-flip p = 0.75. The sign is not consistent across seeds.
- Per-epoch mAP, seed 42: v0 0.4383/0.4571/0.4597/0.4508, c 0.4407/0.4523/0.4554/0.4517. Both peak at epoch 3.

## The "0.43" screenshot
`docs/result/ASL_v0.jpeg` is a monitor snapshot taken at epoch 2 of 4 (2026-10-13 23:55); `BEST mAP` is the value so far, not
the final. The final v0 values are EMA best 0.4597 (seed 42) and SWA 0.4609. Epoch-1 values already match c (0.43-0.44).

## "Buffed 5x to match 712 px"
This is gradient accumulation (`--accum-steps 12`, batch 4), not a model change. 768 px runs fine on v0 (`Size of Image: 768`);
the 512 px limit applies only to batch size, not to the code.

## Housekeeping
`scripts/train_2_v0.sh` header comment still says "train_2_v2.py with Triplet Memory removed ... 512 px, no EMA/SWA"; stale, the
current v0 uses 768 px with EMA/SWA. Not changed in this pass.

## Optional next step (needs GPU time, not run)
To show the memory path can move the model at all: one v5 run, seed 42, 768 px, with `--triplet-lambda 1.0` or `--anchor-rule all`
(about 14 h). It would show sensitivity, not a better system.
