# train_2_v0.py vs train_2_v5.py parity validation (2026-10-13)

One-off validation of the new ASL-only script before v5's ASL/MoE modes were removed. Not a result: none of these runs enter
a results table. Plan: `docs/plan/2026-10-13-asl-v0-vs-entropy-v5-plan.md`; frozen v5 state: `docs/plan/2026-10-13-v5-cleanup-manifest.md`.

## Checks
| check | tool | result |
|---|---|---|
| Source equality of 9 functions/classes, 7 `Trainer` methods, 4 constants, defaults of the 33 shared flags (v5 read from git tag `v5-pre-moe-removal`) | `scripts/check_v0_v5_parity.py --v5-ref v5-pre-moe-removal` | PARITY OK |
| First 32 train and 32 val samples bit-identical (seed 42, 768 px, epoch 1) | `scripts/check_v0_v5_data.py` | DATA PARITY OK |
| 40-batch training + 20-batch eval, same seed/args: v0 (job 52393) vs v5 `--no-triplet` (job 52392) | Slurm smoke, `--shuffle-buf 1 --accum-steps 2 --img-size 768 --drop-path-rate 0.1 --swa-last-k 1` | identical to every printed digit |

Smoke numbers (both scripts): train_loss 0.053920282237231734, val_loss 0.04406631502899386, mAP 0.29699060320854187,
mAUC 0.5847045183181763, mF1 0.2192385196685791, mECE 0.14635971188545227, final lr 3.8722168441432724e-07, SWA metrics equal to
epoch metrics, 131,527,070 trainable parameters, Stage-1 load missing=0 unexpected=0, class weights equal (shard order).

## Finding: the sample-buffer shuffle is not seeded
webdataset 1.0.2 `.shuffle(n)` seeds its RNG from `os.getpid() + time.time()` when no `seed=` is passed. v5 (and v2/v3/v4) never
pass one, so with a real buffer the minibatch composition differs from run to run even with `--seed` fixed; the comments saying
the buffer is "reproducible via seed_worker" are wrong. Seeded in v0 and v5 alike: shard order (`random.Random(seed)`),
per-sample augmentation (`stable_seed_from_key`, keyed on sample and epoch), weight init and torch RNG. Consequence for the study:
two runs of the same seed differ by this extra nondeterminism (job 49375 vs its rerun: val best 0.4793 vs 0.4740), v0 and v5 share
it, and the paired-by-seed statistics already include it. Neither script was changed (that would break parity with every finished
v5 run). The smoke comparison above used `--shuffle-buf 1` (pass-through) so the data order was identical.
