# Stage-2 Training Log

Tracks completed Stage-2 (metric-learning fine-tune) training jobs under
`train/checkpoint_Triplet_3/`, their configuration, and achieved metrics.
Newest run first. Internal validation metrics only unless noted otherwise
(no held-out leaderboard submission has been scored for these runs yet).

Related context: [`docs/paper.tex`](paper.tex) (methods writeup),
`scripts/submit_stage2_v3_res_sweep.sh` (launcher for the run below).

---

## 2026-09-10 — Resolution sweep (img 640 vs 768), `train_2_v3.py`

**Directory:** `train/checkpoint_Triplet_3/res_sweep/`

**Motivation:** `docs/Entropy.pdf` (slide 9) reported that raising input
resolution 384→512 px tripled the Stage-2 mAP gain (+0.007 → +0.020). Every
prior `train_2_v3.py` run used 512 px. This sweep tests 640 and 768 px on the
corrected `fix_only` recipe (query metric-embed head, shard-order class
weights — see below) to see if the trend continues. Shards are native-resolution
PadChest originals, so no data rebuild was needed.

**Shared configuration** (`scripts/submit_stage2_v3_res_sweep.sh`):

| Setting | Value |
|---|---|
| Script | `train/train_2_v3.py` |
| Stage-1 checkpoint (frozen) | `train/checkpoint3/Model_20260119_062652/model_best.pth` |
| Effective batch size | 48 (batch 4 × accum 12) |
| Epochs / patience | 8 / 3 (early stopping on mAP) |
| LR | 1e-4 |
| Metric-embed head | `query` (correct wiring — see [[stage2-triplet-embedding-miswired]]) |
| XBM memory size | 2048 |
| Head-classes | 5 |
| Triplet lambda | 0.1 |
| Class-weight order | `shard` (bug fix — see [[stage2-classweight-order-bug]]) |
| SWA | last 3 EMA checkpoints by monitored metric |
| Cluster | `petra.cs.nott.ac.uk`, partitions `amp48,ada24`, GPU: RTX 4500 Ada (24GB) |

Per-run overrides: `--seed`, `--img-size`.

**Results** (best single-epoch checkpoint, selected by internal val mAP):

| Run | Img size | Seed | Best epoch | mAP | mAUC | mF1 | mECE | SWA mAP | SWA mF1 |
|---|---|---|---|---|---|---|---|---|---|
| `img768_seed86` | 768 | 86 | 2/8 (stopped ep. 5) | **0.4560** | 0.9104 | 0.4661 | 0.1436 | **0.4566** | 0.4676 |
| `img640_seed1024` | 640 | 1024 | 2/8 (stopped ep. 5) | 0.4494 | 0.9062 | 0.4606 | 0.1367 | 0.4501 | 0.4625 |
| `img640_seed86` | 640 | 86 | 3/8 (stopped ep. 6) | 0.4447 | 0.8992 | 0.4536 | 0.1226 | 0.4458 | 0.4483 |

**Job log / directory map:**

| Run | Slurm job | Log | Result dir |
|---|---|---|---|
| `img768_seed86` | 47095 | `/home/psytp7/logs/isbi2026_v3res_img768_s86_47095.out` | `train/checkpoint_Triplet_3/res_sweep/img768_seed86/Model_run/` |
| `img640_seed86` | 47094 | `/home/psytp7/logs/isbi2026_v3res_img640_s86_47094.out` | `train/checkpoint_Triplet_3/res_sweep/img640_seed86/Model_run/` |
| `img640_seed1024` | 47096 | `/home/psytp7/logs/isbi2026_v3res_img640_s1024_47096.out` | `train/checkpoint_Triplet_3/res_sweep/img640_seed1024/Model_run/` |

Each result dir contains `history.json` (per-epoch train/val loss, mAP, mAUC,
mF1, mECE, plus `best_epoch`/`best_metric`/`swa_metrics`), `model_best.pth`
(best single EMA checkpoint, ~526MB), and `model_swa.pth` (SWA-averaged EMA
weights, ~526MB).

**Wall time:** all three jobs were submitted at 14:42:49 BST (10 Sep 2026).
`img768_seed86` finished last (early stop 02:51, SWA written 03:14 on 11 Sep)
— roughly **12.5h** end-to-end at ~3.6–3.8 it/s during training.

**Outcome:** 768 px is the new best Stage-2 result to date, beating the prior
best of 0.4434 from the 512 px `seedsweep` batch
(`train/checkpoint_Triplet_3/seedsweep/`, 6 seeds, same `fix_only` recipe) and
clearing the [[stage2-map-improvement-campaign]] single-model gate (mAP >
0.42) by a wide margin. The 384→512→640→768 resolution trend from
`Entropy.pdf` holds through 768 px, with diminishing but still positive
returns (640: +0.001–0.006 over 512 best; 768: +0.0066 over 640 best; +0.0126
over the 512 seed-sweep best).

**Not yet done:** no per-class / held-out leaderboard evaluation has been run
on these three checkpoints. `submit_stage2_v3_res_sweep.sh` suggests:

```bash
for d in train/checkpoint_Triplet_3/res_sweep/*/Model_run; do
  sbatch -x colossus scripts/eval_perclass.sh "$d/model_best.pth" 512 "$(basename "$(dirname "$d")")"
done
# also worth trying: evaluate_tta.py --scales 512,640,768 across the res models
```

---

## Prior run — Seed sweep (512 px), `train_2_v3.py`

**Directory:** `train/checkpoint_Triplet_3/seedsweep/`

Same `fix_only` recipe as above, fixed at the default 512 px, across 6 seeds.
Best internal mAP **0.4434** (seed 1024, best epoch 2); other seeds: 42 →
0.4414, 123 → 0.4389, 86 → 0.4385, 456 → 0.4375, 789 → 0.4369. Superseded by
the resolution sweep above for single-model quality, but useful as the
512 px baseline the resolution sweep was measured against.
