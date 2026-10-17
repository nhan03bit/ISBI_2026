# Task-1 submission data: what the shared zip contains and how we predict on it (2026-10-17)

The user asked to "use the submission dataset provided by ISBI to submit our best checkpoint" (shared Drive folder
`submission`, owned by a teammate). This note records what that data is, the checks run on it, and the decisions
still open. Which checkpoint to submit comes from `docs/result/2026-10-17-entropy-best-config.md`.

## 1. What the Drive folder holds
- **`Task-1.zip`** (31.5 GB, uploaded 2026-10-15): 3,400 PNGs in `Task-1/`, file dates 2026-01-20, 32-hex image IDs
  (de-identified). **No template CSV and no labels.**
  - Downloaded and extracted by job 53218 (`scripts/download_task1_gdrive.sh`): rclone checksum matched Drive, no
    absolute or `..` member paths. Location: `/data/psytp7/isbi_task1/extracted/Task-1/`.
- **`submit.py`** (7 KB, dated 2026-02-04): the teammate's predictor, not the organisers'. It reads a `labels.csv`
  template (ImageID without `.png` + 30 class columns), runs a 512 px MoE `ConvNeXt3`, and writes
  `predict_test5.csv`. It makes no network calls. Copy: `/data/psytp7/isbi_task1/starter_dl/submit.py`.
- No `labels.csv` exists anywhere in the shared Drive (searched by name).

## 2. What the challenge paper says (arXiv 2604.15555)
- Task 1 evaluation: 1,620 PadChest-GR development + 1,400 test images = 3,020 **frontal** images (Table 1, §2.2).
- §2.7: "Each image in the evaluation set was paired with a single perturbed counterpart generated once."
  "The official challenge ranking was based only on the original test set."
- §2.6: final evaluation phase 20 Jan - 4 Feb 2026, up to 5 valid submissions, on Codabench
  (competition 11470). The paper does not say whether submissions are still accepted.
- The submission format (columns, ImageID form, packaging) is not in the paper, and the challenge site returned
  404. The Codabench API returned an anti-scraping page.

**Reading:** the zip's file date (20 Jan 2026) is the final phase's start, and `submit.py` is dated its last day,
so this is most likely the final-phase image set **including the perturbed counterparts**. 3,400 matches neither
3,020 nor 6,040, so its exact composition is unconfirmed; ask the teammate who uploaded it.

## 3. Checks on the images
- **No overlap with our data:** 0 of the 3,400 IDs are in our train fold, val fold or the official training CSV
  (IDs only; the PadChest CSV was not read).
- **Format:** all 8-bit RGB (PNG colour type 2). Our training and val PNGs are 16-bit grayscale. After our decode
  (`IMREAD_COLOR`), the intensity percentiles match: median 123.5 vs 128, p90 204.5 vs 203, p99 227 vs 225.5
  (60 images each). The conversion is close to our own 16 -> 8-bit decode.
- **Noise** (`/data/psytp7/isbi_task1/tools/noise_probe.py`, all 3,400 images, wavelet MAD estimator):
  - **25.1% (about 850 images) carry independent white noise in each channel, σ ≈ 30 grey levels** at full
    resolution (p10 23, p90 39); it halves at half resolution, as white noise does.
  - The other 75% are clean: σ 1.48, the same as internal val (1.48).
- **Other differences seen in thumbnails:** some images are geometrically warped (skewed quadrilateral on a black
  background), and at least one of 12 sampled thumbnails is a lateral view, although the paper says Task 1 is
  frontal-only. Both are consistent with perturbed counterparts plus an image set we cannot fully identify.

## 4. Predictor
- `evaluate/predict_task1.py` + `scripts/predict_task1.sh`: the internal eval's decode and resize
  (`evaluate_tta.decode_and_resize` path: `IMREAD_COLOR`, BGR -> RGB, square resize with `--interp`, /255, ImageNet
  normalisation), models via `evaluate_tta_v5.load_model`, flip TTA, ensemble = mean of probabilities. Columns are
  mapped to the template by class name. Never reads the PadChest CSV. Writes the CSV, per-model probabilities
  (`<out>.per_model/`) and a manifest with checksums.
- **Parity (job 53224):** on 48 val images extracted from `shard_val_0000.tar` with a shuffled-column template, the
  output matches the shard-based area eval of the same checkpoint to max |Δp| 5.8e-5. Column mapping by name is
  correct.
- **Stand-in template:** `/data/psytp7/isbi_task1/template_from_training_csv.csv` = ImageID (hex, no `.png`, as
  `submit.py` assumes) + the 30 classes in the order of the official training CSV header. **Check it against the
  official sample submission before uploading.**
- **Candidate run (job 53230):** 11 Entropy / v3 entropy-gated checkpoints, area, flip TTA, written to
  `/data/psytp7/isbi_task1/pred/candidates_area.csv` + per-model dumps, so any ensemble can be assembled without a
  rerun.

## 5. Channel-mean decision (rule fixed before the simulation result)
`--channel-mean` averages the 3 decoded channels before resizing. On an image whose channels are identical (75% of
Task-1, and every original if the originals are unperturbed) it reproduces the input exactly, so it can only change
predictions for the noisy perturbed images, never the originals the official ranking used.

- **Simulation** (job 53234, `analysis/task1_shift_probe.py`): 6,000 internal-val images, Entropy c seed 42, area
  768, flip TTA; clean vs + per-channel N(0, 30) vs that + channel mean vs + shared-channel N(0, 30).
- **Rule:** use `--channel-mean` for the submission if `rgbnoise_mean - rgbnoise` > 0 with its 95% image-bootstrap
  interval above 0. Otherwise keep the plain decode.
- **Result (job 53234, 6,000 val images, 4,156 frontal; `analysis/out/task1_shift_probe_sigma30.json`):**

  | Condition | mAP | Frontal mAP |
  |---|---|---|
  | clean | 0.4956 | 0.5067 |
  | + per-channel N(0, 30) | 0.4799 | 0.4951 |
  | + per-channel N(0, 30), then channel mean | 0.4670 | 0.4784 |
  | + shared-channel N(0, 30) | 0.4277 | 0.4392 |

  `rgbnoise_mean - rgbnoise` = -0.0130 [-0.0204, -0.0072] (frontal -0.0167): the interval is below 0, so by the rule
  **the submission keeps the plain decode** (no `--channel-mean`). The existing CSVs (job 53230) already use it.
  Reading: independent per-channel noise costs -0.0157; the network's 3-channel stem already averages it out better
  than an explicit mean, which turns it into shared-channel noise (the same structure as `graynoise`, the worst row,
  at 1/sqrt(3) the amplitude). The perturbed quarter of Task-1 is expected to score about 0.016 below the clean images;
  the official ranking used originals only.

**Result (job 53234, 6,000 val images, 4,156 frontal; `analysis/out/task1_shift_probe_sigma30.json`):**

| Condition | mAP | Frontal | vs clean, 95% CI |
|---|---|---|---|
| clean | 0.4956 | 0.5067 | |
| per-channel noise σ 30 (`rgbnoise`, the Task-1 kind) | 0.4799 | 0.4951 | −0.0157 [−0.0237, −0.0067] |
| that + channel mean (`rgbnoise_mean`) | 0.4670 | 0.4784 | −0.0287 [−0.0392, −0.0192] |
| same σ 30 noise in all channels (`graynoise`) | 0.4277 | 0.4392 | −0.0679 [−0.0823, −0.0540] |

- **`rgbnoise_mean − rgbnoise` = −0.0130 [−0.0204, −0.0072]: the channel mean hurts.** By the rule, `--channel-mean`
  is **rejected** and the submission uses the plain decode.
- The model tolerates noise that differs between channels much better than noise shared by all channels (−0.016 vs
  −0.068 at the same σ). The channel mean turns independent noise into shared noise (σ 30/√3), which costs more than
  the √3 reduction saves.
- Checked on Task-1 itself (job 53236 vs 53230): the channel mean leaves the 2,548 clean images unchanged (max |Δp|
  ≤ 1.2e-4) and moves the 852 noisy ones by mean |Δp| ≈ 0.02. The `_chmean` CSVs in `pred_main/` are not for upload.
- Expected cost of the perturbation on the noisy quarter: about −0.016 mAP for one model in simulation. It does not
  touch the original images the official ranking used.

## 6. Open items for the user
1. Confirm the zip's composition (3,400 vs the paper's 3,020 originals + counterparts) with the teammate.
2. Get the official sample submission (column order, ImageID with or without `.png`, file name, zip) from
   Codabench competition 11470, and whether it still accepts submissions.
3. Uploading is the user's action; nothing is submitted from here.
