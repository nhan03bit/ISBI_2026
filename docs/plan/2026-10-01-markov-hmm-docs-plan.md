# Plan: document the post-prediction patient HMM in the architecture note and the paper

*Approved plan (2026-10-01), copied from plan mode. Follows [`2026-10-01-markov-hmm-redesign-plan.md`](2026-10-01-markov-hmm-redesign-plan.md).*

## Context
- **What exists.** The Markov layer was rebuilt from scratch as a post-prediction model (approved plan `docs/plan/2026-10-01-markov-hmm-redesign-plan.md`). The frozen ConvNeXt + head gives the emission P(Y|X). A transition prior P(Y_t | Y_{t-1}, sex, age, acuity, gap) on (patient, date) states carries the patient information.
  - Code: `analysis/markov_hmm.py`, `analysis/patient_meta.py`, `analysis/markov_hmm_check.py`, `analysis/metadata_eda.py`.
  - The metadata EDA is already in the paper as §V-E.
  - The A/B results are written up: `docs/result/2026-10-01-markov-ab-training.md`, `docs/plan/2026-10-01-markov-ab-verdict.md`.
- **The request.** The user wants `docs/plan/markov-layer-architecture.md` (still the v4 in-network label-layer note) and `docs/paper/markov/paper.pdf` updated to reflect the new architecture.
- **Results to document.** The observed-mode run is done (job 49424). The canonical run with filtered mode (job 49445) is in its last emission.
  - H4 beats the emission on the primary subset at both seeds: +0.038 (s42) and +0.042 (s1024).
  - H4 matches B-early, the in-network prior: the CIs include 0.
  - The simplest calibrated chain (H1) is best.
  - Filtered mode (image-only history) gains only +0.009 on the primary subset, so most of the gain comes from the prior report labels.
  - The within-state MRF edges are never selected.
  - Leaderboard effect: 0 by construction.

## Decisions (user, 2026-10-01)
- **Paper: Extend.**
  - Keep the Task-1 negative-result framing.
  - Rewrite the Longitudinal Markov Model method as the post-prediction HMM, with a TikZ figure.
  - Add an HMM results subsection, including B-early as the in-network comparator.
  - Revise the abstract, Discussion, Conclusion and Reproducibility sections.
- **Architecture note: keep v4 as an appendix.** The main body is rewritten around the HMM, and a condensed v4 (plus v5) description moves to a "Superseded designs" appendix.
- **Wording.** The current Stage-2 head is ConvNeXt2 + ML-Decoder; the MoE head was the earlier ISBI Stage-2. The emission is any classifier's probability dump, so an MoE model plugs in unchanged. Both documents state this.

## Steps
0. **Finish and check job 49445** (`analysis/out/markov_hmm/run.log`, `results.json`, `oof_seed0.npz`).
   - Confirm the built-in asserts printed: H0 = B-late, filtered with one-hot beliefs = observed, the zero-weight identity, and the logistic = counts check.
   - Confirm its observed-mode numbers equal job 49424 (`analysis/out/markov_hmm_obs/`) exactly.
1. **HMM results write-up (cxrlt-research-analyst subagent)** → `docs/plan/2026-10-01-markov-hmm-posthoc.md`. It is an offline analysis, so it goes in docs/plan, linked to the redesign plan, the EDA doc and the A/B verdict. It covers:
   - the arm ladder E/H0–H5/H4-acq/H4-F;
   - the pre-specified decision rule: the H4 win, H4 vs B-early, and the guards (no-prior −0.0009 at s42; new-finding stratum AP −0.003 to −0.006);
   - the exploratory H1 / H4-F − B-early comparisons;
   - the per-half estimator, and why it replaced pooling;
   - caveats: in-sample train emissions in filtered mode (train-dump mAP 0.713), report-derived labels, a val split that is not patient-disjoint, and leaderboard 0.
2. **Rewrite `docs/plan/markov-layer-architecture.md` (main session, design documentation).** Reuse the note's mermaid `classDef` palette and style.
   - Header and TL;DR, dated 2026-10-01. Fix the stale links (`docs/ISBI2026.pdf` → `docs/pdf/ISBI2026.pdf`).
   - §0 big-picture mermaid: X-ray → ConvNeXt → ML-Decoder (frozen) → p(y|x) → calibration → Bayes fusion. Into the fusion feed:
     - the transition prior (patient history: prior report labels or filtered beliefs);
     - metadata (sex, age, acuity, gap);
     - output → fused posterior.
   - §1 lineage, with a one-line verdict and link for each: v3 → v4 in-network label layer (negative) → v5 B-early (prior into the decoder queries) → post-prediction HMM.
   - §2 the model.
     - Graphical-model mermaid: Y_{t-1} → Y_t ← m_t, Y_t → x_t.
     - Equations, from the `analysis/markov_hmm.py` docstring: emission scaled likelihood, initial and transition logistic models, fusion with w_h / w_0, and the observed vs filtered (Boyen–Koller) modes.
     - The within-state MRF ablation.
   - §3 fitting and data flow (mermaid):
     - the prior is fitted on train states only;
     - calibration and weights are cross-fitted on patient halves of val;
     - per-half AP;
     - the data rules (column whitelist, never joining dev/test).
   - §4 results summary: numbers copied from the step-1 doc and `results.json`, with links. It also covers the EDA verdict that metadata are redundant with the image.
   - §5 why post-prediction and not before or inside ConvNeXt:
     - B-early ≈ late fusion;
     - copy risk;
     - metadata are redundant with the image;
     - the post-prediction model is identity when there is no history.
   - §6 code map with `file:line` references for the new modules, plus the reused `patient_prior.py` functions.
   - §7 scope: leaderboard 0, clinical simulation only.
   - Appendix A: the superseded v4 in-network layer (the condensed old §§0, 3–5, its mermaid, and the result line). Appendix B: v5 B-early.
3. **Paper: method, figure and setup (main session).** Back up first: `docs/paper/markov/paper.tex.pre-hmm.bak`.
   - §III-F (`\label{sec:hmm}` kept) becomes "Post-Prediction Patient HMM", with:
     - the equations as in step 2;
     - the observed and filtered modes;
     - the MRF ablation;
     - 2–3 sentences on the B-early comparator (the prior vector added to the ML-Decoder label queries).
   - New TikZ `figure` `fig:hmm`: a two-slice graphical model plus the fusion block (the paper already loads tikz). Add one sentence to the Fig. 1 caption on where the HMM acts.
   - Experimental Setup: a "Longitudinal simulation" paragraph covering the data rules, the train-only prior fit, cross-fitting on patient halves, the per-half AP, the arms, and the bootstrap.
4. **Paper: results and summary text (cxrlt-research-analyst subagent)**, from the step-1 doc and `results.json` only.
   - In §V-D, keep the leaderboard-impossibility paragraph, then add the HMM results.
   - Add `table*` `tab:hmm`: arms × {full, primary, no-prior} with Δ [CI] at both seeds, including B-early and B-early0. Add a filtered-mode paragraph and an ablation paragraph, and cross-reference §V-E.
   - Abstract: replace "A longitudinal hidden Markov model is ruled out by the challenge design." with at most two sentences: unusable on the leaderboard, but in a longitudinal simulation +0.04 primary, matching B-early, with most of it from the prior report labels.
   - Update Discussion, Conclusion and Reproducibility (add the four scripts).
5. **Compile and check:** `latexmk -pdf` in `docs/paper/markov/`.
   - No errors and no undefined references.
   - No overfull box over 10pt from the new text.
   - Page count, and float pages from `paper.aux`.
6. **Memory:** update `posthoc-patient-hmm.md` with the canonical numbers (filtered mode, H1 − B-early).

## Files
- **Rewritten:** `docs/plan/markov-layer-architecture.md`.
- **Edited:** `docs/paper/markov/paper.tex` (and `paper.pdf`).
- **New:** `docs/plan/2026-10-01-markov-hmm-posthoc.md`.
- **No code changes.** Figures are TikZ in the paper and mermaid in the note.

## Verification
- **Asserts:** the job-49445 asserts are present in `run.log`, and the observed numbers equal job 49424.
- **Numbers:** every number in the note and the paper traces to `results.json` / `run.log`. Spot-check with a grep of 5–6 values against `results.json`.
- **Paper build:** latexmk is clean, and the `\ref`/`\label` pairs for `fig:hmm`, `tab:hmm` and `sec:hmm` resolve in `paper.aux`.
- **Note:** the mermaid blocks keep the existing syntax and palette, and the relative links in the note resolve to existing files (`ls` each one).
