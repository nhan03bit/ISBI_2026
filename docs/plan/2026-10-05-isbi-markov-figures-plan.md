# Plan: add the Markov core-design figures to the merged ISBI paper

## Context
`docs/paper/isbi/paper.tex` is now the entropy paper copied verbatim, followed by a shortened Section VII, "Markov Priors for Label Co-occurrence" (rev5 and rev5b). The user is happy with the entropy part. In the Markov part, the trimming dropped all the design figures:
- the pipeline figure, which shows where the prior enters;
- the graphical HMM figure;
- the four metadata figures in `docs/paper/markov/figures/`.

No figure shows the label-to-label co-occurrence chain itself.

The user chose:
- all four metadata figures;
- a new co-occurrence graph built from the real training-fold transition matrix P.

This is a paper edit, not a finished run, so the cxrlt-research-analyst subagent has nothing to analyse. All numbers come from existing results (`docs/paper/markov/paper.tex`, `analysis/markov_check.py`, `analysis/out/metadata_eda/`) or from P, computed fresh.

## Steps

1. **Back up.** Copy `isbi/paper.tex` and `isbi/paper.pdf` to `*.pre-rev6.bak`.

2. **Preamble.** Add `\usepackage{tikz}` and `\usetikzlibrary{positioning,arrows.meta,calc}`.

3. **Fig. A, pipeline.** Use the user's TikZ code verbatim. Place it at the top of §VII, after the route (A)/(B) paragraph.
   - Fix the caption references to the merged labels:
     - `sec:phprior` and `sec:hmm` become `sec:mk-patient`;
     - `sec:smoothing` and `sec:gating` become `sec:mk-labelgraph`;
     - `fig:hmm` stays.

4. **Fig. B, co-occurrence chain (new), in Route A after Eq. (transition).**
   - Compute P read-only with the existing code: `markov_check.train_label_matrix(classes)` with the class order from `LABEL_INFO`. Use the same normalisation as `label_graph_smoothing()` (row-stochastic, zero diagonal; this equals Eq. `transition`).
   - Run it as a small scratchpad script with `.venv/bin/python`. Print, for each of these rows, the top 2–3 outgoing P_ij and the training count:
     - Hydropneumothorax
     - Pneumothorax
     - Pleural effusion
     - Support Devices
     - Normal
     - one more tail class
   - Draw a TikZ node-link graph with about 6 nodes, edge labels set to the printed P values, and nodes shaded head/medium/tail.
   - Caption: one random-walk step (Eq. `smoothing`) moves probability along these edges. The Normal row must sum to one, so P cannot encode exclusivity (the Normal-gating contrast).
   - Every value in the figure is copied from the script output, not typed by hand.

5. **Fig. C, graphical HMM.** Use the user's TikZ code verbatim and place it in Route B, after the post-prediction paragraph. To make its caption references real:
   - Expand Eq. `trans` to the full H4 form: α_c + β_c y'_c + κ_c y'_c g + λ_c g + Σ_{k≠c} B_ck y'_k + γ_cᵀφ(m), with g = log(1+Δ) and φ(m) = sex, age spline, acuity. Add the one-line init prior π⁰ as `eq:init`.
   - Label the calibrated emission ℓ_c(x) = τ logit p_c + δ_c − logit π̄_c as `eq:emission`.
   - Add the case split for (q_c, w) as `eq:qw`, reusing `eq:fuse`.
   - The filtered-mode equations are not in the shortened text, so the caption's "Eqs. predict–update" becomes "by a forward pass over the earlier images". Caption otherwise verbatim.
   - Add one sentence pointing out that the cross-label edges B_ck (H4) and within-visit couplings J (H5) are the co-occurrence terms inside this model. Results already show they add nothing.

6. **New subsection "Patient Metadata", before Discussion.** About 2 paragraphs, condensed from `markov/paper.tex` §V-E (lines ~438–498), with the numbers unchanged:
   - Associations: acuity V 0.528, projection 0.504, age 0.447, gap 0.447, sex 0.133.
   - Metadata alone: AUROC 0.760, AP 0.142.
   - Stacking metadata on the image lowers mAP by 0.003 to 0.016, with CIs excluding 0 in 15 of 16 cells.
   - Persistence depends on acuity: Support Devices 0.578 vs 0.834, CVC 0.269 vs 0.812, Normal 0.676 vs 0.258.
   - Include the four figures (`assoc_heatmap` and `age_prevalence` as `figure*`, `persistence_by_acuity` and `metadata_only_auroc` as column figures) with their original captions.
   - `\includegraphics` paths point to `figures/…`, i.e. `isbi/figures/`, whose files are byte-identical to `markov/figures/` (verify with `cmp`).
   - Remove the now-duplicated metadata sentence from "Inside the patient model…" and cross-reference this subsection instead.

7. **Leave everything else in rev5b as is.** That covers the entropy text, the tables and the corrected claims. Specifically:
   - no "solves co-occurrence" wording;
   - no tail claims;
   - the rerun numbers stay.

## Verification
- `cd docs/paper/isbi && TEXINPUTS=../../pdf//: latexmk -pdf -interaction=nonstopmode paper.tex`, then grep `paper.log` for `undefined`, `Overfull` and `^!`. All three must be empty.
- Render pages 6 onward with pymupdf, already in the venv, and inspect:
  - figure placement and legibility;
  - that no TikZ node overlaps another;
  - that `figure*` floats are not pushed past the references.
- Cross-check every P value in Fig. B against the script output, and every metadata number against `markov/paper.tex`.
- Report the final page count (expected about 12).
- Update the rev6 line in the `paper-revision-confounds` memory.
