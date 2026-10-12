# Supervisor framing check of `isbi_submit/paper.tex` (2026-10-12)

**Document:** `docs/paper/isbi_submit/paper.tex` (line numbers below refer to the file as read on 2026-10-12).
**Context read:** `docs/plan/2026-10-08-entropy-ablation-isbi-submit-plan.md` (pre-registration), `docs/plan/2026-10-11-isbi-author-rewrite.md`.
**Instruction audited (verbatim):** "So I think the novelty needs to be framed specifically around the entropy-gated anchor selection + cross-batch memory, rather than as a general solution to long-tail imbalance/co-occurrence."
**Scope:** framing and claims audit only. No result numbers are introduced; `paper.tex` was not edited. This is not a training-run report.

## Verdict: PARTLY satisfied

The structure already points the right way: the title names the gate and the bank, the method sections are titled "Cross-batch memory bank" and "Entropy-gated anchors", contribution 2 is a controlled ablation, and the text refuses to credit the gate before the ablation (L51, L86, L145). The framing still falls short in five ways.

1. The Introduction (L47) and the Abstract (L38) open with long-tail imbalance and label co-occurrence as the problem the method addresses. This is the framing the supervisor asked us to drop.
2. The title ends in "for Long-Tailed Multi-Label Chest X-Ray Classification", which reads as a general long-tail solution.
3. Nowhere does the paper say what is new and what is borrowed (triplet loss, XBM, entropy-as-uncertainty are all existing).
4. Contribution 3 (head/medium/tail AP) is a long-tail reporting item, not a statement about the entropy gate.
5. Several sentences presuppose that the gate works (L49, L53, L118, L147, L162).

All five are fixable with local text edits. They matter most if the gate effect is null (section 3).

## 1. Line-by-line inventory

### 1a. Wording that frames the work as a general long-tail / co-occurrence solution

| Line | Text | Problem | Severity |
|---|---|---|---|
| 26 | Title: "...for Long-Tailed Multi-Label Chest X-Ray Classification" | Names the problem class as the contribution; the gate and bank are only the modifiers. | Medium |
| 38 | Abstract sentence 1: "Long-tailed, multi-label chest X-ray classification is hard because rare findings receive little gradient signal and seldom appear without the common findings they co-occur with." | The abstract opens with long-tail and co-occurrence as the problem. The mechanism claim ("little gradient signal", "co-occur") is uncited, and nothing in the paper tests it. | High |
| 42 | Keywords: "long-tailed classification, multi-label classification" | Acceptable as task keywords, but "anchor selection" and "ablation" are absent, so the novelty is not indexed. | Low |
| 47 | Intro paragraph 1: "Rare findings are also seldom seen alone, so their small positive gradient is entangled with that of the common findings they co-occur with." | A co-occurrence mechanism asserted as fact; no experiment here measures it. | High |
| 47 | "Imbalance-aware ... losses ... reweight ... but do not act on the shared representation directly." | Sets up "ASL fixes the loss, we fix the representation", i.e. the triplet term as the long-tail fix. | Medium |
| 49 | "...supplies positives and negatives for rare findings that a mini-batch of eight rarely contains." | A rationale for the bank (reasonable), but it implies a rare-finding benefit that Table 1's tail column has not shown. | Low-Medium |
| 55 | Contribution 3: "head, medium and tail AP for every arm" | Frames a long-tail breakdown as a contribution. It is a reporting item and does not concern the gate. | Medium |
| 120, 128 | head/medium/tail definition and table columns | Fine as descriptive reporting. The danger is only a Results or Conclusion sentence that reads a tail "gain" into 7 findings with 7-140 validation positives. | Low |
| 145 | Higher-resolution paragraph (0.460/0.462 single, 0.472 ensemble) | Sits in Results next to the ablation and can read as a general performance claim for the gate. The sentence "All of these used the entropy gate and were not ablated" is correct but is too weak (see 3b). | Medium |
| 147 | MoE comparison paragraph and Table `tab:moe` | A "general improvement vs the previous system" story. It is also outside the plan's scope ("Scope: entropy only; no MoE experiments", plan L148-150). | Medium |

### 1b. Wording that frames the novelty correctly (keep)

| Line | Text | Why it is good |
|---|---|---|
| 26 | "Entropy-Gated Triplet Cross-Memory-Bank Learning" | Names both the gate and the bank. |
| 38 | "a triplet loss against a cross-batch memory bank whose anchors are the images with above-average predictive entropy" | States the mechanism exactly. |
| 38 | "To attribute any gain to the entropy gate, we run a like-for-like ablation..." | Puts the gate at the centre and treats the gain as unknown. |
| 49 | "Anchors are restricted to the images whose predictive entropy is at or above the batch mean" | Precise statement of the rule. |
| 51 | "Whether such a gate helps cannot be read from the accuracy of the full model, because the gate, the triplet loss and the memory bank are confounded..." | The core of the supervisor's point; keep verbatim. |
| 53 | Contribution 1: "an entropy-gated anchor rule... adds no parameters and is discarded at inference" | Names the gate; does not claim the bank. See 2. |
| 54 | Contribution 2: the A/B/B'/C ablation | The most compliant item. |
| 73 | "...at no extra forward cost~\cite{wang2020xbm}" | Credits XBM. |
| 75-86 | Section "Entropy-gated anchors"; "it is the rule we ablate" | Correct and honest. |
| 105 | "These are easy pairs rather than the hardest ones; we did not compare harder mining." | Honest scoping. |
| 145 | "All of these used the entropy gate and were not ablated." | Honest, but see 3b. |
| 162 | Limitations: softmax over binary labels; no harder mining; one internal fold | Good. |

## 2. The three contributions, checked

**What is actually new, stated honestly.** Triplet loss (Schroff, Hermans), the cross-batch memory bank (Wang et al., XBM), predictive entropy as an uncertainty score (active learning, Settles) and ASL are all existing. The only candidate novelty is a design choice. In an XBM-style multi-label triplet objective, the classifier's own batch-relative softmax entropy decides which batch images become anchors against the bank. A second, smaller design element is the multi-label positive definition: at least one shared finding outside the five most frequent, with a 30-way multi-hot overlap test for negatives. No part of the paper currently says this plainly.

I have not run a literature search. The authors should check whether uncertainty- or entropy-based anchor or sample selection already exists in metric learning before writing "to our knowledge, new". Safer wording is "we study", "we examine" or "we propose and test", not "novel".

| # | Text (L) | Names gate? | Names bank? | Verdict |
|---|---|---|---|---|
| 1 (L53) | "an entropy-gated anchor rule for multi-label triplet learning with a memory bank, with positives defined on shared non-dominant findings; it adds no parameters and is discarded at inference" | Yes | Yes, but only as the setting | Mostly OK. Problems: (a) it does not say the bank, triplet loss and entropy are existing; (b) it bundles an untested design detail (the positive definition) with the gate, which blurs what the paper evidences; (c) as a "contribution" it presupposes that the rule is a positive result. |
| 2 (L54) | "a like-for-like ablation with checkpoint, resolution, schedule, augmentation and six seeds held fixed: ASL alone; triplet loss and bank with every image as an anchor, or with a random subset of the gate's size; and with the entropy gate" | Yes (isolates it) | Yes (B vs A isolates triplet/bank) | Compliant. Strengthen by naming the primary contrast (C-B) and B' as the "which anchors vs how many" control. |
| 3 (L55) | "head, medium and tail AP for every arm, with seed-level and image-level uncertainty" | No | No | Not compliant as a novelty. It is a long-tail reporting item that says nothing about the gate or the bank. Fold it into contribution 2 and use slot 3 for the finding or the protocol. |

**Honesty point to fix in the Statistics paragraph (L122).** The text says "The primary contrast, fixed before the runs, is C-B in mAP". Per the plan, arm C reuses six runs that existed before the plan was written (plan L41-45; their mean was already known, 0.4394 in the rewrite fact sheet L60). Only A, B, B' and the two C reruns were run after the contrast was fixed. The paper should say so. It should also say "fixed before the new runs (plan dated 2026-10-08)" rather than "pre-registered" (L1 comment and the plan call it pre-registration, but it is an internal dated plan, not a public registry).

## 3. Evidence state and wording that pre-judges the gate

The C-B contrast has not been measured; the analysis has not been run. The caller reports that early per-run mAPs for A, B, B' and C all lie around 0.433-0.448. The arm-C seeds already known span 0.4369-0.4434 (plan L42), so the seed spread is of the same order as that overlap. A null gate effect is therefore a live outcome, and the framing must work for it.

### 3a. Sentences that presuppose a positive gate effect

| Line | Text | Issue |
|---|---|---|
| 26 | Title as a whole | A functioning method named "Entropy-Gated ... Learning" before the evidence. |
| 38 | "We study a training-only regulariser for fine-tuning..." | Fine, but "regulariser" assumes a benefit. Say "objective". Low severity. |
| 49 | "so that the metric-learning signal is spent on the images the current model resolves least well" | Presupposes that high softmax entropy identifies the images the model resolves worst. This is untested: no analysis relates entropy to error, and the entropy is a softmax over independent binary labels (L86). Make it a stated hypothesis. |
| 53 | Contribution 1 listed as a contribution before the result | If C-B is null, it is a candidate rule that was evaluated, not a validated contribution. |
| 118 | "\emph{C}: the method" | Calls the gated arm "the method" in an ablation whose point is to test it. Use "the entropy-gated rule". |
| 147 | "the entropy-gated triplet loss and the MoE are indistinguishable... it adds no parameters at inference" | Labels the comparison arm "entropy-gated" although the gate is not isolated at 768 px and above. Call it "the triplet/memory-bank objective (with the entropy gate)". |
| 145 | Resolution and ensemble paragraph, "with the same objective" | Reads as gate credit. Needs one added clause: these numbers cannot be attributed to the gate or the bank. |
| 162 | "the absolute values are therefore likely optimistic, although every arm is affected in the same way" | The "every arm is affected in the same way" assertion is untested. A patient-overlapping fold could favour a metric-learning term differently from ASL alone (plan Step 3 lists a frontal and unseen-patient subset). Hedge, or support it with that subset. |
| 143 (TBD) | "(2) the entropy gate, primary C-B ..." | Fine as a TBD. |

### 3b. Pre-judging by omission

Nothing in the draft says what the paper will claim if C-B is null. The plan's decision point (plan L138-140) says to agree the framing with the user before Results are written. The L143 TBD records this correctly, and the same discipline should be applied to the Abstract (L38 TBD), Discussion and Conclusion.

### 3c. Honest framing if the result is null

Keep the supervisor's focus: the paper's subject remains entropy-gated anchor selection plus the cross-batch memory bank. The question changes from "does it help?" to "is the gate what helps?". The pre-registered contrasts then decide the sentence:

| Outcome | Honest claim |
|---|---|
| C-B positive with a 95% CI excluding 0, and C-B' positive | Entropy-selected anchors outperform both all anchors and equally many random anchors in mAP. State the CI, the sign-flip p, wins out of 6, and the single-recipe, single-dataset, internal-fold limits. Do not generalise to long-tail imbalance. |
| C-B null, B-A positive | The gain over ASL is attributable to the triplet/memory-bank objective, not to entropy gating. The gate is not shown to be necessary. Report the CI as the bound on any gate effect we could exclude. |
| C-B null, C-B' null | Which anchors are chosen does not measurably matter at this scale, including with only a few of the 8 batch images used as anchors. Do not claim a compute or efficiency advantage unless it is measured; anchor-count (anchors/step) is reported, wall-clock is not. |
| C-B null and B-A null | Under a controlled 512-px recipe, neither the bank objective nor the gate measurably improves on ASL alone. The earlier higher-resolution gains (0.460/0.462; ensemble 0.472) cannot be attributed to either; they come from a different recipe (resolution, schedule, weight averaging, ensembling). Make this the main finding. |
| C-B negative | Report the gate as harmful or neutral at 512 px, with the CI. |

In every null case:
- Say "no measurable difference at six seeds (95% CI [..])", not "no effect".
- Do not claim a tail benefit from the 7-finding tail group (7-140 validation positives per finding) unless the paired contrast supports it. Report it descriptively.
- The Abstract's TBD sentence is already marked "worded by the pre-registered outcome". Keep that rule.

## 4. Suggested rewrites (LaTeX-ready)

None of these contain result numbers. Where a result is needed, a `\TBD[...]` marks it.

### 4.1 Title (L26)

Neutral and outcome-safe (preferred until results are known):
```latex
\title{Entropy-Gated Anchor Selection for Triplet Learning with a Cross-Batch Memory Bank: A Controlled Ablation on Multi-Label Chest X-Ray Classification}
```
If the authors want to keep the setting in the title:
```latex
\title{Entropy-Gated Anchors for Cross-Batch-Memory Triplet Learning: A Controlled Ablation in Long-Tailed Multi-Label Chest X-Ray Classification}
```
If C-B is positive, a shorter positive form is acceptable: "Entropy-Gated Anchor Selection for Cross-Batch Memory Triplet Learning in Chest X-Ray Classification". If it is null, consider "Is Entropy-Gated Anchor Selection Necessary for Cross-Batch Memory Triplet Learning? A Controlled Ablation on Chest X-Ray Classification".

### 4.2 Keywords (L42)
```latex
Chest X-ray, multi-label classification, deep metric learning, triplet loss, cross-batch memory, anchor selection, predictive entropy, ablation
```
("Long-tailed" is dropped as a keyword; it stays in the Setup as the dataset property.)

### 4.3 Abstract, everything except the TBD sentence (L38)
```latex
A triplet loss against a cross-batch memory bank needs a rule for which images in a mini-batch act as anchors. We study one such rule, entropy-gated anchor selection: only images whose predictive entropy is at or above the batch mean serve as anchors, and positives and negatives are mined from the bank by shared findings. The objective is added to Asymmetric Loss (ASL) when fine-tuning a ConvNeXt/ML-Decoder multi-label chest X-ray classifier; it is used only in training and adds no parameters at inference. To isolate the gate, we run a like-for-like ablation on CXR-LT 2026 Task~1, a long-tailed 30-finding benchmark, with the Stage-1 checkpoint, input resolution, schedule, augmentation and six seeds held fixed: ASL alone; the triplet/memory-bank objective with every image as an anchor, or with a random subset of the gate's size; and the same objective with the entropy gate. \TBD[: two sentences with the primary result (C\,$-$\,B mAP with 95\% CI), the triplet/bank effect (B\,$-$\,A) and the head/medium/tail pattern, worded by the outcome]
```
Null-safe form of the final TBD sentence, to adapt once results exist:
```latex
Entropy gating did not change mAP measurably relative to using every image, or an equally large random subset, as anchors (\TBD[: C\,$-$\,B with 95\% CI]), whereas the triplet/memory-bank objective \TBD[: B\,$-$\,A with 95\% CI]; the gate is therefore \TBD[: not shown to be necessary / not the source of the gain].
```

### 4.4 Introduction paragraph 1 (L47)
```latex
Findings on chest radiographs follow a long-tailed distribution. In the CXR-LT 2026 Task~1 data~\cite{dong2026cxrlt}, built on PadChest~\cite{bustos2020padchest}, three of the 30 findings are each present in more than 10\% of our training images, while seven are present in fewer than 1\%, the rarest in 38 of 103{,}303. We use this benchmark as a test bed for one design choice of a metric-learning regulariser, namely which images serve as anchors, and not as a claim to solve long-tail imbalance or label co-occurrence. Imbalance-aware losses such as Asymmetric Loss (ASL)~\cite{ridnik2021asl} or distribution-balanced loss~\cite{wu2020dbloss} reshape the per-label loss terms; a metric-learning term instead acts on the shared embedding.
```
The sentence on co-occurrence entanglement (L47) is deleted. If the authors want to keep it, reword it as a hypothesis ("may be entangled") and cite or measure it.

### 4.5 Introduction paragraph 2, last two sentences (L49)
```latex
Here we instead add a training-only triplet loss~\cite{schroff2015facenet,hermans2017defense} computed against a cross-batch memory bank~\cite{wang2020xbm}, which supplies positives and negatives for rare findings that a mini-batch of eight images rarely contains. Triplet loss, the bank and entropy as an uncertainty score~\cite{settles2009active} are existing components; what we examine is their combination, in which the classifier's own predictive entropy decides which batch images become anchors: only images at or above the batch mean. The intent is to spend the metric-learning signal on images the current model resolves least well; whether this selection matters is the question we test.
```
(Keep the first sentence of paragraph 2, about the prior MoE entry, unchanged.)

### 4.6 Contributions (L51-56)
```latex
Whether such a gate helps cannot be read from the accuracy of the full model, because the gate, the triplet loss and the memory bank are confounded with each other and with the fine-tuning itself. We therefore make the gate the object of study. Our contributions are:
\begin{tightlist}
\item the entropy-gated anchor rule: in a triplet loss against a cross-batch memory bank, only batch images whose softmax entropy is at or above the batch mean are anchors. The bank, the triplet loss and entropy are existing; the rule is training-only and adds no parameters at inference;
\item a like-for-like ablation that separates the gate from the triplet/bank objective and from the number of anchors (arms A, B, B$'$, C; checkpoint, resolution, schedule, augmentation and six seeds held fixed), with the primary contrast C\,$-$\,B fixed before the new runs, reported as mAP and as head, medium and tail AP with seed-level and image-level uncertainty;
\item \TBD[: one sentence stating the finding, worded by the outcome of C\,$-$\,B and B\,$-$\,A, e.g.\ ``the gain over ASL comes from the triplet/memory-bank objective, not from entropy gating'' if C\,$-$\,B is null].
\end{tightlist}
```
The tail breakdown is now folded into contribution 2 and no longer stands as a contribution of its own.

### 4.7 Arm C and the Statistics paragraph (L118, L122)
In L118 replace "\emph{C}: the method, Eq.~(\ref{eq:gate})." with:
```latex
\emph{C}: entropy-gated anchors, Eq.~(\ref{eq:gate}); its six seeds are earlier runs of the same recipe, with two seeds rerun as a reproducibility check.
```
In L122 replace "The primary contrast, fixed before the runs, is C\,$-$\,B in mAP." with:
```latex
The primary contrast, C\,$-$\,B in mAP, was fixed in a written plan before the arms A, B and B$'$ were run; the six arm-C runs predate that plan.
```

### 4.8 Higher-resolution paragraph (L145), last sentence
Replace "All of these used the entropy gate and were not ablated." with:
```latex
All of these used the entropy gate and were not ablated, so they cannot be attributed to the gate, to the triplet/bank objective, or to the resolution and schedule; only the 512-px ablation above isolates the gate.
```

### 4.9 MoE paragraph (L147)
Preferred: shorten to two sentences or move to supplementary (the plan scoped the paper to entropy only; it also takes space from the four technical pages). If kept, change "the entropy-gated triplet loss and the MoE are indistinguishable" to:
```latex
the triplet/memory-bank objective (with the entropy gate) and the MoE are indistinguishable
```
and add: "This comparison does not isolate the gate."

### 4.10 Discussion first clause and limitation (L162)
Replace "although every arm is affected in the same way" with:
```latex
we expect, but have not shown, that contrasts between arms are less affected than absolute values \TBD[: or cite the frontal / unseen-patient subset result]
```
After the TBD interpretation, add one sentence on scope:
```latex
Our conclusions concern anchor selection in this objective, on this benchmark, at 512~px; they are not evidence that triplet or memory-bank regularisation addresses long-tail imbalance or label co-occurrence in general.
```

### 4.11 Conclusion (L166)
If C-B is clearly positive:
```latex
In a like-for-like ablation on CXR-LT 2026 Task~1 at 512~px, restricting the anchors of a cross-batch-memory triplet loss to images with above-average predictive entropy changed mAP by \TBD[: C\,$-$\,B with 95\% CI] relative to using every image, and by \TBD[: C\,$-$\,B$'$] relative to an equally large random subset. This supports entropy-gated anchor selection as a training-only component; it does not show a general remedy for long-tailed or co-occurring findings.
```
If C-B is null (adapt to B-A):
```latex
In a like-for-like ablation on CXR-LT 2026 Task~1 at 512~px, entropy-gated anchor selection did not change mAP measurably relative to using every image, or an equally large random subset, as anchors (\TBD[: C\,$-$\,B and C\,$-$\,B$'$ with 95\% CIs]). \TBD[: if B\,$-$\,A is positive: ``The gain over ASL is attributable to the triplet/memory-bank objective rather than to the gate.'' If null: ``Neither the gate nor the bank objective measurably improved on ASL alone under this recipe.''] The higher-resolution and ensemble results were not ablated and are not evidence for the gate.
```

## 5. Further checks and caveats

- **Plan scope.** The plan fixes "No Markov, no MoE experiments" (plan L148-150), but the draft carries a full MoE comparison paragraph and `tab:moe` (L147-158). This broadens the paper beyond the supervisor's focus and consumes technical-page space. Raise with the supervisor.
- **Figure caption (L91).** The caption says the figure "calls these the hardest positive and negative; ... they are the easiest pairs". This is self-contradictory text left in a caption; redraw or relabel the figure rather than apologise in the caption.
- **Triplet utilisation.** Anchors/step and triplets/step are TBD (L86). The earlier 7.6% vs 13% triplet-utilisation discrepancy (project memory, paper-revision-confounds) is relevant here: report the measured numbers from the ablation logs and do not reuse an older figure.
- **Statistical power.** With six seeds, the CI on C-B may be wide relative to a plausible gate effect. A null result should be written as "no measurable difference" with the CI, not as evidence of equivalence.
- **AI-drafting disclosure.** The framing rewrites above are drafting suggestions; the authors' own rewrite (per `docs/plan/2026-10-11-isbi-author-rewrite.md`) should replace them, and the disclosure at L178 should be updated accordingly.
- **Not verified.** I have not checked the citations (`wang2020xbm`, `settles2009active`, `hermans2017defense`) against the `.bib` file, and I have not done a literature search on entropy-based anchor selection in metric learning.

## Required changes in priority order (summary)

1. Open the Abstract (L38) and Introduction (L47) with the anchor-selection question, not long-tail or co-occurrence. Delete the unevidenced co-occurrence mechanism sentence. State that the benchmark is a test bed.
2. Add an explicit novelty and attribution sentence: triplet, XBM and entropy are existing; only the entropy gate on anchors, in this multi-label bank objective, is examined (4.5, 4.6).
3. Rewrite the three contributions: contribution 1 attributes the existing components; contribution 2 names the primary contrast C-B; the tail AP item is folded into 2 and slot 3 becomes the outcome-dependent finding (4.6).
4. Change the title so that the gate and bank lead and the task is the setting (4.1). Strip "regulariser" and "the method" presuppositions (L38, L118, L147).
5. Fix the honesty issues: C's runs predate the plan (L122); the resolution, ensemble and MoE results do not isolate the gate (L145, L147); "every arm is affected in the same way" is untested (L162); decide whether to keep the MoE paragraph. Agree the null-result framing (section 3c) with the supervisor before the Results, Discussion and Conclusion TBDs are written.

## Notable

This is an audit, not a new result. It records one inconsistency worth the orchestrator's attention: the paper (L122) says the primary contrast was fixed "before the runs", but the arm-C runs predate the plan. Whether to update `paper-revision-confounds` memory is the main session's call.
