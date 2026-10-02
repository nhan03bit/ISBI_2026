# Literature notes: patient-history / longitudinal CXR modelling (verified 2026-09-30)

All metadata below comes from the arXiv API, Crossref or a page fetch. None of it is from memory.

## Longitudinal CXR models that use prior studies
1. **Santeramo, Withey, Montana**, "Longitudinal detection of radiological abnormalities with time-modulated LSTM", DLMIA/ML-CDS @ MICCAI 2018, arXiv:1807.06144.
   - Models the whole sequence of a patient's radiographs, *including their reports*, with an LSTM that is explicitly modulated by the time lag between exams, because exams are irregularly sampled.
   - Abstract: improved detection of cardiomegaly, consolidation, pleural effusion and hiatus hernia. No numbers in the abstract.
   - Relevance: history helps inside one hospital when priors exist. Integration is a sequence model over per-exam features, i.e. after the CNN.
2. **Bannur et al.**, "Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing" (BioViL-T), CVPR 2023, arXiv:2301.04558.
   - Multi-image encoder that takes the prior image and report when available, in both training and fine-tuning. Targets progression tasks.
3. **Karwande et al.**, "CheXRelNet: An Anatomy-Aware Model for Tracking Longitudinal Relationships between Chest X-Rays", MICCAI 2022, arXiv:2208.03873.
   - Pairwise change relations between two CXRs.
4. **Yu, Ghosh, Liu, Deible, Poynton, Batmanghelich**, "Anatomy-specific Progression Classification in Chest Radiographs via Weakly Supervised Learning", Radiology: Artificial Intelligence 6(5), 2024 (doi 10.1148/ryai.230277).
   - Twin network. Classes: improved / unchanged / worsened / new.
5. **Hu et al.**, "Learning Directional Semantic Transitions for Longitudinal Chest X-ray Analysis" (ProTrans), MICCAI 2026, arXiv:2606.15938.
   - Progression framed as a directional transition between paired studies.
6. **Prakash et al.**, "CheXTemporal: A Dataset for Temporally-Grounded Reasoning in Chest Radiography", arXiv:2605.11304 (2026).

Common thread: every one of these needs the patient's **prior study available at inference**. Most target *progression* labels (improved / stable / worse), not static finding presence.

## Evidence on copying priors when they are fed as model input (the shortcut risk)
7. **Zhu, Mathai, Mukherjee, Peng, Summers, Lu**, "Utilizing Longitudinal Chest X-Rays and Reports to Pre-Fill Radiology Reports", arXiv:2306.08749 (2023).
   - Inputs: prior CXR, prior report and current CXR.
   - Error Analysis, verbatim: *"when the label results of current and previous report are the same, 88.96% percent of the generated results match them. On the other hand, despite mentioning the same observations, when the labels of current and previous report are different, there is an 84.42% probability of generated results being incorrect."*
   - This is direct evidence that input-level conditioning on prior findings makes the model copy the prior. It fails precisely on the changes that matter clinically.

## HMM disease-progression modelling (general)
8. Personalized input-output HMMs for disease progression (medRxiv 10.1101/2020.07.17.20153510), and continuous-time HMMs for progression. Title-level only, from search. The emission × transition factorisation is standard: the per-visit observation model is the emission and the chain is the prior. Cite by title only if used.

## Challenge constraints (CXR-LT 2026 challenge paper, arXiv:2604.15555, pp. 4–5 and §2.7)
- *"Because PadChest-GR is derived from PadChest, we performed both patient-level and study-level de-duplication to prevent leakage."*
- *"Patient identifiers and hidden evaluation labels were not released to participants."*
- Participants must follow *"the applicable data-use restrictions, including research-only use, no redistribution, and no attempts to re-identify individual patients."*
- Participants may not *"use hidden test labels, manually relabel evaluation images, or use external annotations overlapping with the CXR-LT evaluation sets."*
- Local fact: the PadChest 160K CSV has `Report` and `Labels` columns for every PadChest image. That makes it an external annotation overlapping the evaluation set for any PadChest-GR image.
