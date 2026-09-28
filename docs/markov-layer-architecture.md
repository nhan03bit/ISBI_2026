# The Markov label layer in our Stage-2 model — and how it relates to the ML-Decoder

*2026-09-26 · Stage-2 v4 (`train/train_2_v4.py`, `train/markov_layer.py`). Background:
[`docs/ISBI2026.pdf`](ISBI2026.pdf), [`docs/Entropy.pdf`](Entropy.pdf),
[`result/2026-09-26-markov-applicability.md`](result/2026-09-26-markov-applicability.md).*

**TL;DR**

- The Markov layer is a small head **after** the ML-Decoder logits. It passes the predicted label
  probabilities through a 30×30 label-transition matrix `P`, initialised from training co-occurrence
  and trainable. It adds the result back to the logits through a per-class gate that starts at zero,
  so the model **starts exactly as the v3 baseline**.
- The ML-Decoder already does something similar. Its query **self-attention** is a row-stochastic 30×30
  mixing over the label queries. That mixing is *recomputed for every image* and works in the 768-d
  embedding space. The Markov layer adds **one global, interpretable** transition matrix in
  probability space.
- That overlap is why a *fixed* co-occurrence smoothing lowered mAP offline (−0.0006 to −0.0053). The
  trainable, identity-initialised version is being tested in jobs mk1 / mk2 / mk1fix against v3 job
  49000.

---

## How to read the diagrams

Every diagram below uses the same colour key.

| Colour | Meaning |
|---|---|
| 🟦 blue | the existing v3 network: backbone and ML-Decoder |
| 🟧 orange, thick border | **new in v4**: the Markov label layer and its scripts |
| ⬜ white, rounded | data flowing through the model (tensor shapes are at 768 px) |
| 🟪 purple | losses |
| dashed grey | the triplet branch: shapes the embedding space, never touches the logits |
| 🟥 red, dashed | do not use |
| 🟩 green | rare (tail) disease class (§5 only) |
| light grey | earlier work (§1 only) |

Shapes: a **rectangle** is a computation, a **rounded box** is data, a **cylinder** holds stored or
learned values (weights, checkpoints).

## 0. The big picture

```mermaid
%%{init: {"theme": "base", "flowchart": {"wrappingWidth": 400, "padding": 14}, "themeVariables": {"fontSize": "15px", "lineColor": "#64748B", "primaryTextColor": "#0F172A", "edgeLabelBackground": "#FFFFFF", "clusterBkg": "#F8FAFC", "clusterBorder": "#CBD5E1", "titleColor": "#0F172A"}}}%%
flowchart TB
    X(["<b>Chest X-ray</b>"]):::data
    BB["<b>ConvNeXt backbone</b><br/>turns the image into visual tokens"]:::v3
    DEC["<b>ML-Decoder</b><br/>30 label queries, one per disease<br/>① each query looks at the image<br/>② the queries look at each other"]:::v3
    MK["<b>Markov label layer</b> · NEW in v4<br/>co-occurring labels vote for each other"]:::new
    L["<b>Asymmetric loss</b> · mAP"]:::loss
    T["triplet loss<br/>shapes the embeddings only"]:::aux
    X --> BB
    BB -->|"144 visual tokens"| DEC
    DEC -->|"30 scores u"| MK
    MK -->|"30 final scores z"| L
    DEC -.-> T

    classDef v3 fill:#DBEAFE,stroke:#1D4ED8,stroke-width:1.5px,color:#1E3A8A
    classDef new fill:#FFEDD5,stroke:#C2410C,stroke-width:3px,color:#7C2D12
    classDef data fill:#FFFFFF,stroke:#64748B,stroke-width:1px,color:#0F172A
    classDef loss fill:#EDE9FE,stroke:#6D28D9,stroke-width:1.5px,color:#4C1D95
    classDef aux fill:#F1F5F9,stroke:#94A3B8,stroke-width:1px,stroke-dasharray:5 4,color:#475569
    classDef warn fill:#FEE2E2,stroke:#B91C1C,stroke-width:2px,stroke-dasharray:5 4,color:#7F1D1D
    classDef old fill:#F1F5F9,stroke:#94A3B8,stroke-width:1px,color:#334155
    classDef tail fill:#DCFCE7,stroke:#15803D,stroke-width:2.5px,color:#14532D
```

Everything before the orange box is the v3 model, unchanged. The Markov layer is a small add-on
(960 parameters) that starts switched off (§3), so v4 begins training exactly where v3 is.

---

## 1. Lineage — where v4 comes from

```mermaid
%%{init: {"theme": "base", "flowchart": {"wrappingWidth": 400, "padding": 14}, "themeVariables": {"fontSize": "15px", "lineColor": "#64748B", "primaryTextColor": "#0F172A", "edgeLabelBackground": "#FFFFFF", "clusterBkg": "#F8FAFC", "clusterBorder": "#CBD5E1", "titleColor": "#0F172A"}}}%%
flowchart TB
    A["<b>ISBI 2026 Stage-1</b><br/>ConvNeXt-B + query decoder<br/>384 px, ImageNet init<br/>internal mAP 0.385"]:::old
    B["<b>ISBI 2026 Stage-2</b><br/>MoE in stages 4-5 + decoder FFN<br/>test mAP 0.4599, 3rd place"]:::old
    C["<b>Entropy.pdf framework</b><br/>triplet cross-memory bank<br/>+ anchor-based entropy selection"]:::old
    D["<b>train_2_v3.py</b> · current baseline<br/>triplet embedding fixed<br/>shard class weights, EMA + SWA<br/>768-1024 px · best 0.4602, SWA 0.4619"]:::v3
    E["<b>train_2_v4.py</b> · this experiment<br/>v3 + Markov label layer<br/>jobs mk1 / mk2 / mk1fix"]:::new
    A --> B
    A --> C
    C --> D
    D -->|"+ Markov layer after the logits"| E

    classDef v3 fill:#DBEAFE,stroke:#1D4ED8,stroke-width:1.5px,color:#1E3A8A
    classDef new fill:#FFEDD5,stroke:#C2410C,stroke-width:3px,color:#7C2D12
    classDef data fill:#FFFFFF,stroke:#64748B,stroke-width:1px,color:#0F172A
    classDef loss fill:#EDE9FE,stroke:#6D28D9,stroke-width:1.5px,color:#4C1D95
    classDef aux fill:#F1F5F9,stroke:#94A3B8,stroke-width:1px,stroke-dasharray:5 4,color:#475569
    classDef warn fill:#FEE2E2,stroke:#B91C1C,stroke-width:2px,stroke-dasharray:5 4,color:#7F1D1D
    classDef old fill:#F1F5F9,stroke:#94A3B8,stroke-width:1px,color:#334155
    classDef tail fill:#DCFCE7,stroke:#15803D,stroke-width:2.5px,color:#14532D
```

Stage-2 initialises from the Stage-1 checkpoint `train/checkpoint3/Model_20260119_062652` and
fine-tunes the whole network. v4 changes only what
happens **after** the classifier logits.

## 2. Full forward pass (shapes at 768 px, batch B)

```mermaid
%%{init: {"theme": "base", "flowchart": {"wrappingWidth": 400, "padding": 14}, "themeVariables": {"fontSize": "15px", "lineColor": "#64748B", "primaryTextColor": "#0F172A", "edgeLabelBackground": "#FFFFFF", "clusterBkg": "#F8FAFC", "clusterBorder": "#CBD5E1", "titleColor": "#0F172A"}}}%%
flowchart TB
    IMG(["<b>Chest X-ray</b><br/>B × 3 × 768 × 768"]):::data
    BB["<b>ConvNeXt2 backbone</b> · 5 stages, stride 64<br/>dims 128 → 256 → 512 → 1024 → 1024<br/>depths 3, 3, 27, 3, 2"]:::v3
    TOK(["<b>S = 144 image tokens</b><br/>12 × 12 feature map + 2-D sin/cos position<br/>B × 144 × 1024"]):::data
    IMG --> BB --> TOK

    subgraph DEC["ML-Decoder head"]
        MEM["<b>memory M</b><br/>Linear 1024 → 768 + ReLU"]:::v3
        Q["<b>30 label queries</b><br/>one per disease, frozen<br/>30 × 768"]:::v3
        subgraph CASA["decoder block × 2 · pre-LN · 8 heads"]
            CA["<b>① cross-attention</b><br/>each query looks at the image tokens"]:::v3
            SA["<b>② self-attention</b><br/>queries look at each other<br/>30 × 30 mixing, recomputed per image"]:::v3
            CA -->|"FFN"| SA
        end
        H(["<b>label embeddings h</b><br/>B × 30 × 768"]):::data
        GFC["<b>GroupFC</b><br/>one small classifier per disease"]:::v3
        MEM --> CA
        Q --> CA
        SA -->|"FFN"| H
        H --> GFC
    end

    TOK --> MEM
    GFC --> U(["<b>logits u</b> · B × 30"]):::data
    U --> MK["<b>Markov label layer</b> · NEW in v4<br/>z = u + g ⊙ (σ(z) P) + b<br/>starts as the identity: g = b = 0"]:::new
    MK --> Z(["<b>refined logits z</b> · B × 30"]):::data
    Z --> ASL["<b>Asymmetric loss</b><br/>class weights in shard order"]:::loss
    Z --> OUT(["probabilities → mAP"]):::data
    ASL --> LOSS["<b>total loss</b><br/>ASL + 0.1 × triplet"]:::loss

    H -.-> EMB["triplet embedding<br/>mean of the 30 queries, L2-norm"]:::aux
    EMB -.-> TRI["triplet loss, λ = 0.1<br/>XBM memory bank of 2048"]:::aux
    TRI -.-> LOSS

    style DEC fill:#EFF6FF,stroke:#1D4ED8,stroke-width:1px,color:#1E3A8A
    style CASA fill:#FFFFFF,stroke:#93C5FD,stroke-width:1px,stroke-dasharray:5 4,color:#1E3A8A

    classDef v3 fill:#DBEAFE,stroke:#1D4ED8,stroke-width:1.5px,color:#1E3A8A
    classDef new fill:#FFEDD5,stroke:#C2410C,stroke-width:3px,color:#7C2D12
    classDef data fill:#FFFFFF,stroke:#64748B,stroke-width:1px,color:#0F172A
    classDef loss fill:#EDE9FE,stroke:#6D28D9,stroke-width:1.5px,color:#4C1D95
    classDef aux fill:#F1F5F9,stroke:#94A3B8,stroke-width:1px,stroke-dasharray:5 4,color:#475569
    classDef warn fill:#FEE2E2,stroke:#B91C1C,stroke-width:2px,stroke-dasharray:5 4,color:#7F1D1D
    classDef old fill:#F1F5F9,stroke:#94A3B8,stroke-width:1px,color:#334155
    classDef tail fill:#DCFCE7,stroke:#15803D,stroke-width:2.5px,color:#14532D
```

- At 1024 px the feature map is 16×16, so there are S = 256 image tokens. Everything after the
  decoder keeps the same shape.
- Following ISBI2026.pdf §2.2, each block applies **cross-attention first** ("let these queries
  interact with the visual tokens via cross-attention first") and **then self-attention among the
  label queries**. That second step is where the decoder already models label co-occurrence.
- The triplet branch (dashed) reads `h` through `Decoder.last_h`. It shapes the embedding space only
  and never touches the logits, so it is unaffected by the Markov layer.

## 3. Inside the Markov layer

```mermaid
%%{init: {"theme": "base", "flowchart": {"wrappingWidth": 400, "padding": 14}, "themeVariables": {"fontSize": "15px", "lineColor": "#64748B", "primaryTextColor": "#0F172A", "edgeLabelBackground": "#FFFFFF", "clusterBkg": "#F8FAFC", "clusterBorder": "#CBD5E1", "titleColor": "#0F172A"}}}%%
flowchart TB
    U(["<b>u</b> · 30 logits from GroupFC"]):::data
    subgraph PAR["learned once · same for every image"]
        A[("<b>A</b><br/>30 × 30 weights<br/>init = log co-occurrence")]:::new
        P[("<b>P = row-softmax(A)</b><br/>diagonal masked<br/>P[i, j] ≈ P(j | i)<br/>each row sums to 1")]:::new
        A --> P
    end
    Q["<b>q = σ(u)</b><br/>30 label probabilities"]:::new
    M["<b>m = q P</b><br/>each label collects votes<br/>from the labels that co-occur with it"]:::new
    G["<b>z = u + g ⊙ m + b</b><br/>g: per-class gate, signed<br/>b: per-class bias<br/>both start at 0"]:::new
    Z(["<b>z</b> · refined logits"]):::data
    U --> Q --> M --> G --> Z
    P ==> M
    U -->|"original logits<br/>pass straight through"| G
    style PAR fill:#FFF7ED,stroke:#C2410C,stroke-width:1px,color:#7C2D12

    classDef v3 fill:#DBEAFE,stroke:#1D4ED8,stroke-width:1.5px,color:#1E3A8A
    classDef new fill:#FFEDD5,stroke:#C2410C,stroke-width:3px,color:#7C2D12
    classDef data fill:#FFFFFF,stroke:#64748B,stroke-width:1px,color:#0F172A
    classDef loss fill:#EDE9FE,stroke:#6D28D9,stroke-width:1.5px,color:#4C1D95
    classDef aux fill:#F1F5F9,stroke:#94A3B8,stroke-width:1px,stroke-dasharray:5 4,color:#475569
    classDef warn fill:#FEE2E2,stroke:#B91C1C,stroke-width:2px,stroke-dasharray:5 4,color:#7F1D1D
    classDef old fill:#F1F5F9,stroke:#94A3B8,stroke-width:1px,color:#334155
    classDef tail fill:#DCFCE7,stroke:#15803D,stroke-width:2.5px,color:#14532D
```

**Worked example (one hop, K = 1; the numbers are illustrative).** Suppose the decoder is confident an
image shows a central venous catheter: q_CVC = 0.9. §5 gives P[CVC, Support Devices] = 0.34, so
Support Devices collects a vote of 0.9 × 0.34 ≈ 0.31 from CVC, plus smaller votes from the other
labels. The gate decides what the vote does:

- at initialisation g = 0, so the Support Devices logit is unchanged;
- if training learns g_SD = +2, the logit rises by about 0.6;
- if training learns a negative g_SD, the same vote *lowers* it.

$$
P=\operatorname{softmax}_{\text{row}}\!\big(A \odot (1-I) - \infty\cdot I\big),\qquad
z^{(0)}=u,\qquad
z^{(k)} = u + g\odot\big(\sigma(z^{(k-1)})\,P\big) + b,\quad k=1..K
$$

- **Identity at initialisation.** With `g = b = 0`, `z = u`, so v4 starts exactly at the v3 model.
  The layer can only contribute what training finds useful. The gradient initially reaches only
  `g` (∂L/∂A ∝ g = 0), so `P` starts adapting once the gates move.
- **A true Markov matrix throughout training.** `P` is a row-softmax, so every row stays a
  probability distribution over "next" labels.
- **Signed gate.** A negative `g_j` lets co-occurring evidence *suppress* label j. That is the only
  way the layer can express exclusivity (see §5).
- **Separate optimiser group.** `A`, `g` and `b` train at 10× the base LR with no weight decay,
  because they start from zero. `--markov-fixed-transition` freezes `A` at the co-occurrence init.
- **Arms:** K = 1 (`mk1`), K = 2 (`mk2`), and K = 1 with `P` frozen (`mk1fix`).

## 4. ML-Decoder self-attention vs the Markov layer

Both are *row-stochastic 30×30 mixings over the labels*. They differ in where they act and in what
decides the weights.

```mermaid
%%{init: {"theme": "base", "flowchart": {"wrappingWidth": 400, "padding": 14}, "themeVariables": {"fontSize": "15px", "lineColor": "#64748B", "primaryTextColor": "#0F172A", "edgeLabelBackground": "#FFFFFF", "clusterBkg": "#F8FAFC", "clusterBorder": "#CBD5E1", "titleColor": "#0F172A"}}}%%
flowchart LR
    subgraph SAB["ML-Decoder self-attention"]
        direction TB
        I1(["<b>mixes</b><br/>30 query embeddings<br/>768-d each"]):::data
        W1["<b>with weights</b><br/>softmax(Q Kᵀ / √d)<br/><b>new for every image</b><br/>8 heads × 2 blocks"]:::v3
        O1["<b>where</b><br/>inside the decoder,<br/>before GroupFC"]:::v3
        I1 --> W1 --> O1
    end
    subgraph MKB["Markov label layer"]
        direction TB
        I2(["<b>mixes</b><br/>30 probabilities<br/>q = σ(u)"]):::data
        W2["<b>with weights</b><br/>P = row-softmax(A)<br/><b>one matrix for all images</b><br/>init from co-occurrence"]:::new
        O2["<b>where</b><br/>after the logits,<br/>z = u + g ⊙ (q P) + b"]:::new
        I2 --> W2 --> O2
    end
    SAB ~~~ MKB
    style SAB fill:#EFF6FF,stroke:#1D4ED8,stroke-width:1px,color:#1E3A8A
    style MKB fill:#FFF7ED,stroke:#C2410C,stroke-width:1px,color:#7C2D12

    classDef v3 fill:#DBEAFE,stroke:#1D4ED8,stroke-width:1.5px,color:#1E3A8A
    classDef new fill:#FFEDD5,stroke:#C2410C,stroke-width:3px,color:#7C2D12
    classDef data fill:#FFFFFF,stroke:#64748B,stroke-width:1px,color:#0F172A
    classDef loss fill:#EDE9FE,stroke:#6D28D9,stroke-width:1.5px,color:#4C1D95
    classDef aux fill:#F1F5F9,stroke:#94A3B8,stroke-width:1px,stroke-dasharray:5 4,color:#475569
    classDef warn fill:#FEE2E2,stroke:#B91C1C,stroke-width:2px,stroke-dasharray:5 4,color:#7F1D1D
    classDef old fill:#F1F5F9,stroke:#94A3B8,stroke-width:1px,color:#334155
    classDef tail fill:#DCFCE7,stroke:#15803D,stroke-width:2.5px,color:#14532D
```

| | ML-Decoder self-attention | Markov label layer |
|---|---|---|
| Where it acts | inside each decoder block, after cross-attention | after `GroupFC`, on the logits |
| Space | 768-d label-query embeddings | 30 label probabilities |
| Mixing matrix | `softmax(QKᵀ/√d)`, 30×30, row-stochastic | `P = softmax_row(A)`, 30×30, row-stochastic |
| Depends on the image? | **Yes**: recomputed for every image | **No**: one global matrix (its *input* `q` is image-specific) |
| How many | 8 heads × 2 blocks = 16 mixings | 1 matrix, applied K = 1 or 2 times |
| Initialisation | random projections, learned in Stage-1 | empirical co-occurrence P(j given i) |
| Interpretable? | Not directly (per-image, per-head attention maps) | Yes: `P[i, j]` reads as "given label i, how likely label j" |
| Can it express "A excludes B"? | Implicitly, through the learned embeddings | Not in `P` (all entries ≥ 0, rows sum to 1); **only via a negative gate** |
| Parameters | about 2 × 2.4 M for the attention projections | 30×30 + 30 + 30 = 960 |

**Takeaway.** The decoder already performs **image-conditioned** label-to-label message passing,
which is a Markov-like transition that changes per image. A fixed global `P` on top adds little new
information and blurs each image's own evidence. That is what the offline test showed (fixed smoothing:
−0.0006 to −0.0053 mAP; the winner's normal-gating: −0.024). The v4 layer is therefore trainable and
starts as the identity, and the experiment asks whether *any* global label prior helps once the
gates are free to learn it.

## 5. The label Markov chain in our training data

These are the strongest transitions in the initial `P`, built from label co-occurrence in
`train/CXRLT_2026_training_filtered.csv` (the file `train_2_v4.py` reads), N = 103,303 images. Edge label = `P(target | source)`; thick arrows mark P ≥ 0.20. Rare (tail) classes are green.

```mermaid
%%{init: {"theme": "base", "flowchart": {"wrappingWidth": 400, "padding": 14}, "themeVariables": {"fontSize": "15px", "lineColor": "#64748B", "primaryTextColor": "#0F172A", "edgeLabelBackground": "#FFFFFF", "clusterBkg": "#F8FAFC", "clusterBorder": "#CBD5E1", "titleColor": "#0F172A"}}}%%
flowchart TB
    HP(["<b>Hydropneumothorax</b><br/>tail · 38 images"]):::tail
    PTX(["<b>Pneumothorax</b><br/>tail · 420 images"]):::tail
    CVC(["central venous<br/>catheter"]):::data
    ST(["sternotomy"]):::data
    HER(["Hernia"]):::data
    NRM(["Normal<br/>40% of images"]):::aux
    PEF(["pleural effusion"]):::data
    SD(["Support Devices"]):::data
    AE(["aortic elongation"]):::data
    CM(["cardiomegaly"]):::data
    HP == "0.20" ==> PEF
    PTX -- "0.19" --> SD
    PEF -- "0.12" --> SD
    CVC == "0.34" ==> SD
    SD -- "0.21" --> CVC
    ST == "0.22" ==> CM
    HER -- "0.15" --> AE
    AE == "0.27" ==> CM
    CM -- "0.19" --> AE
    NRM -. "0.13, forced by row sums" .-> CM

    classDef v3 fill:#DBEAFE,stroke:#1D4ED8,stroke-width:1.5px,color:#1E3A8A
    classDef new fill:#FFEDD5,stroke:#C2410C,stroke-width:3px,color:#7C2D12
    classDef data fill:#FFFFFF,stroke:#64748B,stroke-width:1px,color:#0F172A
    classDef loss fill:#EDE9FE,stroke:#6D28D9,stroke-width:1.5px,color:#4C1D95
    classDef aux fill:#F1F5F9,stroke:#94A3B8,stroke-width:1px,stroke-dasharray:5 4,color:#475569
    classDef warn fill:#FEE2E2,stroke:#B91C1C,stroke-width:2px,stroke-dasharray:5 4,color:#7F1D1D
    classDef old fill:#F1F5F9,stroke:#94A3B8,stroke-width:1px,color:#334155
    classDef tail fill:#DCFCE7,stroke:#15803D,stroke-width:2.5px,color:#14532D
```

- **Tail diseases lean on head diseases.** This is challenge 1 in Entropy.pdf (slide 3): *"the tail
  disease often co-exist with head diseases"*. Hydropneumothorax (38 training images) mostly
  transitions to pleural effusion, and Pneumothorax to Support Devices. A global prior can pass head
  evidence to a tail class. It can equally produce false tail positives on every effusion
  case. That trade-off is what the gate `g` learns to balance.
- **Row normalisation forces Normal to transition somewhere.** Normal appears together with a finding
  in only 152 training images. Even so, its row in `P` must sum to 1, so it still "points" to
  cardiomegaly (0.13), Support Devices (0.10), and so on. `P` has no way to say "Normal excludes
  every finding". Only a negative gate on the finding classes can express that.

## 6. Training and evaluation flow for v4

```mermaid
%%{init: {"theme": "base", "flowchart": {"wrappingWidth": 400, "padding": 14}, "themeVariables": {"fontSize": "15px", "lineColor": "#64748B", "primaryTextColor": "#0F172A", "edgeLabelBackground": "#FFFFFF", "clusterBkg": "#F8FAFC", "clusterBorder": "#CBD5E1", "titleColor": "#0F172A"}}}%%
flowchart TB
    subgraph L1["1 · launch"]
        SUB["<b>submit_stage2_v4_markov.sh</b><br/>768 px, seed 42, dp 0.1<br/>4 of 8 cosine epochs, SWA of 3"]:::new
        JOB["<b>train_2_v4.sh</b><br/>Slurm job"]:::new
    end
    subgraph L2["2 · train"]
        TR["<b>train_2_v4.py</b><br/>ConvNeXt2Markov<br/>Stage-1 init, layer = identity"]:::new
        OPT["<b>AdamW, two groups</b><br/>model: lr 1e-4, wd 1e-2<br/>Markov: lr 1e-3, wd 0"]:::new
        EMA["<b>EMA weights</b><br/>evaluated each epoch"]:::v3
        MON["<b>mon.py --mk</b><br/>gate size per epoch<br/>Δ vs job 49000"]:::v3
        CK[("<b>model_best.pth</b><br/><b>model_swa.pth</b><br/>include label_refine.*")]:::new
    end
    subgraph L3["3 · evaluate"]
        EV["<b>evaluate_tta_v4.py</b><br/>detects the layer<br/>in the checkpoint"]:::new
        OLD["<b>evaluate_tta.py</b><br/>do NOT use:<br/>silently drops label_refine.*"]:::warn
        PR(["dumped probabilities"]):::data
        SUBS["<b>markov_check.py</b> val_subsets()<br/>all / frontal /<br/>frontal and unseen-patient"]:::v3
    end
    SUB --> JOB --> TR --> OPT --> EMA --> CK
    TR -.->|"log"| MON
    CK --> EV --> PR --> SUBS
    CK -.-x OLD
    style L1 fill:#F8FAFC,stroke:#CBD5E1,color:#0F172A
    style L2 fill:#F8FAFC,stroke:#CBD5E1,color:#0F172A
    style L3 fill:#F8FAFC,stroke:#CBD5E1,color:#0F172A

    classDef v3 fill:#DBEAFE,stroke:#1D4ED8,stroke-width:1.5px,color:#1E3A8A
    classDef new fill:#FFEDD5,stroke:#C2410C,stroke-width:3px,color:#7C2D12
    classDef data fill:#FFFFFF,stroke:#64748B,stroke-width:1px,color:#0F172A
    classDef loss fill:#EDE9FE,stroke:#6D28D9,stroke-width:1.5px,color:#4C1D95
    classDef aux fill:#F1F5F9,stroke:#94A3B8,stroke-width:1px,stroke-dasharray:5 4,color:#475569
    classDef warn fill:#FEE2E2,stroke:#B91C1C,stroke-width:2px,stroke-dasharray:5 4,color:#7F1D1D
    classDef old fill:#F1F5F9,stroke:#94A3B8,stroke-width:1px,color:#334155
    classDef tail fill:#DCFCE7,stroke:#15803D,stroke-width:2.5px,color:#14532D
```

`train_2_v3.py`, `convnext.py` and `evaluate_tta.py` are untouched, so existing checkpoints and runs
are unaffected. With `--markov-steps 0`, v4 is identical to v3.

## 7. The experiment

Every arm uses the push-wave recipe at 768 px, seed 42 (drop-path 0.1, 4 of 8 cosine epochs, SWA of 3
EMA epochs, triplet λ = 0.1). Each is matched to v3 job **49000**: best 0.4585 @ epoch 3, SWA 0.4600,
flip-TTA 0.4650, frontal & unseen-patient 0.4706.

| Arm | Markov steps K | `P` | Status (2026-09-26) |
|---|---|---|---|
| mk1 | 1 | learnable | smoke 49053 queued (per-user memory cap) |
| mk2 | 2 | learnable | smoke 49054 queued |
| mk1fix | 1 | frozen at co-occurrence | smoke 49055 queued |

**What to watch:**
- `|g|` in `mon.py --mk`: near 0 means training chose not to use the layer.
- Δ best / Δ SWA versus 49000.
- The frontal & unseen-patient subset, the closest local proxy for the leaderboard.
- Per-class AP on tail classes: does the prior help them, or create false positives?

## 8. Code map

| What | Where |
|---|---|
| Backbone `ConvNeXt2`, 5 stages /64 | `train/convnext.py:369` |
| Decoder head construction (30 queries, 2 blocks, GELU) | `train/convnext.py:436` |
| 2D positional encoding | `train/convnext.py:445` |
| `ConvNeXt2.forward` returns (logits, memory) | `train/convnext.py:475` |
| `Decoder` (ML-Decoder head) | `train/ml_decoder.py:301` |
| Memory projection `embed_standart` | `train/ml_decoder.py:321` |
| Frozen label queries | `train/ml_decoder.py:325` |
| CA → FFN → SA → FFN block | `train/ml_decoder.py:208` |
| `last_h` exposed for the triplet embedding | `train/ml_decoder.py:394` |
| `GroupFC` per-class output | `train/ml_decoder.py:86` |
| Co-occurrence transition init | `train/markov_layer.py:35` |
| `MarkovLabelRefiner` / `transition()` / `forward()` | `train/markov_layer.py:43` / `:60` / `:64` |
| `ConvNeXt2Markov` | `train/markov_layer.py:84` |
| Checkpoint-aware loader | `train/markov_layer.py:99` |
| v4 `create_model` | `train/train_2_v4.py:442` |
| Markov optimiser group | `train/train_2_v4.py:701` |
| Triplet embedding / mining / total loss | `train/train_2_v4.py:866` / `:871` / `:888` |
| Per-epoch gate log | `train/train_2_v4.py:1012` |
| `--markov-steps` CLI | `train/train_2_v4.py:1200` |
| `P` built from training labels | `train/train_2_v4.py:1436` |
| v4 evaluation wrapper | `evaluate/evaluate_tta_v4.py:18` |
| Offline checks (smoothing, normal-gating, subsets) | `analysis/markov_check.py:68` / `:114` / `:188` |

## Sources

- `docs/ISBI2026.pdf`: §2.2 model architecture (query decoder replacing global average pooling;
  cross-attention before self-attention); §2.4–2.5 two-stage training, MoE, ASL; Table 3 (test mAP 0.4599).
- `docs/Entropy.pdf`: slide 3 (challenges: label co-occurrence, imbalance); slides 5–7 (triplet
  cross-memory bank, anchor-based entropy selection); slide 9 (384 vs 512 px internal results).
- `docs/result/2026-09-26-markov-applicability.md`: offline smoothing and normal-gating results;
  CXR-LT 2026 test-set facts.
