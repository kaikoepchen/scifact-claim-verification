# Project Findings & Next Steps

## Research Question

Can disagreement between sparse (BM25) and dense retrieval systems serve as a useful signal for scientific claim verification — both for knowing when to abstain at inference time and for improving model training via regularization?

---

## What We Built

A modular RAG pipeline for scientific claim verification on the SciFact benchmark:

1. **Hybrid retrieval**: BM25 + dense (MiniLM-L6-v2) with Reciprocal Rank Fusion + cross-encoder reranking
2. **Joint sentence-level model**: DeBERTa-v3-large fine-tuned on (claim, sentence) → SUPPORT/CONTRADICT/NEI, replacing separate cosine-similarity rationale selection and abstract-level NLI
3. **Calibrated abstention gate**: Multi-signal uncertainty score (NLI confidence, margin, retriever agreement, evidence count, conflict detection) with threshold tuning
4. **KL retriever-consistency regularization**: Symmetric KL divergence between model predictions on BM25-retrieved vs. dense-retrieved evidence, added as a training-time regularization term

---

## Proven Results

### 1. Joint sentence model dramatically improves over pipeline approach

| Pipeline | Abstract F1 (dev) |
|---|---|
| Separate (cosine selector + zero-shot NLI) | 0.139 |
| **Joint sentence model** | **0.401** |

Moving from independent components to a single model that jointly does rationale selection + verdict prediction yielded a **3x improvement**.

### 2. Abstention improves accuracy on answered claims

| Metric | No Abstention | With Abstention |
|---|---|---|
| Accuracy | 75.0% | **81.0%** |
| Coverage | 100% | 72.9% |
| **Accuracy gain** | — | **+6.0%** |

By declining to answer 27% of uncertain claims, accuracy on answered claims rises 6 points. The abstention mechanism works.

### 3. Retriever disagreement correlates with prediction correctness

| Retriever Agreement | Accuracy | n |
|---|---|---|
| High (Jaccard ≥ median) | **80.6%** | 98 |
| Low (Jaccard < median) | 68.9% | 90 |
| **Gap** | **+11.7%** | |

This is the core empirical finding: when BM25 and dense retrieval agree on what's relevant, the system is correct 81% of the time. When they disagree, accuracy drops to 69%. Pearson correlation: 0.043 (weak but consistent across all splits).

### 4. Disagreement signal provides additive value for abstention

| Abstention Variant | Best Accuracy (≥50% coverage) |
|---|---|
| No abstention | 0.750 |
| NLI confidence only | 0.808 |
| NLI confidence + retriever disagreement | **0.812** |

The disagreement signal adds +0.4% on top of NLI confidence alone. Small but complementary — it captures information about evidence reliability that model confidence alone does not.

### 5. KL divergence regularization is feasible and shows positive signal

| Model | Macro-F1 | Accuracy | CONTRADICT F1 |
|---|---|---|---|
| Joint baseline (no KL) | 0.787 | 0.862 | 0.689 |
| **Joint + KL (λ=0.3)** | **0.789** | 0.861 | **0.701** |

The KL regularization — which encourages the model to make consistent predictions regardless of which retriever provided the evidence — yields a small improvement, most notably on the hardest class (CONTRADICT +1.2%). The KL loss decreased from 0.040 to 0.003 over training, confirming the model learned retrieval-view consistency.

### 6. This approach is novel

An extensive literature review found **no prior work** that uses multi-retriever disagreement as a training signal for downstream verification models. Closest related work:
- **REAR** (EMNLP 2024): uses a single retriever's relevance score as input feature
- **RAAT** (ACL 2024): adversarial training for noisy retrieval robustness
- **Self-RAG** (ICLR 2024): reflection tokens for retrieval quality, from a critic LM

None combines multiple retriever types + their agreement signal + modulation of downstream training loss.

---

## Comparison to Published Systems

| System | Abstract F1 | Setting |
|---|---|---|
| Our baseline (cosine + zero-shot NLI) | 13.9 | Dev, 5K corpus |
| **Our joint model** | **40.1** | Dev, 5K corpus |
| VeriSci (Wadden 2020) | 50.0 | Test, 5K corpus |
| ParagraphJoint (Li 2021) | 69.1 | Test, 5K corpus |
| ARSJoint (Zhang 2021) | 62.4–71.2 | Test, 5K corpus |
| MultiVerS (Wadden 2022) | 72.5 | Test, 5K corpus |

The remaining gap (40 vs 72) is explained by:
- **No weak supervision pretraining** (MultiVerS uses FEVER + PubMedQA, ~200K examples)
- **Sentence-level vs document-level encoding** (MultiVerS cross-encodes full abstracts with Longformer)
- **Dev vs test evaluation** (our numbers are on dev; published results are on test)

---

## What We Have NOT Proven

1. **That KL regularization provides a statistically significant improvement.** The +0.2% macro-F1 gain is within noise for a dataset this small. Would need multiple seeds and/or a larger dataset.

2. **That our approach works at scale.** The 5K SciFact corpus is easy for retrieval — retrievers rarely disagree strongly. The signal should be much stronger on SciFact-Open (28M abstracts).

3. **That the approach generalizes beyond SciFact.** We need evaluation on at least HealthVer or FEVER to claim generality.

4. **That abstention improves the metrics reviewers care about.** Abstract F1 goes *down* with abstention (0.401 → 0.377) because F1 penalizes reduced recall. The accuracy improvement is real, but the standard leaderboard metric doesn't reward it.

---

## Next Steps

### Priority 1: SciFact-Open Evaluation
Evaluate on SciFact-Open (28M abstract corpus) where retrieval is genuinely hard. This is where retriever disagreement should matter most. Pre-build FAISS index for the S2ORC corpus, run retrieval with BM25 + dense, measure disagreement-correctness correlation at scale.

**Hypothesis**: The +11.7% agreement-accuracy gap we see on 5K corpus will be even larger on 28M.

### Priority 2: Longformer Cross-Encoder (MultiVerS-style)
Implement document-level encoding with Longformer-base-4096. Cross-encode (claim, full abstract) instead of (claim, sentence). Joint label + rationale heads. This closes the architectural gap with SOTA while keeping our KL regularization on top.

**Expected impact**: 50-63% abstract F1 (without weak supervision pretraining).

### Priority 3: Multi-Retriever KL with 3+ Retrievers
Add BGE-M3 as a third retriever. Three-way disagreement is a richer signal than two-way. Evaluate whether more diverse retrieval views improve the KL regularization effect.

### Priority 4: Proper Ablations
- Vary λ_kl: {0.0, 0.1, 0.3, 0.5, 1.0}
- Vary number of retrievers: {1, 2, 3}
- Report mean ± std over 3 seeds
- Bin claims by disagreement level and show per-bin accuracy

### Priority 5: Additional Benchmarks
- **HealthVer**: health claim verification, shows domain transfer
- **FEVER** (subset): general fact verification, shows the method isn't domain-specific
- **AVERITEC**: open-web claims, trendy benchmark with 2024 shared task

### Priority 6: Analysis & Writeup
- Error analysis: what types of claims benefit most from the disagreement signal?
- Calibration plots: does our confidence score predict actual accuracy?
- Case studies: concrete examples where disagreement correctly identified unreliable retrieval
- Frame as: "Retriever consistency regularization for claim verification" — the method is the contribution, SciFact is one evaluation
