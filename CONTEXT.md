# Project Context: Active Scientific Claim Verification with RAG

> University of Zurich -- Retrieval-Augmented Generation Course, FS 2026
> Group Project -- 4 members

---

## 1. Project Overview

### What we are building
A **scientific claim verification RAG system**. Given a scientific claim (e.g. "Aspirin reduces the risk of colorectal cancer"), the system:

1. Retrieves relevant research abstracts from a corpus
2. Selects the most relevant rationale sentences from those abstracts
3. Predicts a verdict: **SUPPORT** / **CONTRADICT** / **NOT ENOUGH EVIDENCE**
4. Generates a short (2-4 sentence) grounded explanation with citations to the retrieved evidence
5. Optionally abstains when evidence is weak, conflicting, or unreliable

### Main hypothesis
A hybrid evidence pipeline (BM25 + dense retrieval + RRF fusion + abstention calibration) outperforms single-retriever baselines on both retrieval quality (nDCG@10) and verification accuracy (macro-F1), while uncertainty-driven annotation extends the training set more efficiently than random sampling.

### What this is NOT
- Not "judge whether an entire paper is truthful"
- Not a generic research-paper chatbot
- Not a pure reasoning benchmark
- Not a pure LLM fine-tuning project

### Why this project?
- Scientific misinformation is a concrete, important problem
- Claim-level verification with traceable evidence is tractable and useful
- The project is **inherently retrieval-centered** (retrieval quality directly limits verdict quality)
- **Benchmark-comparable** (SciFact is a standard task with published baselines)
- **Naturally extensible** (active learning, abstention, preference optimization build cleanly on the core)

---

## 2. System Architecture

### Online inference path
```
Claim Input
    |
Normalize / Atomic Claim Split
    |
    +---> BM25 (sparse retrieval, Assignment 1 reuse)
    +---> Dense Retriever (SPECTER + FAISS)
    +---> Citation Priors (optional)
    |
Hybrid Fusion (Reciprocal Rank Fusion)
    + optional cross-encoder reranker
    |
Evidence Pack (top-k abstracts + rationale sentences)
    |
    +---> Verdict Layer (NLI: SUPPORT / CONTRADICT / NEI)
    +---> Generator (short cited explanation)
    |
Abstention / Calibration Gate
    |
Final Output: verdict + explanation + source sentences
```

### Offline data and training loop
```
Output --> Uncertainty Queue --> Human Annotation --> Hard Negative Mining --> DPO/SimPO
                                     |                       |                    |
                                     v                       v                    v
                               Updated corpus         Dense retriever       Generator
                               + labels               fine-tuning           preference opt.
```

---

## 3. Component Details

### 3.1 Retrieval Stack

#### BM25 (Sparse Retrieval)
- **Status:** REUSE (Assignment 1)
- **Role:** Primary sparse baseline
- **Implementation:** Inverted index on SciFact 5,183-abstract corpus
- **Benchmark:** BEIR SciFact nDCG@10 = 0.665
- **Why:** BM25 is one of the most consistently strong zero-shot retrievers across BEIR's 18 datasets (Thakur et al., 2021). Scientifically non-negotiable as a baseline.

#### Dense Retrieval (SPECTER + FAISS)
- **Status:** REUSE (standard DPR setup)
- **Role:** Semantic bi-encoder retriever
- **Implementation:** SPECTER (Cohan et al., 2020) pre-trained on scientific citation pairs. FAISS flat or IVF index. Off-the-shelf first, fine-tune later via hard-negative mining.
- **Why:** Handles vocabulary mismatch where exact terms differ but meaning is the same. SPECTER is domain-adapted to scientific abstracts.

#### Hybrid Fusion (RRF)
- **Status:** REUSE (established method)
- **Formula:** `RRF(d) = 1/(k + r_BM25(d)) + 1/(k + r_dense(d))`, k=60
- **Core experiment:** Three-way ablation: BM25 alone vs. dense alone vs. RRF hybrid
- **Why:** Parameter-free, consistently competitive (Cormack et al., 2009)

#### Cross-encoder Reranker (Optional)
- **Status:** EXTENSION
- **Role:** Second-stage reranking of top-50 candidates
- **Implementation:** e.g. ms-marco-MiniLM-L-6-v2
- **Why:** Adds quality at compute cost. Only after core retrieval works.

### 3.2 Evidence Selection

#### Rationale Sentence Selection
- **Status:** NEW
- **Role:** Select 1-3 most relevant sentences per retrieved abstract
- **Implementation:** Cosine similarity between claim embedding and sentence embeddings, or cross-encoder sentence scorer
- **Why:** Full abstracts contain irrelevant content. Standard in SciFact pipeline (Wadden et al., 2020) and CliVER (Liu et al., 2024).

### 3.3 Verdict Layer

- **Status:** NEW
- **Three options (priority order):**
  1. NLI model (DeBERTa-v3 fine-tuned on MultiNLI) -- zero-shot, maps entailment/contradiction/neutral
  2. Prompted LLM with chain-of-thought reasoning
  3. Fine-tuned classifier on SciFact labels
- **Metric:** macro-F1 on SciFact dev/test
- **Compare to:** Wadden et al. (2020), CliVER (Liu et al., 2024)

### 3.4 Reasoning as a Layer
- **Status:** NEW (but subordinate to retrieval)
- **Claim decomposition:** Split conjunctive claims into atomic sub-claims before retrieval
- **Multi-evidence aggregation:** Aggregate verdicts when multiple documents offer partial support
- **Key constraint:** Retrieval must remain central; this is not a pure reasoning project

### 3.5 Abstention / Calibration
- **Status:** NEW (genuine novel contribution)
- **Signals for abstention:**
  - Top-1 retrieval score margin (top-1 vs top-2 gap)
  - BM25/dense ranking disagreement (Jaccard overlap of top-10)
  - Evidence conflict (both SUPPORT and CONTRADICT signals present)
  - No document above minimum similarity threshold
- **Key metric:** Coverage-vs-risk curve
- **Inspired by:** Self-RAG (Asai et al., 2024)

### 3.6 Generator
- **Status:** NEW (grounded citations)
- **Role:** Produce 2-4 sentence explanation citing specific rationale sentences
- **Constraint:** Cite at most 3 rationale sentences
- **Failure modes to guard against:** Hallucination, uncited statements, wrong verdict in explanation
- **Metrics:** Citation support rate, unsupported sentence rate (ALCE-style, Gao et al., 2023)
- **Approach:** First test prompted generation (frozen LLM), add supervised fine-tuning only if format is broken

### 3.7 Preference Optimization (DPO / SimPO)
- **Status:** NEW (extension)
- **Preference pairs:**
  - Grounded vs. hallucinated explanation
  - Correct abstention vs. overconfident wrong verdict
  - High vs. low citation precision
- **DPO:** Rafailov et al. (NeurIPS 2023) -- no separate reward model
- **SimPO:** Meng et al. (2024) -- also removes reference model, halving memory
- **Do NOT use PPO** -- impractical at course-project scale

### 3.8 LoRA / QLoRA
- **Status:** EXTENSION (optional, generator-only)
- **Where LoRA helps:** Generator supervised format tuning, DPO/SimPO training, domain adaptation
- **Where LoRA does NOT help:** BM25 (non-parametric), dense retriever embeddings, NLI verdict layer
- **Key decision:** Do not build the project around LoRA. Apply only after retrieval + verdict pipeline works.

### 3.9 Active Data Accumulation Loop
- **Status:** NEW
- **Claim acquisition:** LLM-assisted extraction from abstract conclusion sentences + human filtering
- **Target:** 100-200 high-quality new claims (not thousands)
- **Uncertainty-driven annotation:**
  1. Retrieve evidence for new claims
  2. Produce weak verdict
  3. Score uncertainty (entropy, BM25/dense disagreement)
  4. Send highest-uncertainty cases to human annotation
  5. Retrain on corrected examples
- **Hard-negative mining:** Documents ranked high by BM25 but marked non-supporting by verdict = informative hard negatives for dense retriever

---

## 4. Benchmark and Data

### 4.1 Primary Benchmark: SciFact

| Property | Value |
|----------|-------|
| Paper | Wadden et al., "Fact or Fiction: Verifying Scientific Claims" (EMNLP 2020) |
| Total claims | 1,409 |
| Train split | 809 claims |
| Dev split | 300 claims |
| Test split | 300 claims (labels hidden) |
| Corpus | 5,183 research abstracts |
| Labels | SUPPORTS (39.5%), REFUTES (23.9%), NOINFO (36.6%) |
| Format | JSONL: claim text, evidence doc IDs, evidence labels, rationale sentence indices |
| Download | [github.com/allenai/scifact](https://github.com/allenai/scifact) |
| HuggingFace | [huggingface.co/datasets/allenai/scifact](https://huggingface.co/datasets/allenai/scifact) (~6 MB) |
| Leaderboard | [leaderboard.allenai.org/scifact](https://leaderboard.allenai.org/scifact) |

### 4.2 SciFact-Open (Optional Extension)

| Property | Value |
|----------|-------|
| Paper | Wadden et al., "SciFact-Open" (Findings of EMNLP 2022) |
| Corpus | ~500K abstracts (from S2ORC) |
| Claims | Same as SciFact |
| Key finding | Systems drop at least 15 F1 points vs. small-corpus setting |
| Download | [github.com/dwadden/scifact-open](https://github.com/dwadden/scifact-open) |

### 4.3 Published Retrieval Baselines (BEIR SciFact, nDCG@10)

| Retriever | nDCG@10 | Notes |
|-----------|---------|-------|
| BM25 | 0.665 | Our A1 baseline; target to match |
| docT5query | 0.675 | Query expansion |
| ColBERT | 0.671 | Late interaction model |
| Contriever | 0.677 | Unsupervised dense |
| uniCOIL | 0.686 | Learned sparse |
| BM25 + CE reranker | 0.688 | BM25 + cross-encoder |
| SPLADE | 0.699 | Sparse-dense hybrid |
| E5-base (pretrained) | **0.737** | Best published; contrastive pretrained |

**Our target:** RRF hybrid should aim for 0.68-0.72 range. Beating BM25 alone (0.665) is the minimum bar.

### 4.4 Published End-to-End Verification Baselines (SciFact test)

| System | Abstract-level F1 | Sentence-level F1 | Notes |
|--------|-------------------|-------------------|-------|
| VeriSci (open retrieval) | 47.4 | 39.5 | Original baseline; retrieval is bottleneck |
| VeriSci (oracle abstracts) | 72.7 | 60.6 | Gold evidence; shows ceiling |
| ParagraphJoint | 69.1 | 60.9 | Fully supervised |
| Vert5erini | 68.2 | 63.4 | T5-based |
| ARSJoint | 71.2 | -- | Leaderboard runner-up |
| **MultiVerS** | **72.5** | **67.2** | Current leaderboard leader |

**Key insight:** VeriSci jumps from 47 to 73 F1 with oracle abstracts. Retrieval quality is the main bottleneck -- exactly the lever our project focuses on.

### 4.5 All Data Sources

| Dataset | Size | Role in our project | Source |
|---------|------|---------------------|--------|
| **SciFact** | 1,409 claims / 5,183 abstracts | PRIMARY: main evaluation benchmark | [GitHub](https://github.com/allenai/scifact) / [HF](https://huggingface.co/datasets/allenai/scifact) |
| **SciFact-Open** | ~500K abstracts | OPTIONAL: open-domain stress test | [GitHub](https://github.com/dwadden/scifact-open) |
| **BEIR SciFact** | Same corpus, standardized format | PRIMARY: retrieval-only comparison with published nDCG@10 | [GitHub](https://github.com/beir-cellar/beir) |
| **MultiNLI** | 433K sentence pairs | PRIMARY: NLI pre-training for verdict layer (entailment/contradiction/neutral) | [HF](https://huggingface.co/datasets/nyu-mll/multi_nli) |
| **SNLI** | 570K sentence pairs | PRIMARY: combined with MultiNLI (~1M total) for robust NLI | [HF](https://huggingface.co/datasets/stanfordnlp/snli) |
| **FEVER** | 185K claims (Wikipedia) | TRANSFER: same task structure; VeriSci used FEVER pre-training | [HF](https://huggingface.co/datasets/fever/fever) / [fever.ai](https://fever.ai) |
| **S2ORC** | 81M papers / 8.1M full text | OPTIONAL: large-scale corpus; source for claim extraction | [GitHub](https://github.com/allenai/s2orc) |
| **PubMed** | 24.6M abstracts | OPTIONAL: biomedical corpus for retrieval or claim extraction | [pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov/) |
| **HealthVer** | 14,330 pairs | OPTIONAL: health claim verification (COVID-19 focus) | [ACL Anthology](https://aclanthology.org/2021.findings-emnlp.297/) |
| **COVID-Fact** | 4,086 claims | OPTIONAL: COVID claims from Reddit vs. CORD-19 | [GitHub](https://github.com/asaakyan/covidfact) |
| **Our extension set** | 100-200 claims (self-curated) | NEW: LLM-extracted, human-filtered, uncertainty-annotated | Self-collected |

### 4.6 Data Flow: Which Data Goes Where

- **Retrieval index:** SciFact 5,183 abstracts (core) or SciFact-Open 500K (optional)
- **Retrieval evaluation:** BEIR SciFact standardized queries and relevance judgments
- **Verdict layer pre-training:** MultiNLI + SNLI (~1M NLI pairs); optionally FEVER (185K claims)
- **Verdict layer fine-tuning:** SciFact train split (809 claims with labels + rationale indices)
- **Generator training:** SciFact dev split for preference pair construction; LoRA/QLoRA on frozen LLM
- **Active loop input:** New claims from S2ORC or PubMed abstracts, filtered by humans
- **Hard-negative mining:** Generated from pipeline output during the active loop

---

## 5. Comparison Strategy

### Where we compare to published external baselines
| What | Compared against | Metric |
|------|------------------|--------|
| Retrieval quality | BEIR SciFact (BM25, SPLADE, Contriever, E5) | nDCG@10, Recall@k, MRR |
| End-to-end verification | Wadden et al. (2020), CliVER (Liu et al., 2024), MultiVerS | macro-F1 on SciFact test |

### Where we compare only against our own ablations
| What | Ablation | Metric |
|------|----------|--------|
| Retrieval variant | BM25 vs. dense vs. RRF hybrid | nDCG@10, Recall@5/10 |
| Abstention trade-off | With vs. without abstention gate | Coverage vs. error rate curve |
| Active loop efficiency | Uncertainty sampling vs. random | F1 gain per annotation |
| Citation quality | Before vs. after DPO/SimPO | Citation support rate, unsupported sentence rate |

**We do NOT claim to beat the best published systems.** The goal is measurable improvement from each pipeline component.

---

## 6. What Is New vs. Reused

### Reused / Standard
- SciFact benchmark
- BM25 retriever (Assignment 1)
- Dense bi-encoder + FAISS (standard DPR setup)
- Hybrid fusion (RRF)
- Prompted generator baseline (Assignment 2 setup)

### Genuinely New Contributions
- Rationale sentence selection from retrieved abstracts
- Three-class NLI verdict layer
- Claim decomposition + multi-evidence aggregation (reasoning as a layer on RAG)
- Abstention / calibration gate using retriever disagreement + score margins + evidence conflict
- Citation-faithful short explanation with unsupported-sentence rate metric
- Active claim-evidence data flywheel with uncertainty-driven annotation
- Hard-negative mining from pipeline output to improve dense retriever
- DPO or SimPO for grounded explanation preference optimization

### Optional Extensions
- Cross-encoder reranker (adds quality at compute cost)
- LoRA/QLoRA (generator-only adaptation)
- SciFact-Open (500K-abstract scalability stress test)

---

## 7. Evaluation Plan (Four Layers)

### Layer 1 -- Retrieval Quality
- **Metrics:** Recall@5, Recall@10, nDCG@10, MRR
- **Ablation:** BM25 vs. dense vs. RRF (three-way)
- **Compare to:** BEIR SciFact published numbers

### Layer 2 -- Verification Accuracy
- **Metrics:** Accuracy, macro-F1 over 3 classes, confusion matrix
- **Compare to:** Wadden et al. (2020), CliVER (Liu et al., 2024)
- **Key question:** Does better retrieval propagate into better verdict accuracy?

### Layer 3 -- Grounding and Citation Quality
- **Metrics:** Citation support rate, unsupported sentence rate, evidence coverage
- **Inspired by:** ALCE (Gao et al., 2023)

### Layer 4 -- Abstention Trade-off
- **Metrics:** Coverage (fraction answered), error rate among answered claims, area under coverage-risk curve

---

## 8. Implementation Roadmap

| Phase | Deliverable | Priority | Done when |
|-------|-------------|----------|-----------|
| 1 | SciFact corpus setup, BM25 baseline, retrieval metrics | **CORE** | nDCG@10 matches BEIR number |
| 2 | SPECTER dense retriever + FAISS, RRF fusion, ablation table | **CORE** | 3-way retrieval ablation complete |
| 3 | Rationale selector, NLI verdict layer, macro-F1 on dev | **CORE** | Verdict F1 reported |
| 4 | Grounded generation, citation metrics (ALCE-style) | **CORE** | Citation support rate measured |
| 5 | Abstention gate, coverage-risk curve | **CORE** | Curve improves on random threshold |
| 6 | Active loop: claim extraction, uncertainty queue | Extension | 100+ new claims annotated |
| 7 | Hard-negative mining, dense retriever fine-tuning | Extension | Retrieval nDCG improves |
| 8 | DPO/SimPO preference pairs, generator fine-tuning | Extension | Citation precision improves |
| 9 | Cross-encoder reranker, LoRA, SciFact-Open | Stretch | Only if time allows |

**The project is complete and defensible at the end of Phase 5.** Phases 6-8 are bonus. Phase 9 is ambitious stretch.

---

## 9. Team Split

| Student | Lead Role | Responsibilities |
|---------|-----------|------------------|
| Student 1 | Retrieval lead | BM25 pipeline, BEIR evaluation, RRF fusion, retrieval ablation table, SciFact corpus setup |
| Student 2 | Dense retrieval | SPECTER + FAISS index, optional cross-encoder reranker, hard-negative mining, retriever fine-tuning |
| Student 3 | Evidence + verdict | Rationale sentence selector, NLI verdict layer, claim decomposition, multi-evidence aggregation, evaluation pipeline |
| Student 4 | Generation + loop | Grounded explanation generator, citation metrics, abstention module, uncertainty queue, active data loop, demo interface |
| All 4 | Shared | Final evaluation table, report writing, DPO/SimPO (if pursued) |

---

## 10. Risks and Fallbacks

| Risk | Fallback |
|------|----------|
| Verdict layer underperforms despite correct retrieval | Reduce to retrieval-focused project with majority-vote verdict. Retrieval ablation story is independently strong. |
| Active loop produces noisy labels | Report loop design and annotation-efficiency on small held-out set without full retraining. |
| DPO/SimPO training is unstable | Report prompted generation only. Preference optimization is optional; removing it doesn't break core. |
| SciFact corpus too small to show retrieval differences | Extend to first 100K abstracts from SciFact-Open, or report at k=5 where differences are more visible. |

---

## 11. Key References

| Short name | Full reference | Used for |
|------------|----------------|----------|
| SciFact | Wadden et al., "Fact or Fiction: Verifying Scientific Claims" (EMNLP 2020) | Primary benchmark |
| SciFact-Open | Wadden et al., "SciFact-Open" (Findings of EMNLP 2022) | Optional large-corpus extension |
| BEIR | Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models" (NeurIPS 2021) | Retrieval comparison |
| DPR | Karpukhin et al., "Dense Passage Retrieval for Open-Domain QA" (EMNLP 2020) | Dense retrieval architecture |
| SPECTER | Cohan et al., "SPECTER: Document-level Representation Learning using Citation-informed Transformers" (ACL 2020) | Scientific document embeddings |
| RRF | Cormack et al., "Reciprocal Rank Fusion" (SIGIR 2009) | Hybrid fusion method |
| BERT reranker | Nogueira and Cho, "Passage Re-ranking with BERT" (2019) | Cross-encoder reranking |
| CliVER | Liu et al., "Retrieval Augmented Scientific Claim Verification" (JAMIA Open 2024) | Pipeline comparison |
| Self-RAG | Asai et al., "Self-RAG: Learning to Retrieve, Generate, and Critique" (ICLR 2024) | Abstention inspiration |
| ALCE | Gao et al., "Enabling LLMs to Generate Text with Citations" (2023) | Citation evaluation framework |
| DPO | Rafailov et al., "Direct Preference Optimization" (NeurIPS 2023) | Preference optimization |
| SimPO | Meng et al., "SimPO: Simple Preference Optimization" (2024) | Reference-free preference opt. |
| LoRA | Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022) | Efficient fine-tuning |
| QLoRA | Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (NeurIPS 2023) | 4-bit quantized adaptation |
| Hard negatives | Zhan et al., "Optimizing Dense Retrieval Model Training with Hard Negatives" (2021) | Retriever improvement |
| Active learning | Zeng et al., "Active Data Annotation Prioritization for Low-Resource Claim Verification" (Findings of EACL 2023) | Uncertainty sampling |
| Sci IE | Dagdelen et al., "Structured Information Extraction from Scientific Text with LLMs" (Nature Comms 2024) | Claim extraction |

---

## 12. Project Files

| File | Description |
|------|-------------|
| `scientific_validation_project.tex` | Full LaTeX project proposal with TikZ diagrams, all sections, citations |
| `executive_summary.html` | Standalone HTML executive summary with pipeline visuals, benchmark tables, data sources |
| `diagram.html` | Interactive clickable system architecture diagram (open in browser) |
| `project_advisor_report.tex` | Earlier advisor report draft |
| `CONTEXT.md` | This file -- complete project context |

---

## 13. Course Fit

This project is designed for a **RAG course**. Retrieval must stay central and independently measurable:

- **Sparse retrieval / BM25 / inverted index** -- core baseline
- **Dense retrieval / FAISS / ANN** -- core component
- **Hybrid retrieval** -- core experiment (RRF ablation)
- **IR evaluation** (Recall, Precision, MRR, nDCG) -- standard metrics throughout
- **Evidence-grounded generation** -- constrained, citation-faithful
- **Hallucination / faithfulness / citation support** -- measured explicitly
- **Backend/frontend/testing style system design** -- demo interface planned
- **Reasoning over retrieved evidence** -- but only as a layer ON TOP of RAG, not replacing it

The project is NOT a pure reasoning project, NOT a pure LLM fine-tuning project. Retrieval quality is the primary bottleneck and the primary research lever.
