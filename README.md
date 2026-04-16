# SciFact Claim Verification

A scientific claim verification system built on hybrid retrieval-augmented generation. Given a claim like *"Aspirin reduces the risk of colorectal cancer"*, the system retrieves relevant research abstracts, selects rationale sentences, predicts a verdict (SUPPORT / CONTRADICT / NOT ENOUGH EVIDENCE), and generates a short cited explanation.

**Research angle:** We use the disagreement between sparse (BM25) and dense retrieval as a first-class uncertainty signal for calibrated abstention. When the two retrievers disagree on what's relevant, the system knows it should be less confident.

Built for the RAG course at the University of Zurich, FS 2026.

## Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="Pipeline Architecture" width="750"/>
</p>

## Results

### Retrieval Ablation (SciFact Dev, 300 queries)

| Condition | nDCG@10 | Recall@5 | Recall@10 | MRR |
|-----------|---------|----------|-----------|-----|
| BM25 only | 0.830 | 0.878 | 0.941 | 0.797 |
| Dense only (MiniLM-L6) | 0.774 | 0.869 | 0.912 | 0.731 |
| **Hybrid (RRF)** | **0.850** | **0.905** | **0.950** | **0.824** |

### Retriever Disagreement Analysis (SciFact Dev, 188 queries with relevance judgments)

| Metric | Value |
|--------|-------|
| Mean Jaccard@10 (BM25 vs. dense top-10 overlap) | 0.224 |
| Top-1 agreement rate | 51.3% |
| Retrieval success — high agreement group | 99.0% |
| Retrieval success — high disagreement group | 94.4% |
| **Success gap (agreement - disagreement)** | **+4.5%** |
| Pearson(Jaccard, retrieval success) | 0.135 |

When retrievers agree, retrieval almost always succeeds (99%). When they disagree, success drops and 11% of cases have only one retriever finding the relevant document.

### Joint Sentence-Level Model (SciFact Dev)

We replace the separate cosine-similarity rationale selector and abstract-level NLI predictor with a single DeBERTa-v3-large model fine-tuned on SciFact sentence-level annotations. Each (claim, sentence) pair is classified as SUPPORT / CONTRADICT / NEI, jointly performing rationale selection and verdict prediction.

**Sentence-level metrics (dev set):**

| Metric | Value |
|--------|-------|
| Accuracy | 0.862 |
| Macro-F1 | 0.787 |
| SUPPORT P / R / F1 | 0.748 / 0.770 / 0.759 |
| CONTRADICT P / R / F1 | 0.676 / 0.702 / 0.689 |
| NEI P / R / F1 | 0.920 / 0.908 / 0.914 |

**End-to-end leaderboard evaluation (abstract-level, dev set):**

| Pipeline | Precision | Recall | F1 |
|----------|-----------|--------|----|
| Separate (cosine + zero-shot NLI) | 0.139 | 0.139 | 0.139 |
| **Joint sentence model** | **0.311** | **0.565** | **0.401** |

The joint model improved abstract-level F1 by nearly 3x (0.139 → 0.401).

### Comparison to Published Systems

| System | Abstract F1 | Notes |
|--------|-------------|-------|
| VeriSci (Wadden 2020) | 50.0 | Original baseline |
| **Ours (joint, dev)** | **40.1** | Hybrid retrieval + joint sentence model |
| ParagraphJoint (Li 2021) | 69.1 | End-to-end paragraph model |
| ARSJoint (Zhang 2021) | 62.4–71.2 | Joint model with rationale regularization |
| MultiVerS (Wadden 2022) | 72.5 | SOTA — Longformer + weak supervision |

Note: published results are on the test set; ours are on dev. Our system uses a modular pipeline (separate retrieval stage), while top systems use architectures specifically designed for joint verification.

### Verdict Prediction — Zero-Shot Baseline (SciFact Dev)

Pipeline: hybrid retrieval (top-5) -> full abstract -> DeBERTa-v3-large-mnli-fever-anli-ling-wanli (zero-shot).

| Metric | Value |
|--------|-------|
| Accuracy | 0.260 |
| Macro-F1 | 0.245 |
| SUPPORT P / R / F1 | 0.663 / 0.292 / 0.405 |
| CONTRADICT P / R / F1 | 0.862 / 0.205 / 0.331 |

Zero-shot NLI struggles with scientific claims — the model defaults to "neutral" for domain-specific evidence. The joint sentence-level model (above) addresses this bottleneck.

### Abstention with Retriever Disagreement (SciFact Dev, Joint Model)

The core hypothesis: disagreement between BM25 and dense retrievers signals when the system should abstain. We evaluate this with the joint sentence model.

**Does abstention improve accuracy?** Yes.

| Metric | No Abstention | With Abstention |
|--------|---------------|-----------------|
| Accuracy | 0.750 | **0.810** |
| Coverage | 100% | 72.9% |
| **Accuracy gain** | — | **+6.0%** |

By abstaining on 27% of uncertain claims, accuracy on answered claims rises from 75% to 81%.

**Does retriever disagreement predict correctness?** Yes.

| Retriever Agreement | Accuracy | n |
|---------------------|----------|---|
| High (Jaccard ≥ median) | **0.806** | 98 |
| Low (Jaccard < median) | 0.689 | 90 |
| **Gap** | **+11.7%** | |

When BM25 and dense retrieval agree on relevant documents, the system is correct 81% of the time. When they disagree, accuracy drops to 69%.

**Ablation: does the disagreement signal help beyond NLI confidence?**

| Abstention Variant | Best Accuracy (≥50% coverage) |
|--------------------|-------------------------------|
| No abstention | 0.750 |
| NLI confidence only (no disagreement) | 0.808 |
| NLI confidence + retriever disagreement | **0.812** |

The disagreement signal provides a small additional gain (+0.4%) on top of NLI confidence. Its main value is as a complementary signal: Pearson correlation between agreement and correctness is 0.043, confirming it captures information distinct from model confidence.

**Trade-off:** Abstention improves accuracy but reduces leaderboard F1 (0.401 → 0.377) because F1 penalizes lower recall. This is expected — a cautious system that abstains when uncertain will always trade recall for precision.

### Explanation Generation

The system supports three explanation methods:

| Method | Description | Model |
|--------|-------------|-------|
| **extractive** | Concatenates top evidence sentences with verdict framing | None |
| **template** | Fills predefined templates with evidence citations | None |
| **llm** | Generates fluent cited explanations via a causal LM | Gemma 4 E4B (default) |

The LLM method uses [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B) to produce natural-sounding 2-4 sentence explanations with `[N]` citation markers grounded in retrieved evidence. The model is loaded lazily and requires ~8 GB VRAM (GPU) or runs on CPU with float32.

## Setup

```bash
git clone https://github.com/Kai3421/scifact-claim-verification.git
cd scifact-claim-verification
pip install -e ".[dev]"

# Download SciFact data
mkdir -p data/raw && cd data/raw
wget https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
tar -xzf data.tar.gz
cd ../..
```

## Usage

```bash
# BM25 baseline
python scripts/01_baseline_retrieval.py

# Dense + hybrid ablation
python scripts/02_dense_retrieval.py
python scripts/02_dense_retrieval.py --model intfloat/e5-base-v2

# Retriever disagreement analysis
python scripts/03_disagreement_analysis.py
python scripts/03_disagreement_analysis.py --k 5

# Evidence selection + verdict prediction
python scripts/04_evidence_verdict.py
python scripts/04_evidence_verdict.py --mode bm25
python scripts/04_evidence_verdict.py --nli-model roberta-large-mnli --top-k 10

# Abstention evaluation
python scripts/05_abstention.py
python scripts/05_abstention.py --threshold 0.35
python scripts/05_abstention.py --mode bm25 --no-conflict-override

# Claim decomposition analysis
python scripts/06_claim_decomposition.py

# Explanation generation
python scripts/08_generation.py
python scripts/08_generation.py --method template
python scripts/08_generation.py --method llm
python scripts/08_generation.py --method llm --llm-model google/gemma-4-E4B

# Fine-tune joint sentence-level model
python scripts/07_finetune_joint.py
python scripts/07_finetune_joint.py --epochs 10 --lr 1e-5 --batch-size 4 --grad-accum 8

# Leaderboard predictions (AllenAI SciFact format)
python scripts/09_leaderboard_predictions.py --pipeline joint --split dev
python scripts/09_leaderboard_predictions.py --pipeline separate --split dev
python scripts/09_leaderboard_predictions.py --pipeline joint --split test

# Run tests
pytest tests/ -v
```

## Project Structure

```
src/claimverify/
    data/           SciFact corpus loader
    retrieval/      BM25, dense (FAISS), RRF fusion, reranker, disagreement analysis
    reasoning/      Rationale selection, NLI verdict prediction, joint sentence model, aggregation
    calibration/    Uncertainty signals, abstention gate, threshold tuning
    preprocessing/  Claim decomposition for compound claims
    evaluation/     Retrieval metrics, verdict metrics, citation fidelity, leaderboard formatter
    generation/     Cited explanation generation (template, extractive, LLM)
scripts/            Evaluation scripts
configs/            Hydra configuration
tests/              Unit tests (88 total)
results/            Evaluation outputs (JSON)
docs/               Architecture diagram
```

## Dataset

[SciFact](https://github.com/allenai/scifact) (Wadden et al., EMNLP 2020): 1,409 claims against 5,183 research abstracts with sentence-level evidence annotations.

## References

- Wadden et al., *Fact or Fiction: Verifying Scientific Claims* (EMNLP 2020)
- Wadden et al., *MultiVerS* (Findings of NAACL 2022)
- Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models* (NeurIPS 2021)
- Cormack et al., *Reciprocal Rank Fusion* (SIGIR 2009)
- Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique* (ICLR 2024)
- Atanasova et al., *Fact Checking with Insufficient Evidence* (TACL 2022)
