# SciFact Claim Verification

A scientific claim verification system built on hybrid retrieval-augmented generation. Given a claim like *"Aspirin reduces the risk of colorectal cancer"*, the system retrieves relevant research abstracts, selects rationale sentences, predicts a verdict (SUPPORT / CONTRADICT / NOT ENOUGH EVIDENCE), and generates a cited explanation.

**Research angle:** We use the disagreement between sparse (BM25) and dense retrieval as a first-class uncertainty signal — for calibrated abstention at inference and for KL-regularized training.

Built for the RAG course at the University of Zurich, FS 2026.

## Architecture

<p align="center">
  <img src="docs/architecture.png" alt="Pipeline Architecture" width="750"/>
</p>

## Key Results

### Retrieval (SciFact Dev, 300 queries)

| Method | nDCG@10 |
|--------|---------|
| BM25 only | 0.830 |
| Dense only (MiniLM-L6) | 0.774 |
| **Hybrid (RRF)** | **0.850** |

### Verdict Prediction (SciFact Dev)

| Pipeline | Abstract F1 |
|----------|-------------|
| Separate (cosine + zero-shot NLI) | 0.139 |
| **Joint sentence model (DeBERTa-v3-large)** | **0.401** |

### Abstention via Retriever Disagreement

| Retriever Agreement | Accuracy |
|---------------------|----------|
| High (Jaccard ≥ median) | **80.6%** |
| Low (Jaccard < median) | 68.9% |
| **Gap** | **+11.7pp** |

With multi-signal abstention (NLI confidence + retriever disagreement), accuracy on answered claims rises from 75.0% → 81.0% at 72.9% coverage.

### KL Regularization (λ ablation)

| λ | Macro-F1 | CONTRADICT F1 |
|---|----------|---------------|
| 0.0 (baseline) | 0.789 | 0.696 |
| **0.1** | **0.798** | **0.729** |
| 0.3 | 0.789 | 0.701 |
| 0.5 | 0.797 | 0.712 |

KL regularization at λ=0.1 improves the hardest class (CONTRADICT) by +3.3pp F1.

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

## Project Structure

```
src/claimverify/
    data/           SciFact corpus loader
    retrieval/      BM25, dense (FAISS), RRF fusion, reranker, disagreement
    reasoning/      Rationale selection, NLI verdict, joint sentence model
    calibration/    Uncertainty signals, abstention gate
    preprocessing/  Claim decomposition
    evaluation/     Retrieval metrics, verdict metrics, leaderboard formatter
    generation/     Cited explanation generation (extractive, template, LLM)
scripts/            Evaluation and training scripts
results/            Evaluation outputs (JSON)
docs/               Architecture diagram, technical report
```

## Dataset

[SciFact](https://github.com/allenai/scifact) (Wadden et al., EMNLP 2020): 1,409 claims against 5,183 research abstracts with sentence-level evidence annotations.

## References

- Wadden et al., *Fact or Fiction: Verifying Scientific Claims* (EMNLP 2020)
- Wadden et al., *MultiVerS* (Findings of NAACL 2022)
- Cormack et al., *Reciprocal Rank Fusion* (SIGIR 2009)
- Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique* (ICLR 2024)
