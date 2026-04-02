# SciFact Claim Verification

A scientific claim verification system built on hybrid retrieval-augmented generation. Given a claim like *"Aspirin reduces the risk of colorectal cancer"*, the system retrieves relevant research abstracts, selects rationale sentences, predicts a verdict (SUPPORT / CONTRADICT / NOT ENOUGH EVIDENCE), and generates a short cited explanation.

Built for the RAG course at the University of Zurich, FS 2026.

## Retrieval Results (SciFact Dev, 300 queries)

| Condition | nDCG@10 | Recall@5 | Recall@10 | MRR |
|-----------|---------|----------|-----------|-----|
| BM25 only | 0.830 | 0.878 | 0.941 | 0.797 |
| Dense only (MiniLM-L6) | 0.774 | 0.869 | 0.912 | 0.731 |
| **Hybrid (RRF)** | **0.850** | **0.905** | **0.950** | **0.824** |

## Architecture

```
Claim  -->  BM25 (sparse)  --+
                              +--> RRF Fusion --> [Reranker] --> Evidence Pack
Claim  -->  Dense (FAISS)  --+                                      |
                                                        Rationale Selection
                                                              |
                                                     Verdict (NLI) + Explanation
                                                              |
                                                      Abstention Gate
                                                              |
                                                        Final Output
```

## Setup

```bash
# Clone and install
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
# Phase 1: BM25 baseline
python scripts/01_baseline_retrieval.py

# Phase 2: Dense + hybrid ablation
python scripts/02_dense_retrieval.py
python scripts/02_dense_retrieval.py --model intfloat/e5-base-v2
python scripts/02_dense_retrieval.py --no-reranker

# Run tests
pytest tests/ -v
```

## Project Structure

```
src/claimverify/
    data/           SciFact corpus loader
    retrieval/      BM25, dense (FAISS), RRF fusion, cross-encoder reranker
    evaluation/     Retrieval metrics (Recall, nDCG, MRR) and verdict metrics (macro-F1)
    reasoning/      Verdict prediction (planned)
    generation/     Cited explanation generation (planned)
    calibration/    Abstention gate (planned)
scripts/            Evaluation scripts per phase
configs/            Hydra configuration
tests/              Unit tests
results/            Evaluation outputs (JSON)
```

## Dataset

[SciFact](https://github.com/allenai/scifact) (Wadden et al., EMNLP 2020): 1,409 claims against 5,183 research abstracts with sentence-level evidence annotations.

## References

- Wadden et al., *Fact or Fiction: Verifying Scientific Claims* (EMNLP 2020)
- Thakur et al., *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models* (NeurIPS 2021)
- Cormack et al., *Reciprocal Rank Fusion* (SIGIR 2009)
- Cohan et al., *SPECTER* (ACL 2020)
