"""Tests for retrieval components — no external data needed."""

from claimverify.retrieval.bm25 import BM25Retriever
from claimverify.retrieval.fusion import ReciprocalRankFusion
from claimverify.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    macro_f1,
)


# --- Fixtures ---

MINI_CORPUS = {
    "d1": "Aspirin reduces the risk of colorectal cancer in randomized trials",
    "d2": "BERT outperforms LSTM on the GLUE benchmark for natural language understanding",
    "d3": "Vitamin D supplementation does not prevent cardiovascular disease",
    "d4": "Deep learning models require large datasets for training",
    "d5": "Aspirin has antiplatelet effects and reduces inflammation",
}


# --- BM25 Tests ---

class TestBM25:
    def test_build_and_retrieve(self):
        bm25 = BM25Retriever()
        bm25.build_index(MINI_CORPUS)
        assert bm25.N == 5
        assert bm25.avgdl > 0

        results = bm25.retrieve("aspirin cancer", top_k=3)
        assert len(results) > 0
        # d1 should rank highest for "aspirin cancer"
        assert results[0][0] == "d1"

    def test_empty_query(self):
        bm25 = BM25Retriever()
        bm25.build_index(MINI_CORPUS)
        assert bm25.retrieve("", top_k=5) == []

    def test_no_match(self):
        bm25 = BM25Retriever()
        bm25.build_index(MINI_CORPUS)
        results = bm25.retrieve("quantum computing entanglement", top_k=5)
        # might return empty or very low scores
        assert isinstance(results, list)

    def test_batch_retrieve(self):
        bm25 = BM25Retriever()
        bm25.build_index(MINI_CORPUS)
        queries = {"q1": "aspirin cancer", "q2": "BERT GLUE"}
        results = bm25.batch_retrieve(queries, top_k=3)
        assert "q1" in results and "q2" in results


# --- RRF Tests ---

class TestRRF:
    def test_basic_fusion(self):
        rrf = ReciprocalRankFusion(k=60)
        list_a = [("d1", 10.0), ("d2", 8.0), ("d3", 5.0)]
        list_b = [("d2", 0.9), ("d3", 0.8), ("d1", 0.7)]
        fused = rrf.fuse(list_a, list_b, top_k=3)
        assert len(fused) == 3
        # d2 ranks 2nd in both, d1 ranks 1st and 3rd -> both should be near top
        doc_ids = [d for d, _ in fused]
        assert "d1" in doc_ids
        assert "d2" in doc_ids

    def test_disjoint_lists(self):
        rrf = ReciprocalRankFusion()
        list_a = [("d1", 10.0), ("d2", 8.0)]
        list_b = [("d3", 0.9), ("d4", 0.8)]
        fused = rrf.fuse(list_a, list_b, top_k=4)
        assert len(fused) == 4


# --- Metric Tests ---

class TestMetrics:
    def test_recall_at_k(self):
        assert recall_at_k(["d1", "d2", "d3"], {"d1", "d4"}, k=3) == 0.5
        assert recall_at_k(["d1", "d4"], {"d1", "d4"}, k=2) == 1.0
        assert recall_at_k(["d2", "d3"], {"d1"}, k=2) == 0.0

    def test_precision_at_k(self):
        assert precision_at_k(["d1", "d2", "d3"], {"d1"}, k=3) == 1 / 3

    def test_mrr(self):
        assert mrr(["d3", "d1", "d2"], {"d1"}) == 0.5
        assert mrr(["d1", "d2"], {"d1"}) == 1.0
        assert mrr(["d2", "d3"], {"d1"}) == 0.0

    def test_ndcg_at_k(self):
        # Perfect ranking
        relevance = {"d1": 1, "d2": 1}
        assert ndcg_at_k(["d1", "d2"], relevance, k=2) == 1.0

    def test_macro_f1(self):
        preds = ["SUPPORT", "CONTRADICT", "SUPPORT", "NEI"]
        labels = ["SUPPORT", "CONTRADICT", "CONTRADICT", "NEI"]
        result = macro_f1(preds, labels)
        assert 0.0 <= result["macro_f1"] <= 1.0
        assert 0.0 <= result["accuracy"] <= 1.0
