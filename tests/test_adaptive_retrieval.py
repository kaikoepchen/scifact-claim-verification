"""Tests for adaptive retrieval depth selection.

The pipeline routes between a base depth and an expanded depth based on the
Jaccard@k overlap between BM25 and dense top-100 lists. We mock both
retrievers so the test is deterministic and needs no models or data.
"""

from __future__ import annotations

import pytest

from claimverify.retrieval.pipeline import (
    AdaptiveRetrievalResult,
    RetrievalConfig,
    RetrievalPipeline,
)


class StubRetriever:
    """Returns a fixed ranked list regardless of query."""

    def __init__(self, ranked: list[tuple[str, float]]):
        self.ranked = ranked

    def retrieve(self, query: str, top_k: int) -> list[tuple[str, float]]:
        return self.ranked[:top_k]


def make_pipeline(
    bm25_ranked: list[tuple[str, float]],
    dense_ranked: list[tuple[str, float]],
    threshold: float = 0.2,
    base_top_k: int = 3,
    expanded_top_k: int = 10,
    corpus: dict[str, str] | None = None,
) -> RetrievalPipeline:
    config = RetrievalConfig(
        reranker_enabled=False,
        adaptive_disagreement_threshold=threshold,
        adaptive_base_top_k=base_top_k,
        adaptive_expanded_top_k=expanded_top_k,
    )
    pipe = RetrievalPipeline(config=config)
    pipe.bm25 = StubRetriever(bm25_ranked)
    pipe.dense = StubRetriever(dense_ranked)
    pipe.corpus = corpus or {doc_id: doc_id for doc_id, _ in bm25_ranked + dense_ranked}
    return pipe


def _ranked(doc_ids: list[str]) -> list[tuple[str, float]]:
    n = len(doc_ids)
    return [(d, float(n - i)) for i, d in enumerate(doc_ids)]


class TestAdaptiveRetrieval:
    def test_high_disagreement_expands_depth(self):
        # Disjoint top-10 → jaccard = 0 → expand
        bm25 = _ranked([f"a{i}" for i in range(20)])
        dense = _ranked([f"b{i}" for i in range(20)])
        pipe = make_pipeline(bm25, dense, threshold=0.2)

        out = pipe.adaptive_retrieve("any query")

        assert isinstance(out, AdaptiveRetrievalResult)
        assert out.expanded is True
        assert out.agreement == 0.0
        assert out.depth_used == 10
        assert len(out.results) == 10

    def test_high_agreement_uses_base_depth(self):
        # Identical top-10 → jaccard = 1 → base
        ranked = _ranked([f"d{i}" for i in range(20)])
        pipe = make_pipeline(ranked, ranked, threshold=0.2)

        out = pipe.adaptive_retrieve("any query")

        assert out.expanded is False
        assert out.agreement == 1.0
        assert out.depth_used == 3
        assert len(out.results) == 3

    def test_threshold_controls_routing(self):
        # 5/10 overlap → jaccard = 5/15 ≈ 0.333
        bm25 = _ranked([f"d{i}" for i in range(10)] + [f"x{i}" for i in range(10)])
        dense = _ranked([f"d{i}" for i in range(5)] + [f"y{i}" for i in range(15)])

        # Threshold below the agreement → base depth
        low_pipe = make_pipeline(bm25, dense, threshold=0.1)
        low_out = low_pipe.adaptive_retrieve("any query")
        assert low_out.agreement == pytest.approx(5 / 15)
        assert low_out.expanded is False
        assert low_out.depth_used == 3

        # Threshold above the agreement → expanded depth
        high_pipe = make_pipeline(bm25, dense, threshold=0.5)
        high_out = high_pipe.adaptive_retrieve("any query")
        assert high_out.agreement == pytest.approx(5 / 15)
        assert high_out.expanded is True
        assert high_out.depth_used == 10

    def test_per_call_overrides(self):
        ranked = _ranked([f"d{i}" for i in range(20)])
        pipe = make_pipeline(ranked, ranked, threshold=0.2)

        # Override base/expanded top_k at call time
        out = pipe.adaptive_retrieve(
            "q",
            disagreement_threshold=0.99,
            base_top_k=5,
            expanded_top_k=15,
        )
        # agreement is 1.0 but threshold is 0.99 → still base since 1.0 >= 0.99
        assert out.expanded is False
        assert out.depth_used == 5
        assert len(out.results) == 5

    def test_expanded_returns_more_docs_than_base(self):
        # The core invariant: when disagreement is high, we get more docs.
        bm25 = _ranked([f"a{i}" for i in range(20)])
        dense = _ranked([f"b{i}" for i in range(20)])
        identical = _ranked([f"d{i}" for i in range(20)])

        pipe_disagree = make_pipeline(bm25, dense, threshold=0.2)
        pipe_agree = make_pipeline(identical, identical, threshold=0.2)

        n_disagree = len(pipe_disagree.adaptive_retrieve("q").results)
        n_agree = len(pipe_agree.adaptive_retrieve("q").results)

        assert n_disagree > n_agree
        assert n_disagree == 10
        assert n_agree == 3

    def test_existing_retrieve_unchanged(self):
        # adaptive_retrieve must not break the original retrieve() contract.
        ranked = _ranked([f"d{i}" for i in range(20)])
        pipe = make_pipeline(ranked, ranked)

        out = pipe.retrieve("q", mode="bm25", top_k=5)
        assert len(out) == 5
        assert out[0][0] == "d0"
