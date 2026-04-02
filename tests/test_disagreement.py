"""Tests for retriever disagreement analysis."""

from claimverify.retrieval.disagreement import (
    jaccard_at_k,
    overlap_at_k,
    spearman_rank_correlation,
    score_margin,
    reciprocal_rank,
    compute_disagreement,
    disagreement_retrieval_correlation,
)


# --- Fixtures ---

# Identical rankings = perfect agreement
SAME_A = [("d1", 10.0), ("d2", 8.0), ("d3", 5.0), ("d4", 3.0), ("d5", 1.0)]
SAME_B = [("d1", 0.9),  ("d2", 0.8), ("d3", 0.7), ("d4", 0.6), ("d5", 0.5)]

# Completely disjoint = maximum disagreement
DISJOINT_A = [("d1", 10.0), ("d2", 8.0), ("d3", 5.0)]
DISJOINT_B = [("d4", 0.9),  ("d5", 0.8), ("d6", 0.7)]

# Partial overlap, different order
PARTIAL_A = [("d1", 10.0), ("d2", 8.0), ("d3", 5.0), ("d4", 3.0)]
PARTIAL_B = [("d3", 0.9),  ("d5", 0.8), ("d1", 0.7), ("d6", 0.6)]


class TestJaccard:
    def test_identical_sets(self):
        assert jaccard_at_k(SAME_A, SAME_B, k=5) == 1.0

    def test_disjoint_sets(self):
        assert jaccard_at_k(DISJOINT_A, DISJOINT_B, k=3) == 0.0

    def test_partial_overlap(self):
        # A top-4: {d1, d2, d3, d4}, B top-4: {d3, d5, d1, d6}
        # intersection: {d1, d3}, union: {d1, d2, d3, d4, d5, d6}
        j = jaccard_at_k(PARTIAL_A, PARTIAL_B, k=4)
        assert abs(j - 2 / 6) < 1e-9

    def test_empty_lists(self):
        assert jaccard_at_k([], [], k=10) == 1.0

    def test_k_larger_than_list(self):
        j = jaccard_at_k(SAME_A, SAME_B, k=100)
        assert j == 1.0


class TestOverlap:
    def test_identical(self):
        assert overlap_at_k(SAME_A, SAME_B, k=5) == 5

    def test_disjoint(self):
        assert overlap_at_k(DISJOINT_A, DISJOINT_B, k=3) == 0

    def test_partial(self):
        assert overlap_at_k(PARTIAL_A, PARTIAL_B, k=4) == 2


class TestRankCorrelation:
    def test_identical_ranking(self):
        rho = spearman_rank_correlation(SAME_A, SAME_B, k=5)
        assert abs(rho - 1.0) < 1e-9

    def test_reversed_ranking(self):
        reversed_b = list(reversed(SAME_A))
        rho = spearman_rank_correlation(SAME_A, reversed_b, k=5)
        assert rho < 0  # should be negative

    def test_no_shared_docs(self):
        rho = spearman_rank_correlation(DISJOINT_A, DISJOINT_B, k=3)
        assert rho == 0.0

    def test_one_shared_doc(self):
        a = [("d1", 10.0), ("d2", 8.0)]
        b = [("d1", 0.9), ("d3", 0.8)]
        rho = spearman_rank_correlation(a, b, k=2)
        assert rho == 0.0  # n < 2 -> 0


class TestScoreMargin:
    def test_normal(self):
        assert score_margin(SAME_A) == 2.0  # 10.0 - 8.0

    def test_single_result(self):
        assert score_margin([("d1", 5.0)]) == 0.0

    def test_empty(self):
        assert score_margin([]) == 0.0


class TestReciprocalRank:
    def test_first_is_relevant(self):
        assert reciprocal_rank(SAME_A, {"d1"}) == 1.0

    def test_second_is_relevant(self):
        assert reciprocal_rank(SAME_A, {"d2"}) == 0.5

    def test_none_relevant(self):
        assert reciprocal_rank(SAME_A, {"d99"}) == 0.0

    def test_empty_relevant(self):
        assert reciprocal_rank(SAME_A, set()) == 0.0


class TestComputeDisagreement:
    def test_basic(self):
        sparse = {"q1": SAME_A, "q2": DISJOINT_A}
        dense = {"q1": SAME_B, "q2": DISJOINT_B}
        signals = compute_disagreement(sparse, dense, k=3)
        assert len(signals) == 2

        s1 = [s for s in signals if s.query_id == "q1"][0]
        assert s1.jaccard_at_k == 1.0
        assert s1.top1_same is True

        s2 = [s for s in signals if s.query_id == "q2"][0]
        assert s2.jaccard_at_k == 0.0
        assert s2.top1_same is False

    def test_with_qrels(self):
        sparse = {"q1": SAME_A}
        dense = {"q1": SAME_B}
        qrels = {"q1": {"d1": 1}}
        signals = compute_disagreement(sparse, dense, qrels, k=5)
        assert signals[0].min_rr_sparse == 1.0
        assert signals[0].min_rr_dense == 1.0


class TestCorrelationAnalysis:
    def test_basic_correlation(self):
        # High agreement query (same docs) -> relevant doc found
        # High disagreement query (disjoint) -> relevant doc NOT found by dense
        sparse = {"q1": SAME_A, "q2": DISJOINT_A}
        dense = {"q1": SAME_B, "q2": DISJOINT_B}
        qrels = {"q1": {"d1": 1}, "q2": {"d1": 1}}  # d1 is relevant for both
        signals = compute_disagreement(sparse, dense, qrels, k=3)

        corr = disagreement_retrieval_correlation(signals, qrels, sparse, dense, k=3)
        assert corr["n_evaluated"] == 2
        assert "success_gap" in corr
        assert "pearson_jaccard_success" in corr
