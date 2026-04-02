"""Retriever disagreement analysis: sparse-dense signals for uncertainty estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class DisagreementSignals:
    """Per-query disagreement signals between two ranked lists."""

    query_id: str
    jaccard_at_k: float          # |intersection| / |union| of top-k doc sets
    rank_correlation: float      # Spearman rho over shared documents
    overlap_count: int           # number of shared docs in top-k
    top1_same: bool              # whether both retrievers agree on rank-1
    score_margin_sparse: float   # gap between rank-1 and rank-2 sparse scores
    score_margin_dense: float    # gap between rank-1 and rank-2 dense scores
    min_rr_sparse: float         # reciprocal rank of first relevant doc (sparse)
    min_rr_dense: float          # reciprocal rank of first relevant doc (dense)


def jaccard_at_k(
    list_a: list[tuple[str, float]],
    list_b: list[tuple[str, float]],
    k: int = 10,
) -> float:
    """Jaccard similarity of top-k document sets."""
    set_a = {doc_id for doc_id, _ in list_a[:k]}
    set_b = {doc_id for doc_id, _ in list_b[:k]}
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def overlap_at_k(
    list_a: list[tuple[str, float]],
    list_b: list[tuple[str, float]],
    k: int = 10,
) -> int:
    """Count of shared documents in top-k."""
    set_a = {doc_id for doc_id, _ in list_a[:k]}
    set_b = {doc_id for doc_id, _ in list_b[:k]}
    return len(set_a & set_b)


def spearman_rank_correlation(
    list_a: list[tuple[str, float]],
    list_b: list[tuple[str, float]],
    k: int = 10,
) -> float:
    """Spearman rank correlation over documents shared in both top-k lists.

    Returns 0.0 if fewer than 2 shared documents (undefined correlation).
    """
    rank_a = {doc_id: rank for rank, (doc_id, _) in enumerate(list_a[:k])}
    rank_b = {doc_id: rank for rank, (doc_id, _) in enumerate(list_b[:k])}
    shared = set(rank_a) & set(rank_b)

    n = len(shared)
    if n < 2:
        return 0.0

    d_squared_sum = sum((rank_a[d] - rank_b[d]) ** 2 for d in shared)
    return 1.0 - (6.0 * d_squared_sum) / (n * (n * n - 1))


def score_margin(ranked_list: list[tuple[str, float]]) -> float:
    """Gap between rank-1 and rank-2 scores. Large margin = confident retriever."""
    if len(ranked_list) < 2:
        return 0.0
    return ranked_list[0][1] - ranked_list[1][1]


def reciprocal_rank(ranked_list: list[tuple[str, float]], relevant: set[str]) -> float:
    """Reciprocal rank of first relevant document. 0.0 if none found."""
    for i, (doc_id, _) in enumerate(ranked_list):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_disagreement(
    sparse_results: dict[str, list[tuple[str, float]]],
    dense_results: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]] | None = None,
    k: int = 10,
) -> list[DisagreementSignals]:
    """Compute per-query disagreement signals between sparse and dense retrieval.

    Args:
        sparse_results: {query_id: [(doc_id, score), ...]} from BM25.
        dense_results: {query_id: [(doc_id, score), ...]} from dense retriever.
        qrels: Optional {query_id: {doc_id: relevance}} for RR computation.
        k: Number of top results to compare.

    Returns:
        List of DisagreementSignals, one per query.
    """
    all_qids = set(sparse_results) | set(dense_results)
    signals = []

    for qid in sorted(all_qids):
        sparse = sparse_results.get(qid, [])
        dense = dense_results.get(qid, [])
        relevant = set()
        if qrels and qid in qrels:
            relevant = {did for did, rel in qrels[qid].items() if rel > 0}

        top1_a = sparse[0][0] if sparse else ""
        top1_b = dense[0][0] if dense else ""

        signals.append(DisagreementSignals(
            query_id=qid,
            jaccard_at_k=jaccard_at_k(sparse, dense, k),
            rank_correlation=spearman_rank_correlation(sparse, dense, k),
            overlap_count=overlap_at_k(sparse, dense, k),
            top1_same=(top1_a == top1_b and top1_a != ""),
            score_margin_sparse=score_margin(sparse),
            score_margin_dense=score_margin(dense),
            min_rr_sparse=reciprocal_rank(sparse, relevant) if relevant else 0.0,
            min_rr_dense=reciprocal_rank(dense, relevant) if relevant else 0.0,
        ))

    return signals


def disagreement_retrieval_correlation(
    signals: list[DisagreementSignals],
    qrels: dict[str, dict[str, int]],
    sparse_results: dict[str, list[tuple[str, float]]],
    dense_results: dict[str, list[tuple[str, float]]],
    k: int = 10,
) -> dict[str, float]:
    """Analyze whether retriever disagreement predicts retrieval failures.

    Splits queries into high-agreement and high-disagreement groups
    and compares retrieval success rates.

    Returns dict with correlation statistics.
    """
    # Only analyze queries that have relevance judgments
    evaluated = [s for s in signals if s.query_id in qrels]
    if not evaluated:
        return {}

    jaccards = [s.jaccard_at_k for s in evaluated]
    median_j = sorted(jaccards)[len(jaccards) // 2]

    high_agree = [s for s in evaluated if s.jaccard_at_k >= median_j]
    high_disagree = [s for s in evaluated if s.jaccard_at_k < median_j]

    def success_rate(group: list[DisagreementSignals]) -> float:
        """Fraction of queries where at least one relevant doc is in top-k of EITHER retriever."""
        if not group:
            return 0.0
        successes = 0
        for s in group:
            relevant = {d for d, r in qrels[s.query_id].items() if r > 0}
            sparse_top = {d for d, _ in sparse_results.get(s.query_id, [])[:k]}
            dense_top = {d for d, _ in dense_results.get(s.query_id, [])[:k]}
            if relevant & (sparse_top | dense_top):
                successes += 1
        return successes / len(group)

    def only_one_retriever_rate(group: list[DisagreementSignals]) -> float:
        """Fraction where only ONE retriever found a relevant doc (not both)."""
        if not group:
            return 0.0
        count = 0
        for s in group:
            relevant = {d for d, r in qrels[s.query_id].items() if r > 0}
            in_sparse = bool(relevant & {d for d, _ in sparse_results.get(s.query_id, [])[:k]})
            in_dense = bool(relevant & {d for d, _ in dense_results.get(s.query_id, [])[:k]})
            if in_sparse != in_dense:  # XOR: exactly one found it
                count += 1
        return count / len(group)

    # Point-biserial: correlation between jaccard and retrieval success
    successes = []
    for s in evaluated:
        relevant = {d for d, r in qrels[s.query_id].items() if r > 0}
        sparse_top = {d for d, _ in sparse_results.get(s.query_id, [])[:k]}
        dense_top = {d for d, _ in dense_results.get(s.query_id, [])[:k]}
        successes.append(1.0 if relevant & (sparse_top | dense_top) else 0.0)

    # Pearson correlation between jaccard and success
    n = len(evaluated)
    mean_j = sum(jaccards) / n
    mean_s = sum(successes) / n
    cov = sum((j - mean_j) * (s - mean_s) for j, s in zip(jaccards, successes)) / n
    std_j = math.sqrt(sum((j - mean_j) ** 2 for j in jaccards) / n) if n > 1 else 0
    std_s = math.sqrt(sum((s - mean_s) ** 2 for s in successes) / n) if n > 1 else 0
    pearson = cov / (std_j * std_s) if std_j > 0 and std_s > 0 else 0.0

    return {
        "n_evaluated": len(evaluated),
        "median_jaccard": median_j,
        "mean_jaccard": sum(jaccards) / len(jaccards),
        "n_high_agreement": len(high_agree),
        "n_high_disagreement": len(high_disagree),
        "success_rate_high_agreement": success_rate(high_agree),
        "success_rate_high_disagreement": success_rate(high_disagree),
        "success_gap": success_rate(high_agree) - success_rate(high_disagree),
        "one_retriever_only_rate_agree": only_one_retriever_rate(high_agree),
        "one_retriever_only_rate_disagree": only_one_retriever_rate(high_disagree),
        "pearson_jaccard_success": pearson,
        "top1_agreement_rate": sum(1 for s in evaluated if s.top1_same) / len(evaluated),
    }
