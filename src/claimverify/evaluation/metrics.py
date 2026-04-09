"""Retrieval and verification evaluation metrics."""

from __future__ import annotations

import math
from typing import Optional


def recall_at_k(
    retrieved: list[str], relevant: set[str], k: int
) -> float:
    """Recall@k: fraction of relevant documents found in top-k."""
    if not relevant:
        return 0.0
    found = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return found / len(relevant)


def precision_at_k(
    retrieved: list[str], relevant: set[str], k: int
) -> float:
    """Precision@k: fraction of top-k documents that are relevant."""
    if k == 0:
        return 0.0
    found = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return found / k


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant document."""
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(
    retrieved: list[str],
    relevance: dict[str, int],
    k: int,
) -> float:
    """Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = relevance.get(doc_id, 0)
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return dcg


def ndcg_at_k(
    retrieved: list[str],
    relevance: dict[str, int],
    k: int,
) -> float:
    """Normalized DCG at k."""
    actual_dcg = dcg_at_k(retrieved, relevance, k)
    # Ideal ranking: sort by relevance descending
    ideal_docs = sorted(relevance.keys(), key=lambda d: relevance[d], reverse=True)
    ideal_dcg = dcg_at_k(ideal_docs, relevance, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def evaluate_retrieval(
    results: dict[str, list[tuple[str, float]]],
    qrels: dict[str, dict[str, int]],
    ks: list[int] = [5, 10],
) -> dict[str, float]:
    """Compute aggregate retrieval metrics over a set of queries.

    Args:
        results: {query_id: [(doc_id, score), ...]}
        qrels: {query_id: {doc_id: relevance}}
        ks: list of k values for Recall@k, nDCG@k

    Returns:
        Dictionary of metric_name -> mean value
    """
    metrics: dict[str, list[float]] = {}
    for metric_name in [f"recall@{k}" for k in ks] + [f"ndcg@{k}" for k in ks] + ["mrr"]:
        metrics[metric_name] = []

    for qid, qrel in qrels.items():
        if qid not in results:
            continue
        retrieved_ids = [doc_id for doc_id, _ in results[qid]]
        relevant = set(doc_id for doc_id, rel in qrel.items() if rel > 0)

        for k in ks:
            metrics[f"recall@{k}"].append(recall_at_k(retrieved_ids, relevant, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(retrieved_ids, qrel, k))

        metrics["mrr"].append(mrr(retrieved_ids, relevant))

    return {name: sum(vals) / len(vals) if vals else 0.0 for name, vals in metrics.items()}


def macro_f1(
    predictions: list[str],
    labels: list[str],
    classes: Optional[list[str]] = None,
) -> dict[str, float]:
    """Compute macro-F1 and per-class F1 for verdict evaluation."""
    if classes is None:
        classes = sorted(set(labels) | set(predictions))

    per_class = {}
    for cls in classes:
        tp = sum(1 for p, l in zip(predictions, labels) if p == cls and l == cls)
        fp = sum(1 for p, l in zip(predictions, labels) if p == cls and l != cls)
        fn = sum(1 for p, l in zip(predictions, labels) if p != cls and l == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}

    accuracy = sum(1 for p, l in zip(predictions, labels) if p == l) / len(labels) if labels else 0.0
    macro = sum(v["f1"] for v in per_class.values()) / len(per_class) if per_class else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro,
        "per_class": per_class,
    }


def sentence_selection_metrics(
    predicted: dict[str, list[int]],
    gold: dict[str, list[int]],
) -> dict[str, float]:
    """Evaluate rationale sentence selection quality.

    Args:
        predicted: {doc_id: [sentence_indices]} from the selector.
        gold: {doc_id: [sentence_indices]} from gold annotations.

    Returns:
        Precision, recall, F1 at the sentence level.
    """
    tp = fp = fn = 0

    all_doc_ids = set(predicted.keys()) | set(gold.keys())
    for doc_id in all_doc_ids:
        pred_set = set(predicted.get(doc_id, []))
        gold_set = set(gold.get(doc_id, []))
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"sentence_precision": precision, "sentence_recall": recall, "sentence_f1": f1}


def citation_quality(
    explanations: list[dict],
) -> dict[str, float]:
    """Evaluate citation fidelity of generated explanations.

    Args:
        explanations: list of {cited_refs: [int], available_refs: [int], verdict: str}

    Returns:
        citation_precision: fraction of cited refs that exist in evidence
        citation_recall: fraction of available refs that are cited
        unsupported_rate: fraction of explanations with at least one invalid citation
        empty_citation_rate: fraction of non-NEI explanations with zero citations
    """
    if not explanations:
        return {
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "unsupported_rate": 0.0,
            "empty_citation_rate": 0.0,
        }

    total_cited = 0
    total_valid = 0
    total_available = 0
    total_recalled = 0
    unsupported_count = 0
    empty_count = 0
    non_nei_count = 0

    for ex in explanations:
        cited = set(ex["cited_refs"])
        available = set(ex["available_refs"])

        total_cited += len(cited)
        valid = cited & available
        total_valid += len(valid)
        total_available += len(available)
        total_recalled += len(valid)

        if cited - available:
            unsupported_count += 1

        if ex["verdict"] != "NOT_ENOUGH_INFO":
            non_nei_count += 1
            if not cited:
                empty_count += 1

    precision = total_valid / total_cited if total_cited > 0 else 0.0
    recall = total_recalled / total_available if total_available > 0 else 0.0
    unsupported_rate = unsupported_count / len(explanations)
    empty_rate = empty_count / non_nei_count if non_nei_count > 0 else 0.0

    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "unsupported_rate": unsupported_rate,
        "empty_citation_rate": empty_rate,
    }
