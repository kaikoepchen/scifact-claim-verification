"""Hybrid retrieval fusion: RRF and cross-encoder reranking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder as _CrossEncoder


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    Combines multiple ranked lists into a single ranking:
        RRF(d) = sum_i  1 / (k + rank_i(d))
    """

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(
        self, *ranked_lists: list[tuple[str, float]], top_k: int = 50
    ) -> list[tuple[str, float]]:
        """Fuse multiple ranked lists. Each list is [(doc_id, score)]."""
        rrf_scores: dict[str, float] = {}
        for ranked_list in ranked_lists:
            for rank, (doc_id, _score) in enumerate(ranked_list):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (
                    self.k + rank + 1
                )
        results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return results[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker for precision reranking of top candidates.

    Jointly encodes (query, document) pairs for fine-grained relevance scoring.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        import torch
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        corpus: dict[str, str],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Rerank candidates using cross-encoder scores.

        Args:
            query: The search query / claim text.
            candidates: [(doc_id, initial_score)] from first-stage retrieval.
            corpus: {doc_id: text} for looking up document content.
            top_k: Number of results to return after reranking.
        """
        if not candidates:
            return []

        pairs = []
        valid_doc_ids = []
        for doc_id, _ in candidates:
            if doc_id in corpus:
                pairs.append((query, corpus[doc_id]))
                valid_doc_ids.append(doc_id)

        if not pairs:
            return []

        scores = self.model.predict(pairs, show_progress_bar=False)
        reranked = list(zip(valid_doc_ids, [float(s) for s in scores]))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
