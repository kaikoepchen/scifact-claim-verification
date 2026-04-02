"""End-to-end retrieval pipeline: sparse + dense + fusion + reranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .fusion import CrossEncoderReranker, ReciprocalRankFusion


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval pipeline."""

    # BM25
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    bm25_top_k: int = 100

    # Dense
    dense_model: str = "BAAI/bge-m3"
    dense_top_k: int = 100
    dense_batch_size: int = 64
    dense_index_path: Optional[str] = None

    # Fusion
    rrf_k: int = 60
    fusion_top_k: int = 50

    # Reranker
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 10


class RetrievalPipeline:
    """Orchestrates BM25 + Dense + RRF + Reranker."""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.bm25: Optional[BM25Retriever] = None
        self.dense: Optional[DenseRetriever] = None
        self.rrf = ReciprocalRankFusion(k=self.config.rrf_k)
        self.reranker: Optional[CrossEncoderReranker] = None
        self.corpus: dict[str, str] = {}

    def build(self, corpus: dict[str, str], skip_dense: bool = False) -> None:
        """Build all indexes from a {doc_id: text} corpus."""
        self.corpus = corpus

        # BM25
        self.bm25 = BM25Retriever(k1=self.config.bm25_k1, b=self.config.bm25_b)
        self.bm25.build_index(corpus)

        # Dense
        if not skip_dense:
            self.dense = DenseRetriever(model_name=self.config.dense_model)
            if self.config.dense_index_path:
                try:
                    self.dense.load_index(self.config.dense_index_path)
                except (FileNotFoundError, RuntimeError):
                    self.dense.build_index(
                        corpus,
                        batch_size=self.config.dense_batch_size,
                        save_path=self.config.dense_index_path,
                    )
            else:
                self.dense.build_index(corpus, batch_size=self.config.dense_batch_size)

        # Reranker
        if self.config.reranker_enabled:
            self.reranker = CrossEncoderReranker(model_name=self.config.reranker_model)

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: Optional[int] = None,
    ) -> list[tuple[str, float]]:
        """Retrieve documents for a single query.

        Args:
            query: The claim text.
            mode: "bm25", "dense", or "hybrid".
            top_k: Override final top-k.

        Returns:
            [(doc_id, score)] ranked by relevance.
        """
        final_k = top_k or self.config.reranker_top_k

        if mode == "bm25":
            assert self.bm25, "BM25 not built."
            candidates = self.bm25.retrieve(query, self.config.bm25_top_k)
        elif mode == "dense":
            assert self.dense, "Dense retriever not built."
            candidates = self.dense.retrieve(query, self.config.dense_top_k)
        elif mode == "hybrid":
            assert self.bm25 and self.dense, "Both retrievers must be built for hybrid mode."
            bm25_results = self.bm25.retrieve(query, self.config.bm25_top_k)
            dense_results = self.dense.retrieve(query, self.config.dense_top_k)
            candidates = self.rrf.fuse(
                bm25_results, dense_results, top_k=self.config.fusion_top_k
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'bm25', 'dense', or 'hybrid'.")

        # Rerank if enabled
        if self.reranker and self.corpus:
            candidates = self.reranker.rerank(
                query, candidates, self.corpus, top_k=final_k
            )
        else:
            candidates = candidates[:final_k]

        return candidates

    def batch_retrieve(
        self,
        queries: dict[str, str],
        mode: str = "hybrid",
        top_k: Optional[int] = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Retrieve for multiple queries."""
        return {
            qid: self.retrieve(query, mode=mode, top_k=top_k)
            for qid, query in queries.items()
        }
