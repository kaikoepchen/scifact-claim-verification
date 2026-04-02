"""Dense retrieval with sentence-transformer embeddings + FAISS."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-m3"


class DenseRetriever:
    """Bi-encoder dense retriever with FAISS index."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        query_prefix: str = "",
        doc_prefix: str = "",
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()

        self.index: Optional[faiss.IndexFlatIP] = None
        self.doc_ids: list[str] = []
        self._built = False

    def build_index(
        self,
        corpus: dict[str, str],
        batch_size: int = 64,
        save_path: Optional[str] = None,
    ) -> None:
        """Encode corpus and build FAISS index."""
        self.doc_ids = list(corpus.keys())
        texts = [f"{self.doc_prefix}{corpus[did]}" for did in self.doc_ids]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings.astype(np.float32))
        self._built = True

        if save_path:
            self.save_index(save_path)

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Retrieve top-k documents for a query."""
        assert self._built, "Call build_index() first."
        q_text = f"{self.query_prefix}{query}"
        q_emb = self.model.encode(
            [q_text], normalize_embeddings=True, convert_to_numpy=True
        )
        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append((self.doc_ids[idx], float(score)))
        return results

    def batch_retrieve(
        self, queries: dict[str, str], top_k: int = 10, batch_size: int = 64
    ) -> dict[str, list[tuple[str, float]]]:
        """Retrieve for multiple queries."""
        assert self._built, "Call build_index() first."
        qids = list(queries.keys())
        q_texts = [f"{self.query_prefix}{queries[qid]}" for qid in qids]

        q_embs = self.model.encode(
            q_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        scores, indices = self.index.search(q_embs.astype(np.float32), top_k)

        results = {}
        for i, qid in enumerate(qids):
            results[qid] = [
                (self.doc_ids[idx], float(scores[i][j]))
                for j, idx in enumerate(indices[i])
                if idx >= 0
            ]
        return results

    def save_index(self, path: str) -> None:
        """Save FAISS index and doc_ids to disk."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "index.faiss"))
        with open(p / "doc_ids.txt", "w") as f:
            f.write("\n".join(self.doc_ids))

    def load_index(self, path: str) -> None:
        """Load a previously saved index."""
        p = Path(path)
        self.index = faiss.read_index(str(p / "index.faiss"))
        with open(p / "doc_ids.txt") as f:
            self.doc_ids = f.read().strip().split("\n")
        self._built = True
