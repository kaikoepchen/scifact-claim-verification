"""Rationale sentence selection from retrieved abstracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sentence_transformers import SentenceTransformer, util


@dataclass
class ScoredSentence:
    doc_id: str
    sentence_idx: int
    text: str
    score: float


class RationaleSelector:
    """Selects the most claim-relevant sentences from retrieved abstracts.

    Uses a bi-encoder to score each sentence against the claim,
    then picks the top-k highest-scoring sentences per document.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        max_sentences_per_doc: int = 3,
        min_score: float = 0.25,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.max_sentences_per_doc = max_sentences_per_doc
        self.min_score = min_score

    def select(
        self,
        claim: str,
        doc_id: str,
        sentences: list[str],
    ) -> list[ScoredSentence]:
        """Select top rationale sentences from a single abstract."""
        if not sentences:
            return []

        claim_emb = self.model.encode(claim, convert_to_tensor=True)
        sent_embs = self.model.encode(sentences, convert_to_tensor=True)

        scores = util.cos_sim(claim_emb, sent_embs)[0]

        scored = []
        for idx, (sent, score) in enumerate(zip(sentences, scores)):
            s = float(score)
            if s >= self.min_score:
                scored.append(ScoredSentence(
                    doc_id=doc_id,
                    sentence_idx=idx,
                    text=sent,
                    score=s,
                ))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: self.max_sentences_per_doc]

    def select_from_docs(
        self,
        claim: str,
        docs: dict[str, list[str]],
    ) -> dict[str, list[ScoredSentence]]:
        """Select rationale sentences from multiple retrieved abstracts.

        Args:
            claim: The claim text.
            docs: {doc_id: [sentence_1, sentence_2, ...]}

        Returns:
            {doc_id: [ScoredSentence, ...]} for each doc that has rationales.
        """
        results = {}
        for doc_id, sentences in docs.items():
            selected = self.select(claim, doc_id, sentences)
            if selected:
                results[doc_id] = selected
        return results
