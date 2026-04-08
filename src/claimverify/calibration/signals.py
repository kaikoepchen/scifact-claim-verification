"""Uncertainty signal extraction for abstention decisions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UncertaintySignals:
    claim_id: str
    nli_confidence: float
    nli_margin: float
    retrieval_score: float
    retriever_agreement: float
    evidence_count: int
    has_conflict: bool

    @property
    def combined_score(self) -> float:
        """Weighted combination of signals into a single confidence score.

        Higher = more confident the system should answer.
        Lower = more likely the system should abstain.
        """
        w_nli = 0.35
        w_margin = 0.20
        w_retrieval = 0.15
        w_agreement = 0.20
        w_conflict = 0.10

        conflict_penalty = 0.0 if not self.has_conflict else 1.0
        evidence_bonus = min(self.evidence_count / 3.0, 1.0)

        score = (
            w_nli * self.nli_confidence
            + w_margin * self.nli_margin
            + w_retrieval * min(self.retrieval_score, 1.0)
            + w_agreement * self.retriever_agreement
            - w_conflict * conflict_penalty
        )
        return score * (0.5 + 0.5 * evidence_bonus)


def extract_signals(
    claim_id: str,
    nli_logits: dict[str, float],
    retrieval_score: float,
    retriever_agreement: float,
    evidence_count: int,
    has_conflict: bool,
) -> UncertaintySignals:
    """Build UncertaintySignals from pipeline outputs."""
    sorted_probs = sorted(nli_logits.values(), reverse=True)
    confidence = sorted_probs[0] if sorted_probs else 0.0
    margin = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) >= 2 else 0.0

    return UncertaintySignals(
        claim_id=claim_id,
        nli_confidence=confidence,
        nli_margin=margin,
        retrieval_score=retrieval_score,
        retriever_agreement=retriever_agreement,
        evidence_count=evidence_count,
        has_conflict=has_conflict,
    )
