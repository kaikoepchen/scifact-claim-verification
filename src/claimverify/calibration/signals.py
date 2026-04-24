"""Uncertainty signal extraction for abstention decisions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UncertaintySignals:
    claim_id: str
    nli_confidence: float
    nli_margin: float
    retriever_agreement: float

    @property
    def combined_score(self) -> float:
        """Weighted combination of the three core signals.

        Three signals, each capturing a different uncertainty source:
        - nli_confidence: what the model thinks (max class probability)
        - nli_margin: how sure the model is (gap between top two classes)
        - retriever_agreement: what the retrieval thinks (Jaccard@k overlap)

        Higher = more confident the system should answer.
        Lower = more likely the system should abstain.
        """
        w_nli = 0.45
        w_margin = 0.25
        w_agreement = 0.30

        return (
            w_nli * self.nli_confidence
            + w_margin * self.nli_margin
            + w_agreement * self.retriever_agreement
        )


def extract_signals(
    claim_id: str,
    nli_logits: dict[str, float],
    retriever_agreement: float,
) -> UncertaintySignals:
    """Build UncertaintySignals from pipeline outputs."""
    sorted_probs = sorted(nli_logits.values(), reverse=True)
    confidence = sorted_probs[0] if sorted_probs else 0.0
    margin = (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) >= 2 else 0.0

    return UncertaintySignals(
        claim_id=claim_id,
        nli_confidence=confidence,
        nli_margin=margin,
        retriever_agreement=retriever_agreement,
    )
