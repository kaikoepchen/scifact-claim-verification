"""Abstention gate: decide whether to answer or abstain."""

from __future__ import annotations

from dataclasses import dataclass

from .signals import UncertaintySignals


@dataclass
class GateDecision:
    action: str  # "answer" or "abstain"
    confidence_score: float
    reason: str


class AbstentionGate:
    """Three-signal abstention gate.

    Uses the combined uncertainty score (NLI confidence, NLI margin,
    retriever agreement) to decide whether the system is confident
    enough to answer or should abstain.
    """

    def __init__(self, threshold: float = 0.4):
        self.threshold = threshold

    def decide(self, signals: UncertaintySignals) -> GateDecision:
        score = signals.combined_score

        if score < self.threshold:
            reasons = []
            if signals.nli_confidence < 0.5:
                reasons.append("low NLI confidence")
            if signals.nli_margin < 0.2:
                reasons.append("small margin between top predictions")
            if signals.retriever_agreement < 0.15:
                reasons.append("retrievers disagree on relevant documents")

            return GateDecision(
                action="abstain",
                confidence_score=score,
                reason="; ".join(reasons) if reasons else "combined confidence below threshold",
            )

        return GateDecision(
            action="answer",
            confidence_score=score,
            reason="sufficient confidence",
        )

    def batch_decide(
        self, signals_list: list[UncertaintySignals]
    ) -> list[GateDecision]:
        return [self.decide(s) for s in signals_list]
