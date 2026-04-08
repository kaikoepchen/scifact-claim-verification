"""Abstention gate: decide whether to answer or abstain."""

from __future__ import annotations

from dataclasses import dataclass

from .signals import UncertaintySignals


@dataclass
class GateDecision:
    action: str  # "answer", "abstain", "flag_conflict"
    confidence_score: float
    reason: str


class AbstentionGate:
    """Multi-signal abstention gate.

    Uses the combined uncertainty score to decide whether the system
    is confident enough to answer, or should abstain.
    """

    def __init__(
        self,
        threshold: float = 0.4,
        conflict_override: bool = True,
    ):
        self.threshold = threshold
        self.conflict_override = conflict_override

    def decide(self, signals: UncertaintySignals) -> GateDecision:
        score = signals.combined_score

        if self.conflict_override and signals.has_conflict:
            return GateDecision(
                action="flag_conflict",
                confidence_score=score,
                reason="contradicting evidence from multiple sources",
            )

        if score < self.threshold:
            reasons = []
            if signals.nli_confidence < 0.5:
                reasons.append("low NLI confidence")
            if signals.nli_margin < 0.2:
                reasons.append("small margin between top predictions")
            if signals.retriever_agreement < 0.15:
                reasons.append("retrievers disagree on relevant documents")
            if signals.evidence_count == 0:
                reasons.append("no evidence retrieved")

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
