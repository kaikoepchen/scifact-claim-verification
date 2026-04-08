"""Multi-evidence aggregation: combine verdicts from multiple abstracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from .verdict import VerdictResult


@dataclass
class AggregatedVerdict:
    label: str
    confidence: float
    evidence_count: int
    support_score: float
    contradict_score: float
    nei_score: float
    has_conflict: bool
    per_doc_verdicts: dict[str, VerdictResult] = field(default_factory=dict)


def aggregate_verdicts(
    verdicts: dict[str, VerdictResult],
    conflict_threshold: float = 0.3,
) -> AggregatedVerdict:
    """Aggregate per-document verdicts into a single claim-level verdict.

    Strategy: confidence-weighted voting across all evidence documents.
    If both SUPPORT and CONTRADICT have weight above the conflict threshold,
    flag the claim as conflicting evidence.
    """
    if not verdicts:
        return AggregatedVerdict(
            label="NOT_ENOUGH_INFO",
            confidence=0.0,
            evidence_count=0,
            support_score=0.0,
            contradict_score=0.0,
            nei_score=0.0,
            has_conflict=False,
        )

    support_score = 0.0
    contradict_score = 0.0
    nei_score = 0.0

    for v in verdicts.values():
        support_score += v.logits.get("SUPPORT", 0.0)
        contradict_score += v.logits.get("CONTRADICT", 0.0)
        nei_score += v.logits.get("NOT_ENOUGH_INFO", 0.0)

    n = len(verdicts)
    support_score /= n
    contradict_score /= n
    nei_score /= n

    scores = {
        "SUPPORT": support_score,
        "CONTRADICT": contradict_score,
        "NOT_ENOUGH_INFO": nei_score,
    }
    label = max(scores, key=scores.get)
    confidence = scores[label]

    has_conflict = (
        support_score > conflict_threshold and contradict_score > conflict_threshold
    )

    return AggregatedVerdict(
        label=label,
        confidence=confidence,
        evidence_count=n,
        support_score=support_score,
        contradict_score=contradict_score,
        nei_score=nei_score,
        has_conflict=has_conflict,
        per_doc_verdicts=verdicts,
    )
