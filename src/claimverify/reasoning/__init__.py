"""Reasoning: rationale selection, verdict prediction, evidence aggregation."""

from .aggregation import AggregatedVerdict, aggregate_verdicts
from .rationale import RationaleSelector, ScoredSentence
from .verdict import VerdictPredictor, VerdictResult

__all__ = [
    "RationaleSelector",
    "ScoredSentence",
    "VerdictPredictor",
    "VerdictResult",
    "AggregatedVerdict",
    "aggregate_verdicts",
]
