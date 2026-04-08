"""Calibration: uncertainty signals, abstention gate, threshold tuning."""

from .gate import AbstentionGate, GateDecision
from .signals import UncertaintySignals, extract_signals
from .tuning import auc_coverage_risk, coverage_risk_curve, find_optimal_threshold

__all__ = [
    "AbstentionGate",
    "GateDecision",
    "UncertaintySignals",
    "extract_signals",
    "coverage_risk_curve",
    "find_optimal_threshold",
    "auc_coverage_risk",
]
