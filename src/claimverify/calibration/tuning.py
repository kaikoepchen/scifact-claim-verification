"""Threshold tuning and coverage-vs-risk evaluation."""

from __future__ import annotations

import numpy as np

from .signals import UncertaintySignals


def coverage_risk_curve(
    signals: list[UncertaintySignals],
    gold_labels: list[str],
    predicted_labels: list[str],
    n_thresholds: int = 50,
) -> list[dict]:
    """Compute accuracy at varying abstention thresholds.

    Sweeps the combined_score threshold from 0 to 1. At each threshold,
    claims below the threshold are abstained on, and we measure accuracy
    on the remaining (answered) claims.

    Returns list of {threshold, coverage, accuracy, n_answered, n_correct}.
    """
    scores = np.array([s.combined_score for s in signals])
    correct = np.array([p == g for p, g in zip(predicted_labels, gold_labels)])

    thresholds = np.linspace(0.0, float(np.max(scores)) + 0.01, n_thresholds)
    curve = []

    for t in thresholds:
        mask = scores >= t
        n_answered = int(mask.sum())

        if n_answered == 0:
            curve.append({
                "threshold": round(float(t), 4),
                "coverage": 0.0,
                "accuracy": 0.0,
                "n_answered": 0,
                "n_correct": 0,
            })
            continue

        n_correct = int(correct[mask].sum())
        coverage = n_answered / len(signals)
        accuracy = n_correct / n_answered

        curve.append({
            "threshold": round(float(t), 4),
            "coverage": round(coverage, 4),
            "accuracy": round(accuracy, 4),
            "n_answered": n_answered,
            "n_correct": n_correct,
        })

    return curve


def find_optimal_threshold(
    curve: list[dict],
    min_coverage: float = 0.5,
) -> dict:
    """Find the threshold that maximizes accuracy while keeping coverage above min_coverage."""
    valid = [p for p in curve if p["coverage"] >= min_coverage]
    if not valid:
        return curve[0] if curve else {}
    return max(valid, key=lambda p: p["accuracy"])


def auc_coverage_risk(curve: list[dict]) -> float:
    """Area under the coverage-accuracy curve (higher = better calibration)."""
    if len(curve) < 2:
        return 0.0
    coverages = [p["coverage"] for p in curve]
    accuracies = [p["accuracy"] for p in curve]

    auc = 0.0
    for i in range(1, len(curve)):
        dx = abs(coverages[i] - coverages[i - 1])
        avg_y = (accuracies[i] + accuracies[i - 1]) / 2
        auc += dx * avg_y
    return round(auc, 4)
