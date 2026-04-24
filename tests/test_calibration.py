"""Tests for calibration components — no model downloads needed."""

from claimverify.calibration.signals import UncertaintySignals, extract_signals
from claimverify.calibration.gate import AbstentionGate
from claimverify.calibration.tuning import coverage_risk_curve, find_optimal_threshold, auc_coverage_risk


def _make_signals(claim_id, confidence=0.8, margin=0.5, agreement=0.3):
    return UncertaintySignals(
        claim_id=claim_id,
        nli_confidence=confidence,
        nli_margin=margin,
        retriever_agreement=agreement,
    )


class TestUncertaintySignals:
    def test_combined_score_range(self):
        s = _make_signals("c1")
        assert 0.0 <= s.combined_score <= 1.0

    def test_high_agreement_raises_score(self):
        s_low = _make_signals("c1", agreement=0.1)
        s_high = _make_signals("c1", agreement=0.9)
        assert s_high.combined_score > s_low.combined_score

    def test_extract_signals(self):
        logits = {"SUPPORT": 0.7, "CONTRADICT": 0.2, "NOT_ENOUGH_INFO": 0.1}
        s = extract_signals("c1", logits, 0.3)
        assert s.nli_confidence == 0.7
        assert abs(s.nli_margin - 0.5) < 1e-6
        assert s.retriever_agreement == 0.3


class TestAbstentionGate:
    def test_confident_answers(self):
        gate = AbstentionGate(threshold=0.2)
        s = _make_signals("c1", confidence=0.9, margin=0.6, agreement=0.5)
        d = gate.decide(s)
        assert d.action == "answer"

    def test_low_confidence_abstains(self):
        gate = AbstentionGate(threshold=0.5)
        s = _make_signals("c1", confidence=0.3, margin=0.05, agreement=0.05)
        d = gate.decide(s)
        assert d.action == "abstain"

    def test_batch_decide(self):
        gate = AbstentionGate(threshold=0.3)
        signals = [_make_signals(f"c{i}", confidence=0.9) for i in range(5)]
        decisions = gate.batch_decide(signals)
        assert len(decisions) == 5


class TestCoverageRisk:
    def test_curve_shape(self):
        signals = [_make_signals(f"c{i}", confidence=0.5 + i * 0.1) for i in range(5)]
        gold = ["SUPPORT"] * 5
        preds = ["SUPPORT", "SUPPORT", "CONTRADICT", "SUPPORT", "SUPPORT"]
        curve = coverage_risk_curve(signals, gold, preds, n_thresholds=10)
        assert len(curve) == 10
        assert curve[0]["coverage"] >= curve[-1]["coverage"]

    def test_optimal_threshold(self):
        curve = [
            {"threshold": 0.1, "coverage": 1.0, "accuracy": 0.6, "n_answered": 10, "n_correct": 6},
            {"threshold": 0.3, "coverage": 0.7, "accuracy": 0.85, "n_answered": 7, "n_correct": 6},
            {"threshold": 0.5, "coverage": 0.4, "accuracy": 0.95, "n_answered": 4, "n_correct": 4},
        ]
        opt = find_optimal_threshold(curve, min_coverage=0.5)
        assert opt["threshold"] == 0.3

    def test_auc(self):
        curve = [
            {"threshold": 0.0, "coverage": 1.0, "accuracy": 0.7, "n_answered": 10, "n_correct": 7},
            {"threshold": 0.5, "coverage": 0.5, "accuracy": 0.9, "n_answered": 5, "n_correct": 5},
            {"threshold": 1.0, "coverage": 0.0, "accuracy": 0.0, "n_answered": 0, "n_correct": 0},
        ]
        auc = auc_coverage_risk(curve)
        assert auc > 0
