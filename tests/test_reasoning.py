"""Tests for reasoning components — no model downloads needed."""

from claimverify.reasoning.aggregation import aggregate_verdicts, AggregatedVerdict
from claimverify.reasoning.verdict import VerdictResult
from claimverify.reasoning.rationale import ScoredSentence


class TestAggregation:
    def _make_verdict(self, label, support=0.0, contradict=0.0, nei=0.0):
        confidence = max(support, contradict, nei)
        return VerdictResult(
            label=label,
            confidence=confidence,
            logits={"SUPPORT": support, "CONTRADICT": contradict, "NOT_ENOUGH_INFO": nei},
        )

    def test_single_support(self):
        verdicts = {"d1": self._make_verdict("SUPPORT", support=0.85, contradict=0.05, nei=0.1)}
        agg = aggregate_verdicts(verdicts)
        assert agg.label == "SUPPORT"
        assert agg.evidence_count == 1
        assert not agg.has_conflict

    def test_single_contradict(self):
        verdicts = {"d1": self._make_verdict("CONTRADICT", support=0.1, contradict=0.8, nei=0.1)}
        agg = aggregate_verdicts(verdicts)
        assert agg.label == "CONTRADICT"

    def test_empty_verdicts(self):
        agg = aggregate_verdicts({})
        assert agg.label == "NOT_ENOUGH_INFO"
        assert agg.evidence_count == 0
        assert agg.confidence == 0.0

    def test_conflict_detection(self):
        verdicts = {
            "d1": self._make_verdict("SUPPORT", support=0.7, contradict=0.2, nei=0.1),
            "d2": self._make_verdict("CONTRADICT", support=0.15, contradict=0.75, nei=0.1),
        }
        agg = aggregate_verdicts(verdicts, conflict_threshold=0.3)
        assert agg.has_conflict
        assert agg.evidence_count == 2

    def test_no_conflict_when_one_sided(self):
        verdicts = {
            "d1": self._make_verdict("SUPPORT", support=0.8, contradict=0.1, nei=0.1),
            "d2": self._make_verdict("SUPPORT", support=0.9, contradict=0.05, nei=0.05),
        }
        agg = aggregate_verdicts(verdicts, conflict_threshold=0.3)
        assert not agg.has_conflict
        assert agg.label == "SUPPORT"

    def test_averaging_across_docs(self):
        verdicts = {
            "d1": self._make_verdict("SUPPORT", support=0.8, contradict=0.1, nei=0.1),
            "d2": self._make_verdict("SUPPORT", support=0.6, contradict=0.2, nei=0.2),
        }
        agg = aggregate_verdicts(verdicts)
        assert abs(agg.support_score - 0.7) < 1e-6
        assert abs(agg.contradict_score - 0.15) < 1e-6


class TestScoredSentence:
    def test_fields(self):
        s = ScoredSentence(doc_id="d1", sentence_idx=2, text="Some text.", score=0.85)
        assert s.doc_id == "d1"
        assert s.sentence_idx == 2
        assert s.score == 0.85
