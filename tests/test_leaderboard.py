"""Tests for SciFact leaderboard formatting and evaluation."""

import json
import tempfile
from pathlib import Path

from claimverify.evaluation.leaderboard import (
    format_prediction,
    write_predictions,
    load_predictions,
    evaluate_against_gold,
)


class TestFormatPrediction:
    def test_basic(self):
        pred = format_prediction(
            claim_id=84,
            doc_verdicts={"22406695": "SUPPORT", "7521113": "SUPPORT"},
            doc_rationales={"22406695": [1], "7521113": [4]},
        )
        assert pred["id"] == 84
        assert pred["evidence"]["22406695"]["label"] == "SUPPORT"
        assert pred["evidence"]["22406695"]["sentences"] == [1]

    def test_nei_filtered(self):
        pred = format_prediction(
            claim_id=1,
            doc_verdicts={"d1": "NOT_ENOUGH_INFO", "d2": "SUPPORT"},
            doc_rationales={"d1": [0], "d2": [1, 3]},
        )
        assert "d1" not in pred["evidence"]
        assert "d2" in pred["evidence"]

    def test_no_rationale_filtered(self):
        pred = format_prediction(
            claim_id=1,
            doc_verdicts={"d1": "SUPPORT"},
            doc_rationales={},
        )
        assert pred["evidence"] == {}

    def test_max_sentences_truncation(self):
        pred = format_prediction(
            claim_id=1,
            doc_verdicts={"d1": "SUPPORT"},
            doc_rationales={"d1": [0, 1, 2, 3, 4]},
            max_sentences=3,
        )
        assert len(pred["evidence"]["d1"]["sentences"]) == 3

    def test_empty_evidence(self):
        pred = format_prediction(claim_id=1, doc_verdicts={}, doc_rationales={})
        assert pred["evidence"] == {}

    def test_string_claim_id_converted(self):
        pred = format_prediction(claim_id="42", doc_verdicts={}, doc_rationales={})
        assert pred["id"] == 42


class TestWriteLoadPredictions:
    def test_roundtrip(self):
        preds = [
            {"id": 1, "evidence": {}},
            {"id": 84, "evidence": {"22406695": {"sentences": [1], "label": "SUPPORT"}}},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        write_predictions(preds, path)
        loaded = load_predictions(path)
        assert len(loaded) == 2
        assert loaded[0]["id"] == 1
        assert loaded[1]["id"] == 84

        Path(path).unlink()


class TestEvaluateAgainstGold:
    def test_perfect_prediction(self):
        gold = [{"id": 1, "evidence": {"d1": {"sentences": [2, 3], "label": "SUPPORT"}}}]
        preds = [{"id": 1, "evidence": {"d1": {"sentences": [2, 3], "label": "SUPPORT"}}}]
        result = evaluate_against_gold(preds, gold)
        assert result["abstract_f1"] == 1.0

    def test_wrong_label(self):
        gold = [{"id": 1, "evidence": {"d1": {"sentences": [2], "label": "SUPPORT"}}}]
        preds = [{"id": 1, "evidence": {"d1": {"sentences": [2], "label": "CONTRADICT"}}}]
        result = evaluate_against_gold(preds, gold)
        assert result["abstract_f1"] == 0.0

    def test_missing_prediction(self):
        gold = [{"id": 1, "evidence": {"d1": {"sentences": [2], "label": "SUPPORT"}}}]
        preds = [{"id": 1, "evidence": {}}]
        result = evaluate_against_gold(preds, gold)
        assert result["abstract_recall"] == 0.0

    def test_empty_gold(self):
        gold = [{"id": 1, "evidence": {}}]
        preds = [{"id": 1, "evidence": {"d1": {"sentences": [0], "label": "SUPPORT"}}}]
        result = evaluate_against_gold(preds, gold)
        assert result["abstract_precision"] == 0.0
