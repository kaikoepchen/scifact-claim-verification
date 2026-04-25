"""Tests for the FEVER data loader.

Most tests exercise pure helpers (`_parse_lines`, `_build_claims`) on
synthetic rows so they run offline. The full HF download path is only
exercised when `FEVER_LIVE_TEST=1` is set.
"""

from __future__ import annotations

import os

import pytest

from claimverify.data.fever import LABEL_MAP, Fever, _parse_lines
from claimverify.data.scifact import Abstract, Claim


# ── _parse_lines ──────────────────────────────────────────────────────

class TestParseLines:
    def test_empty(self):
        assert _parse_lines("") == []
        assert _parse_lines("\n\n") == []

    def test_basic(self):
        raw = "0\tFirst sentence.\tSomeLink\n1\tSecond sentence.\t"
        out = _parse_lines(raw)
        assert out == ["First sentence.", "Second sentence."]

    def test_missing_tabs(self):
        # Line without a tab (just an index) yields empty string.
        raw = "0\n1\tHas a sentence."
        out = _parse_lines(raw)
        assert out == ["", "Has a sentence."]

    def test_no_links_column(self):
        raw = "0\tOnly two columns."
        assert _parse_lines(raw) == ["Only two columns."]


# ── Label mapping ─────────────────────────────────────────────────────

class TestLabelMap:
    def test_label_keys(self):
        assert set(LABEL_MAP.keys()) == {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}

    def test_label_values(self):
        assert LABEL_MAP["SUPPORTS"] == "SUPPORT"
        assert LABEL_MAP["REFUTES"] == "CONTRADICT"
        assert LABEL_MAP["NOT ENOUGH INFO"] == "NEI"


# ── _build_claims (synthetic FEVER-shaped rows) ───────────────────────

def _row(claim_id, claim, label, doc=None, sent=None, annot=0, evidence_id=None):
    return {
        "id": claim_id,
        "claim": claim,
        "label": label,
        "evidence_wiki_url": doc or "",
        "evidence_sentence_id": sent if sent is not None else -1,
        "evidence_annotation_id": annot,
        "evidence_id": evidence_id if evidence_id is not None else 0,
    }


class TestBuildClaims:
    def test_supports_claim_label_mapped(self):
        rows = [
            _row(1, "X causes Y.", "SUPPORTS", doc="Doc_A", sent=2),
        ]
        claims, titles = Fever._build_claims(rows, max_claims=10, require_evidence=True)
        assert len(claims) == 1
        c = claims[0]
        assert c.claim_id == "1"
        assert c.text == "X causes Y."
        # Only doc with mapped SUPPORT label, sentence 2
        assert "Doc_A" in c.evidence
        ann = c.evidence["Doc_A"][0]
        assert ann["label"] == "SUPPORT"
        assert ann["sentences"] == [2]
        assert "Doc_A" in titles

    def test_refutes_maps_to_contradict(self):
        rows = [_row(2, "X is false.", "REFUTES", doc="Doc_B", sent=0)]
        claims, _ = Fever._build_claims(rows, max_claims=10, require_evidence=True)
        assert claims[0].evidence["Doc_B"][0]["label"] == "CONTRADICT"

    def test_nei_dropped_when_evidence_required(self):
        rows = [
            _row(3, "Unknown.", "NOT ENOUGH INFO"),
            _row(4, "Known.", "SUPPORTS", doc="Doc_C", sent=1),
        ]
        claims, _ = Fever._build_claims(rows, max_claims=10, require_evidence=True)
        assert [c.claim_id for c in claims] == ["4"]

    def test_multiple_evidence_pointers_grouped(self):
        # Same annotator + same doc → sentence ids merged into one entry.
        rows = [
            _row(5, "claim", "SUPPORTS", doc="Doc_D", sent=3, annot=10),
            _row(5, "claim", "SUPPORTS", doc="Doc_D", sent=7, annot=10),
            _row(5, "claim", "SUPPORTS", doc="Doc_D", sent=1, annot=11),  # different annotator
        ]
        claims, _ = Fever._build_claims(rows, max_claims=10, require_evidence=True)
        anns = claims[0].evidence["Doc_D"]
        # Two annotation sets (one per annotator).
        assert len(anns) == 2
        merged_sentences = sorted(s for ann in anns for s in ann["sentences"])
        assert merged_sentences == [1, 3, 7]

    def test_max_claims_cap(self):
        rows = [
            _row(i, f"claim {i}", "SUPPORTS", doc=f"Doc_{i}", sent=0)
            for i in range(20)
        ]
        claims, titles = Fever._build_claims(rows, max_claims=5, require_evidence=True)
        assert len(claims) == 5
        assert len(titles) == 5

    def test_missing_evidence_pointer_skipped(self):
        # Row with empty doc and sent=-1 yields no evidence groups.
        rows = [_row(6, "claim", "SUPPORTS")]
        claims, titles = Fever._build_claims(rows, max_claims=10, require_evidence=True)
        assert claims == []
        assert titles == set()


# ── Interface invariants on a synthetic Fever instance ────────────────

class TestFeverInterface:
    def _make_fever(self) -> Fever:
        fv = Fever()
        fv.abstracts = {
            "Doc_A": Abstract(
                doc_id="Doc_A", title="Doc A",
                text="Sentence one. Sentence two.",
                sentences=["Sentence one.", "Sentence two."],
            ),
            "Doc_B": Abstract(
                doc_id="Doc_B", title="Doc B",
                text="Other sentence.",
                sentences=["Other sentence."],
            ),
        }
        fv.dev_claims = [
            Claim(
                claim_id="100", text="Claim text.",
                evidence={"Doc_A": [{"label": "SUPPORT", "sentences": [0]}]},
            ),
            Claim(
                claim_id="101", text="Another claim.",
                evidence={"Doc_B": [{"label": "CONTRADICT", "sentences": [0]}]},
            ),
        ]
        return fv

    def test_corpus_texts_non_empty(self):
        fv = self._make_fever()
        texts = fv.get_corpus_texts()
        assert len(texts) == 2
        assert all(len(t.strip()) > 0 for t in texts.values())
        assert texts["Doc_A"].startswith("Doc A")

    def test_evidence_doc_ids_in_abstracts(self):
        fv = self._make_fever()
        for claim in fv.dev_claims:
            for doc_id in claim.evidence:
                assert doc_id in fv.abstracts

    def test_label_distribution(self):
        fv = self._make_fever()
        dist = fv.label_distribution("dev")
        assert dist == {"SUPPORT": 1, "CONTRADICT": 1}

    def test_qrels(self):
        fv = self._make_fever()
        qrels = fv.get_qrels("dev")
        assert qrels == {"100": {"Doc_A": 1}, "101": {"Doc_B": 1}}


# ── Live HF download (opt-in) ─────────────────────────────────────────

@pytest.mark.skipif(
    os.environ.get("FEVER_LIVE_TEST") != "1",
    reason="Skipped — set FEVER_LIVE_TEST=1 to enable HF download test.",
)
class TestFeverLive:
    def test_load_small(self):
        fv = Fever.load(max_dev_claims=20, max_corpus_docs=200)
        assert len(fv.dev_claims) > 0
        assert fv.corpus_size > 0
        # Every cited doc is indexed.
        for claim in fv.dev_claims:
            for doc_id in claim.evidence:
                assert doc_id in fv.abstracts, f"Missing gold doc {doc_id}"
        # Labels are mapped.
        labels = {ann["label"] for c in fv.dev_claims for anns in c.evidence.values() for ann in anns}
        assert labels.issubset({"SUPPORT", "CONTRADICT"})
