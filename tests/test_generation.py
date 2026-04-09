"""Tests for generation components."""

from claimverify.generation.citation import (
    CitedEvidence, CitationContext, build_citation_context,
)
from claimverify.generation.generator import ExplanationGenerator, _extract_ref_ids
from claimverify.generation.templates import generate_template_explanation
from claimverify.evaluation.metrics import citation_quality


class TestCitationContext:
    def test_build_context(self):
        doc_sents = {
            "d1": [(0, "Aspirin reduces cancer risk.", 0.9), (2, "The effect is dose-dependent.", 0.7)],
            "d2": [(1, "No significant effect was observed.", 0.6)],
        }
        ctx = build_citation_context("Aspirin prevents cancer", "SUPPORT", doc_sents)
        assert len(ctx.evidence) == 3
        assert ctx.evidence[0].ref_id == 1
        assert ctx.evidence[2].ref_id == 3
        assert ctx.verdict == "SUPPORT"

    def test_format_evidence_block(self):
        ctx = CitationContext(
            claim="test",
            verdict="SUPPORT",
            evidence=[
                CitedEvidence(ref_id=1, doc_id="d1", sentence_idx=0, text="Evidence one.", score=0.9),
                CitedEvidence(ref_id=2, doc_id="d1", sentence_idx=1, text="Evidence two.", score=0.8),
            ],
        )
        block = ctx.format_evidence_block()
        assert "[1]" in block
        assert "[2]" in block

    def test_empty_context(self):
        ctx = build_citation_context("claim", "NOT_ENOUGH_INFO", {})
        assert len(ctx.evidence) == 0
        assert ctx.get_ref_ids() == []


class TestTemplateGeneration:
    def test_support_template(self):
        ctx = CitationContext(
            claim="test claim",
            verdict="SUPPORT",
            evidence=[CitedEvidence(1, "d1", 0, "Some evidence.", 0.9)],
        )
        text = generate_template_explanation(ctx)
        assert "[1]" in text
        assert "support" in text.lower()

    def test_contradict_template(self):
        ctx = CitationContext(
            claim="test claim",
            verdict="CONTRADICT",
            evidence=[CitedEvidence(1, "d1", 0, "Contrary evidence.", 0.9)],
        )
        text = generate_template_explanation(ctx)
        assert "[1]" in text
        assert "contradict" in text.lower() or "refute" in text.lower()

    def test_nei_template(self):
        ctx = CitationContext(claim="test", verdict="NOT_ENOUGH_INFO", evidence=[])
        text = generate_template_explanation(ctx)
        assert any(w in text.lower() for w in ["not enough", "insufficient", "not provide sufficient"])


class TestExplanationGenerator:
    def test_extractive_support(self):
        gen = ExplanationGenerator(method="extractive")
        ctx = CitationContext(
            claim="Aspirin prevents cancer",
            verdict="SUPPORT",
            evidence=[
                CitedEvidence(1, "d1", 0, "Aspirin reduces cancer risk.", 0.9),
                CitedEvidence(2, "d1", 2, "Dose-dependent effect.", 0.7),
            ],
        )
        expl = gen.generate(ctx)
        assert expl.verdict == "SUPPORT"
        assert 1 in expl.cited_refs
        assert 2 in expl.cited_refs
        assert "supported" in expl.text.lower()

    def test_extractive_nei(self):
        gen = ExplanationGenerator(method="extractive")
        ctx = CitationContext(claim="test", verdict="NOT_ENOUGH_INFO", evidence=[])
        expl = gen.generate(ctx)
        assert expl.cited_refs == []

    def test_template_mode(self):
        gen = ExplanationGenerator(method="template")
        ctx = CitationContext(
            claim="test",
            verdict="SUPPORT",
            evidence=[CitedEvidence(1, "d1", 0, "Evidence.", 0.9)],
        )
        expl = gen.generate(ctx)
        assert expl.method == "template"

    def test_batch_generate(self):
        gen = ExplanationGenerator(method="extractive")
        contexts = [
            CitationContext("c1", "SUPPORT", [CitedEvidence(1, "d1", 0, "Ev.", 0.9)]),
            CitationContext("c2", "CONTRADICT", [CitedEvidence(1, "d2", 0, "Ev.", 0.8)]),
        ]
        results = gen.batch_generate(contexts)
        assert len(results) == 2


class TestExtractRefIds:
    def test_basic(self):
        assert _extract_ref_ids("Evidence shows [1] and [2].") == [1, 2]

    def test_no_refs(self):
        assert _extract_ref_ids("No citations here.") == []


class TestCitationQuality:
    def test_perfect_citations(self):
        explanations = [
            {"cited_refs": [1, 2], "available_refs": [1, 2], "verdict": "SUPPORT"},
        ]
        result = citation_quality(explanations)
        assert result["citation_precision"] == 1.0
        assert result["citation_recall"] == 1.0
        assert result["unsupported_rate"] == 0.0

    def test_unsupported_citation(self):
        explanations = [
            {"cited_refs": [1, 5], "available_refs": [1, 2], "verdict": "SUPPORT"},
        ]
        result = citation_quality(explanations)
        assert result["citation_precision"] == 0.5
        assert result["unsupported_rate"] == 1.0

    def test_empty_citations_on_verdict(self):
        explanations = [
            {"cited_refs": [], "available_refs": [1], "verdict": "SUPPORT"},
        ]
        result = citation_quality(explanations)
        assert result["empty_citation_rate"] == 1.0

    def test_empty_input(self):
        result = citation_quality([])
        assert result["citation_precision"] == 0.0
