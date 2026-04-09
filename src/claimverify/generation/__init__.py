"""Generation: cited explanation generation for claim verdicts."""

from .citation import CitationContext, CitedEvidence, build_citation_context
from .generator import Explanation, ExplanationGenerator

__all__ = [
    "CitationContext",
    "CitedEvidence",
    "build_citation_context",
    "Explanation",
    "ExplanationGenerator",
]
