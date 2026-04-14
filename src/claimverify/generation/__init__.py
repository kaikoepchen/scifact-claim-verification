"""Generation: cited explanation generation for claim verdicts."""

from .citation import CitationContext, CitedEvidence, build_citation_context
from .generator import Explanation, ExplanationGenerator
from .llm_generator import LLMExplanationGenerator, LLMGeneratorConfig

__all__ = [
    "CitationContext",
    "CitedEvidence",
    "build_citation_context",
    "Explanation",
    "ExplanationGenerator",
    "LLMExplanationGenerator",
    "LLMGeneratorConfig",
]
