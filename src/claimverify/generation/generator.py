"""Explanation generator: produces cited natural language explanations."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .citation import CitationContext
from .templates import generate_template_explanation


@dataclass
class Explanation:
    text: str
    cited_refs: list[int]
    verdict: str
    method: str


class ExplanationGenerator:
    """Generates cited explanations for claim verdicts.

    Supports three modes:
    - "template": fast, deterministic, no model needed
    - "extractive": selects and combines the most relevant evidence
      sentences into a coherent explanation
    - "llm": generates fluent explanations using a causal language model
    """

    def __init__(self, method: str = "extractive", llm_model: str | None = None):
        self.method = method
        self._llm_generator = None
        self._llm_model = llm_model

    def _get_llm_generator(self):
        if self._llm_generator is None:
            from .llm_generator import LLMExplanationGenerator, LLMGeneratorConfig
            config = LLMGeneratorConfig(model_name=self._llm_model) if self._llm_model else None
            self._llm_generator = LLMExplanationGenerator(config=config)
        return self._llm_generator

    def generate(self, ctx: CitationContext) -> Explanation:
        if self.method == "template":
            return self._template(ctx)
        if self.method == "llm":
            return self._get_llm_generator().generate(ctx)
        return self._extractive(ctx)

    def _template(self, ctx: CitationContext) -> Explanation:
        text = generate_template_explanation(ctx)
        cited = _extract_ref_ids(text)
        return Explanation(
            text=text, cited_refs=cited, verdict=ctx.verdict, method="template",
        )

    def _extractive(self, ctx: CitationContext) -> Explanation:
        """Build explanation by combining evidence with verdict framing."""
        if ctx.verdict == "NOT_ENOUGH_INFO" or not ctx.evidence:
            return Explanation(
                text="There is not enough evidence to verify this claim.",
                cited_refs=[],
                verdict=ctx.verdict,
                method="extractive",
            )

        sorted_evidence = sorted(ctx.evidence, key=lambda e: e.score, reverse=True)

        verdict_word = "supported" if ctx.verdict == "SUPPORT" else "contradicted"

        parts = [f'The claim "{ctx.claim}" is {verdict_word} by the following evidence:']
        for e in sorted_evidence:
            parts.append(f"- {e.text} [{e.ref_id}]")

        text = "\n".join(parts)
        cited = [e.ref_id for e in sorted_evidence]
        return Explanation(
            text=text, cited_refs=cited, verdict=ctx.verdict, method="extractive",
        )

    def batch_generate(self, contexts: list[CitationContext]) -> list[Explanation]:
        return [self.generate(ctx) for ctx in contexts]


def _extract_ref_ids(text: str) -> list[int]:
    """Extract [N] citation markers from generated text."""
    return [int(m) for m in re.findall(r'\[(\d+)\]', text)]
