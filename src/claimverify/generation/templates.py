"""Template-based explanation generation for claim verdicts."""

from __future__ import annotations

import random

from .citation import CitationContext

SUPPORT_TEMPLATES = [
    "The claim is supported by the evidence. {evidence_summary}",
    "Evidence supports this claim. {evidence_summary}",
    "Based on the retrieved evidence, this claim is supported. {evidence_summary}",
]

CONTRADICT_TEMPLATES = [
    "The claim is contradicted by the evidence. {evidence_summary}",
    "Evidence contradicts this claim. {evidence_summary}",
    "Based on the retrieved evidence, this claim is refuted. {evidence_summary}",
]

NEI_TEMPLATES = [
    "There is not enough evidence to verify this claim.",
    "The retrieved evidence does not provide sufficient information to support or refute this claim.",
    "The available evidence is insufficient for a definitive verdict.",
]


def generate_template_explanation(ctx: CitationContext) -> str:
    """Generate an explanation using templates and citation references."""
    if ctx.verdict == "NOT_ENOUGH_INFO" or not ctx.evidence:
        return random.choice(NEI_TEMPLATES)

    evidence_parts = []
    for e in ctx.evidence:
        evidence_parts.append(f"{e.text} [{e.ref_id}]")
    evidence_summary = " ".join(evidence_parts)

    if ctx.verdict == "SUPPORT":
        template = random.choice(SUPPORT_TEMPLATES)
    else:
        template = random.choice(CONTRADICT_TEMPLATES)

    return template.format(evidence_summary=evidence_summary)
