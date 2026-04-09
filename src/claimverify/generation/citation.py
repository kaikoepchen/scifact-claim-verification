"""Citation formatting for evidence-backed explanations."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CitedEvidence:
    ref_id: int
    doc_id: str
    sentence_idx: int
    text: str
    score: float = 0.0


@dataclass
class CitationContext:
    claim: str
    verdict: str
    evidence: list[CitedEvidence] = field(default_factory=list)

    def format_evidence_block(self) -> str:
        lines = []
        for e in self.evidence:
            lines.append(f"[{e.ref_id}] {e.text}")
        return "\n".join(lines)

    def get_ref_ids(self) -> list[int]:
        return [e.ref_id for e in self.evidence]


def build_citation_context(
    claim: str,
    verdict: str,
    doc_sentences: dict[str, list[tuple[int, str, float]]],
) -> CitationContext:
    """Build a CitationContext from selected rationale sentences.

    Args:
        claim: The claim text.
        verdict: SUPPORT / CONTRADICT / NOT_ENOUGH_INFO.
        doc_sentences: {doc_id: [(sentence_idx, text, score), ...]}

    Returns:
        CitationContext with numbered evidence references.
    """
    evidence = []
    ref_id = 1
    for doc_id, sentences in doc_sentences.items():
        for sent_idx, text, score in sentences:
            evidence.append(CitedEvidence(
                ref_id=ref_id,
                doc_id=doc_id,
                sentence_idx=sent_idx,
                text=text.strip(),
                score=score,
            ))
            ref_id += 1
    return CitationContext(claim=claim, verdict=verdict, evidence=evidence)
