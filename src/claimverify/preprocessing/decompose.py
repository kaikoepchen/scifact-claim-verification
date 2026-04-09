"""Claim decomposition: split compound claims into atomic sub-claims."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class DecomposedClaim:
    original: str
    sub_claims: list[str]
    is_compound: bool

    @property
    def n_parts(self) -> int:
        return len(self.sub_claims)


CONJUNCTIONS = [
    (r',\s+and\s+', ', and '),
    (r'\s+and\s+', ' and '),
    (r',\s+while\s+', ', while '),
    (r',\s+whereas\s+', ', whereas '),
    (r',\s+but\s+', ', but '),
    (r'\s+as well as\s+', ' as well as '),
    (r',\s+additionally\s+', ', additionally '),
]


class ClaimDecomposer:
    """Splits compound scientific claims into atomic sub-claims.

    Uses syntactic patterns to identify coordinated clauses joined by
    conjunctions. Each sub-claim retains the shared subject/context
    from the original claim.
    """

    def __init__(self, min_subclaim_words: int = 4):
        self.min_subclaim_words = min_subclaim_words

    def decompose(self, claim: str) -> DecomposedClaim:
        sub_claims = self._split_compound(claim)

        if len(sub_claims) <= 1:
            return DecomposedClaim(
                original=claim,
                sub_claims=[claim],
                is_compound=False,
            )

        # Filter out fragments that are too short
        valid = [s for s in sub_claims if len(s.split()) >= self.min_subclaim_words]
        if len(valid) <= 1:
            return DecomposedClaim(
                original=claim,
                sub_claims=[claim],
                is_compound=False,
            )

        return DecomposedClaim(
            original=claim,
            sub_claims=valid,
            is_compound=True,
        )

    def _split_compound(self, claim: str) -> list[str]:
        """Split on coordinating conjunctions that join independent clauses."""
        # Try splitting on ", and" first (most reliable signal for compound claims)
        parts = re.split(r',\s*and\s+', claim)
        if len(parts) >= 2 and all(len(p.split()) >= self.min_subclaim_words for p in parts):
            return self._attach_subject(parts, claim)

        # Try ", while" / ", whereas" / ", but" (contrastive)
        for pattern in [r',\s*while\s+', r',\s*whereas\s+', r',\s*but\s+']:
            parts = re.split(pattern, claim)
            if len(parts) >= 2 and all(len(p.split()) >= self.min_subclaim_words for p in parts):
                return self._attach_subject(parts, claim)

        # Try "and" mid-sentence with verb phrases on both sides
        parts = self._split_verb_coordination(claim)
        if parts:
            return parts

        return [claim]

    def _attach_subject(self, parts: list[str], original: str) -> list[str]:
        """If later parts lack a subject, prepend the subject from the first part."""
        if len(parts) <= 1:
            return parts

        first = parts[0].strip()
        result = [first]

        subject = self._extract_subject(first)

        for part in parts[1:]:
            part = part.strip()
            if not part:
                continue
            if self._looks_like_full_clause(part):
                result.append(part)
            elif subject:
                result.append(f"{subject} {part}")
            else:
                result.append(part)

        return result

    def _extract_subject(self, clause: str) -> str:
        """Extract the subject phrase (rough heuristic: everything before the first verb-like word)."""
        verb_indicators = [
            'increases', 'decreases', 'reduces', 'causes', 'promotes', 'inhibits',
            'activates', 'suppresses', 'triggers', 'induces', 'prevents', 'alters',
            'enhances', 'impairs', 'mediates', 'regulates', 'modulates', 'leads',
            'results', 'is', 'are', 'was', 'were', 'has', 'have', 'does', 'do',
            'affects', 'plays', 'contributes', 'requires', 'involves', 'shows',
        ]
        words = clause.split()
        for i, word in enumerate(words):
            if word.lower().rstrip('.,;') in verb_indicators and i > 0:
                return ' '.join(words[:i])
        return ''

    def _looks_like_full_clause(self, text: str) -> bool:
        """Check if text looks like it has its own subject (starts with a noun/determiner)."""
        starters = [
            'the', 'a', 'an', 'this', 'these', 'it', 'its', 'they', 'their',
            'there', 'such', 'both', 'each', 'all', 'no', 'some',
        ]
        first_word = text.split()[0].lower() if text.split() else ''
        return first_word in starters or first_word[0].isupper()

    def _split_verb_coordination(self, claim: str) -> list[str] | None:
        """Split 'X does A and does B' patterns."""
        match = re.search(
            r'^(.+?)\s+(increases|decreases|reduces|causes|promotes|inhibits|activates|suppresses'
            r'|enhances|impairs|triggers|induces|prevents|alters)\s+(.+?)\s+and\s+'
            r'(increases|decreases|reduces|causes|promotes|inhibits|activates|suppresses'
            r'|enhances|impairs|triggers|induces|prevents|alters)\s+(.+)$',
            claim, re.IGNORECASE,
        )
        if not match:
            return None

        subject = match.group(1)
        verb1, obj1 = match.group(2), match.group(3)
        verb2, obj2 = match.group(4), match.group(5)

        part1 = f"{subject} {verb1} {obj1}"
        part2 = f"{subject} {verb2} {obj2}"

        if (len(part1.split()) >= self.min_subclaim_words
                and len(part2.split()) >= self.min_subclaim_words):
            return [part1, part2]
        return None

    def batch_decompose(self, claims: list[str]) -> list[DecomposedClaim]:
        return [self.decompose(c) for c in claims]
