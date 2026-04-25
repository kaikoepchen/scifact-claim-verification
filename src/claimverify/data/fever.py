"""FEVER dataset loading. Mirrors the SciFact interface (Claim/Abstract/dev_claims/abstracts).

Source: HuggingFace `fever` dataset. The Wikipedia corpus is large (~5M pages,
~30 GB), so we subset it: docs cited by the evaluated dev claims are always
kept, plus additional docs sampled up to a cap (`max_corpus_docs`, default 50k)
to act as BM25/dense negatives. This keeps memory and disk practical on a T4.

Label mapping:
  SUPPORTS        -> SUPPORT
  REFUTES         -> CONTRADICT
  NOT ENOUGH INFO -> NEI

Evidence format conversion:
  FEVER stores one row per (annotator, evidence sentence) pointer with
  `evidence_wiki_url` (doc title) and `evidence_sentence_id`. We group by
  (annotator, doc) into per-doc {label, sentences} entries to match SciFact's
  `evidence: dict[doc_id, list[{label, sentences}]]` shape.

Requires: `pip install datasets`.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .scifact import Abstract, Claim


LABEL_MAP = {
    "SUPPORTS": "SUPPORT",
    "REFUTES": "CONTRADICT",
    "NOT ENOUGH INFO": "NEI",
}

DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "fever_cache"


def _parse_lines(lines_field: str) -> list[str]:
    """Parse FEVER wiki `lines` column: `<sent_id>\\t<sentence>\\t<links>` per line."""
    if not lines_field:
        return []
    out: list[str] = []
    for raw in lines_field.split("\n"):
        if not raw:
            continue
        parts = raw.split("\t")
        # Format is "idx\tsentence\t<links>". Some entries have empty sentences.
        if len(parts) >= 2:
            sent = parts[1].strip()
            out.append(sent)
        else:
            out.append("")
    return out


@dataclass
class Fever:
    """Loads and holds a subset of FEVER for claim verification eval."""

    abstracts: dict[str, Abstract] = field(default_factory=dict)
    dev_claims: list[Claim] = field(default_factory=list)
    train_claims: list[Claim] = field(default_factory=list)

    @classmethod
    def load(
        cls,
        max_dev_claims: int = 1000,
        max_corpus_docs: int = 50_000,
        cache_dir: Optional[str | Path] = None,
        seed: int = 42,
        hf_dataset: str = "fever",
        include_train: bool = False,
        max_train_claims: int = 5_000,
    ) -> "Fever":
        """Load FEVER `paper_dev` claims (with evidence) and a subset of wiki pages.

        Args:
            max_dev_claims: cap on dev claims with evidence (SUPPORTS/REFUTES only).
            max_corpus_docs: cap on total wiki pages indexed (gold + sampled).
            cache_dir: HF datasets cache directory.
            seed: RNG seed for reservoir sampling of negative docs.
            hf_dataset: HF dataset name, default `fever`.
            include_train: also load train claims (used by KL fine-tuning script).
            max_train_claims: cap on train claims when `include_train=True`.
        """
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "FEVER loader requires `datasets`. Install with: pip install datasets"
            ) from e

        cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)

        fv = cls()

        # ── Load dev claims ──────────────────────────────────────────────
        dev_ds = load_dataset(hf_dataset, "v1.0", split="paper_dev",
                              cache_dir=str(cache_dir))
        fv.dev_claims, dev_titles = cls._build_claims(
            dev_ds, max_claims=max_dev_claims, require_evidence=True,
        )

        needed_titles: set[str] = set(dev_titles)

        if include_train:
            train_ds = load_dataset(hf_dataset, "v1.0", split="train",
                                    cache_dir=str(cache_dir))
            fv.train_claims, train_titles = cls._build_claims(
                train_ds, max_claims=max_train_claims, require_evidence=True,
            )
            needed_titles.update(train_titles)

        # ── Load wiki pages (streaming + reservoir sample for negatives) ─
        wiki_ds = load_dataset(
            hf_dataset, "wiki_pages", split="wikipedia_pages",
            cache_dir=str(cache_dir), streaming=True,
        )

        rng = random.Random(seed)
        n_neg_target = max(0, max_corpus_docs - len(needed_titles))
        reservoir: list[Abstract] = []
        seen_neg = 0
        gold_seen: set[str] = set()

        for row in wiki_ds:
            title = row.get("id") or ""
            if not title:
                continue
            sentences = _parse_lines(row.get("lines", ""))
            text = (row.get("text") or " ".join(sentences)).strip()
            if not sentences and not text:
                continue
            ab = Abstract(
                doc_id=title,
                title=title.replace("_", " "),
                text=text,
                sentences=sentences,
            )
            if title in needed_titles:
                if title not in gold_seen:
                    fv.abstracts[title] = ab
                    gold_seen.add(title)
                # Early stop once gold is complete and reservoir filled.
                if (
                    len(gold_seen) >= len(needed_titles)
                    and len(reservoir) >= n_neg_target
                    and seen_neg >= n_neg_target * 4  # reservoir well-mixed
                ):
                    break
                continue

            # Reservoir sampling for negative docs.
            seen_neg += 1
            if len(reservoir) < n_neg_target:
                reservoir.append(ab)
            else:
                j = rng.randint(0, seen_neg - 1)
                if j < n_neg_target:
                    reservoir[j] = ab

        for ab in reservoir:
            fv.abstracts.setdefault(ab.doc_id, ab)

        return fv

    @staticmethod
    def _build_claims(ds, max_claims: int, require_evidence: bool) -> tuple[list[Claim], set[str]]:
        """Group flat FEVER rows into Claim objects. Returns (claims, doc_titles_used)."""
        # Group rows by claim id.
        by_id: dict[int, dict] = {}
        for row in ds:
            cid = int(row["id"])
            label = row.get("label") or "NOT ENOUGH INFO"
            entry = by_id.setdefault(cid, {
                "claim": row.get("claim", ""),
                "label": label,
                "groups": defaultdict(set),  # (annotator_id, doc_title) -> {sent_ids}
            })
            doc = row.get("evidence_wiki_url") or ""
            sent = row.get("evidence_sentence_id")
            annot = row.get("evidence_annotation_id")
            if annot is None:
                annot = row.get("evidence_id", 0)
            if doc and sent is not None and int(sent) >= 0:
                entry["groups"][(int(annot), doc)].add(int(sent))

        claims: list[Claim] = []
        used_titles: set[str] = set()

        for cid, c in sorted(by_id.items()):
            mapped = LABEL_MAP.get(c["label"], "NEI")
            if require_evidence and mapped == "NEI":
                continue
            if require_evidence and not c["groups"]:
                continue

            evidence: dict[str, list[dict]] = defaultdict(list)
            for (_, doc), sents in c["groups"].items():
                evidence[doc].append({"label": mapped, "sentences": sorted(sents)})
                used_titles.add(doc)

            claims.append(Claim(
                claim_id=str(cid),
                text=c["claim"],
                evidence=dict(evidence),
            ))
            if len(claims) >= max_claims:
                break

        return claims, used_titles

    # ── SciFact-compatible accessors ────────────────────────────────────

    def get_corpus_texts(self) -> dict[str, str]:
        """Return {doc_id: 'title text'} for indexing."""
        return {
            doc_id: f"{ab.title} {ab.text}".strip()
            for doc_id, ab in self.abstracts.items()
        }

    def get_qrels(self, split: str = "dev") -> dict[str, dict[str, int]]:
        claims = {"dev": self.dev_claims, "train": self.train_claims}[split]
        qrels: dict[str, dict[str, int]] = {}
        for claim in claims:
            if claim.evidence:
                qrels[claim.claim_id] = {doc_id: 1 for doc_id in claim.evidence}
        return qrels

    def get_verdict_labels(self, split: str = "dev") -> list[dict]:
        claims = {"dev": self.dev_claims, "train": self.train_claims}[split]
        labels = []
        for claim in claims:
            for doc_id, anns in claim.evidence.items():
                for ann in anns:
                    labels.append({
                        "claim_id": claim.claim_id,
                        "claim_text": claim.text,
                        "doc_id": doc_id,
                        "label": ann["label"],
                        "rationale_sentences": ann["sentences"],
                    })
        return labels

    @property
    def corpus_size(self) -> int:
        return len(self.abstracts)

    @property
    def num_claims(self) -> dict[str, int]:
        return {"train": len(self.train_claims), "dev": len(self.dev_claims)}

    def label_distribution(self, split: str = "dev") -> dict[str, int]:
        labels = self.get_verdict_labels(split)
        dist: dict[str, int] = {}
        for l in labels:
            dist[l["label"]] = dist.get(l["label"], 0) + 1
        return dist
