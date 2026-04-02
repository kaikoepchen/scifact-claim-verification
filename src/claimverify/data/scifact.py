"""SciFact dataset loading and corpus management.

Loads from local JSONL files (downloaded from allenai/scifact GitHub release).
Falls back to ir_datasets (BEIR/scifact) if local files are not found.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Default path relative to project root
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "data"


@dataclass
class Abstract:
    doc_id: str
    title: str
    text: str
    sentences: list[str]
    structured: bool = False


@dataclass
class Claim:
    claim_id: str
    text: str
    evidence: dict[str, list[dict]] = field(default_factory=dict)  # doc_id -> [{label, sentences}]
    cited_doc_ids: list[str] = field(default_factory=list)


@dataclass
class SciFact:
    """Loads and holds the SciFact dataset."""

    abstracts: dict[str, Abstract] = field(default_factory=dict)
    train_claims: list[Claim] = field(default_factory=list)
    dev_claims: list[Claim] = field(default_factory=list)
    test_claims: list[Claim] = field(default_factory=list)

    @classmethod
    def load(cls, data_dir: Optional[str | Path] = None) -> SciFact:
        """Load SciFact from local JSONL files.

        Args:
            data_dir: Path to directory containing corpus.jsonl, claims_*.jsonl.
                      Defaults to data/raw/data/ relative to project root.
        """
        data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

        if not data_dir.exists():
            raise FileNotFoundError(
                f"SciFact data not found at {data_dir}. "
                "Run the download script or set data_dir to the correct path.\n"
                "Download: https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
            )

        sf = cls()
        sf._load_corpus(data_dir / "corpus.jsonl")
        sf._load_claims(data_dir / "claims_train.jsonl", sf.train_claims)
        sf._load_claims(data_dir / "claims_dev.jsonl", sf.dev_claims)
        sf._load_claims(data_dir / "claims_test.jsonl", sf.test_claims)
        return sf

    def _load_corpus(self, path: Path) -> None:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                doc_id = str(row["doc_id"])
                sentences = row.get("abstract", [])
                if isinstance(sentences, list):
                    text = " ".join(sentences)
                else:
                    text = str(sentences)
                    sentences = [text]
                self.abstracts[doc_id] = Abstract(
                    doc_id=doc_id,
                    title=row.get("title", ""),
                    text=text,
                    sentences=sentences,
                    structured=row.get("structured", False),
                )

    def _load_claims(self, path: Path, target: list[Claim]) -> None:
        if not path.exists():
            return
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                # Parse evidence: {doc_id_str: [{sentences: [...], label: "SUPPORT"|"REFUTE"}]}
                raw_evidence = row.get("evidence", {})
                evidence: dict[str, list[dict]] = {}
                for doc_id_str, annotation_sets in raw_evidence.items():
                    evidence[doc_id_str] = annotation_sets

                claim = Claim(
                    claim_id=str(row["id"]),
                    text=row["claim"],
                    evidence=evidence,
                    cited_doc_ids=[str(d) for d in row.get("cited_doc_ids", [])],
                )
                target.append(claim)

    def get_corpus_texts(self) -> dict[str, str]:
        """Return {doc_id: 'title text'} for indexing."""
        return {
            doc_id: f"{ab.title} {ab.text}".strip()
            for doc_id, ab in self.abstracts.items()
        }

    def get_qrels(self, split: str = "dev") -> dict[str, dict[str, int]]:
        """Return qrels in {claim_id: {doc_id: relevance}} format.

        Only includes claims that have evidence annotations.
        """
        claims = {"dev": self.dev_claims, "train": self.train_claims, "test": self.test_claims}[split]
        qrels: dict[str, dict[str, int]] = {}
        for claim in claims:
            if claim.evidence:
                qrels[claim.claim_id] = {doc_id: 1 for doc_id in claim.evidence}
        return qrels

    def get_verdict_labels(self, split: str = "dev") -> list[dict]:
        """Return verdict labels for evaluation.

        Returns list of {claim_id, claim_text, doc_id, label, rationale_sentences}.
        """
        claims = {"dev": self.dev_claims, "train": self.train_claims}[split]
        labels = []
        for claim in claims:
            for doc_id, annotations in claim.evidence.items():
                for ann in annotations:
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
        return {
            "train": len(self.train_claims),
            "dev": len(self.dev_claims),
            "test": len(self.test_claims),
        }

    def label_distribution(self, split: str = "dev") -> dict[str, int]:
        """Count verdict labels in a split."""
        labels = self.get_verdict_labels(split)
        dist: dict[str, int] = {}
        for l in labels:
            dist[l["label"]] = dist.get(l["label"], 0) + 1
        return dist
