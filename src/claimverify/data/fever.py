"""FEVER dataset loading. Mirrors the SciFact interface (Claim/Abstract/dev_claims/abstracts).

Source: direct downloads from https://fever.ai/download/fever/ (the official
distribution). The HF `fever` dataset uses a deprecated loading script and no
longer works on `datasets >= 4.0`, so we fetch the JSONL claim files and
`wiki-pages.zip` corpus directly. Files are cached on disk and re-used.

The Wikipedia corpus dump has ~5.4M pages (~5 GB unzipped). We subset it: docs
cited by the evaluated dev claims are always kept, plus additional docs
reservoir-sampled up to a cap (`max_corpus_docs`, default 50k) to act as
BM25/dense negatives.

Label mapping:
  SUPPORTS        -> SUPPORT
  REFUTES         -> CONTRADICT
  NOT ENOUGH INFO -> NEI

Evidence format conversion:
  FEVER claim rows have `evidence: list[list[list[annotation_id, evidence_id,
  doc_title, sentence_id]]]`. We flatten to (annotator, doc, sentence) tuples
  and group by (annotator, doc_title) into per-doc {label, sentences} entries
  to match SciFact's `evidence: dict[doc_id, list[{label, sentences}]]` shape.

Default cache: $FEVER_CACHE_DIR if set, else /space_mounts/pars/fever_data.
"""

from __future__ import annotations

import json
import os
import random
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .scifact import Abstract, Claim


LABEL_MAP = {
    "SUPPORTS": "SUPPORT",
    "REFUTES": "CONTRADICT",
    "NOT ENOUGH INFO": "NEI",
}

# Cache on LFS by default — the wiki dump is ~5 GB unzipped.
_LFS_DEFAULT = Path("/space_mounts/pars/fever_data")
DEFAULT_CACHE_DIR = Path(os.environ.get("FEVER_CACHE_DIR", _LFS_DEFAULT))

DEFAULT_URLS = {
    "paper_dev":  "https://fever.ai/download/fever/paper_dev.jsonl",
    "train":      "https://fever.ai/download/fever/train.jsonl",
    "wiki_pages": "https://fever.ai/download/fever/wiki-pages.zip",
}


# ── Download helpers ──────────────────────────────────────────────────

def _download(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    """Download a URL to dest with a progress bar. Skip if already present."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    import requests
    from tqdm import tqdm

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) or None
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"⤓ {dest.name}",
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
    tmp.rename(dest)
    return dest


def _extract_wiki_pages(zip_path: Path, extract_dir: Path) -> Path:
    """Extract wiki-pages.zip if not already extracted. Returns the dir holding wiki-*.jsonl."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    sentinel = extract_dir / ".extracted"
    if sentinel.exists():
        # Find the inner directory containing wiki-*.jsonl.
        for sub in extract_dir.iterdir():
            if sub.is_dir() and any(sub.glob("wiki-*.jsonl")):
                return sub
        if any(extract_dir.glob("wiki-*.jsonl")):
            return extract_dir

    from tqdm import tqdm

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        for m in tqdm(members, desc="extract wiki-pages"):
            zf.extract(m, extract_dir)
    sentinel.touch()

    for sub in extract_dir.iterdir():
        if sub.is_dir() and any(sub.glob("wiki-*.jsonl")):
            return sub
    if any(extract_dir.glob("wiki-*.jsonl")):
        return extract_dir
    raise RuntimeError(f"No wiki-*.jsonl files found after extracting {zip_path}")


def _parse_lines(lines_field: str) -> list[str]:
    """Parse FEVER wiki `lines` column: `<sent_id>\\t<sentence>\\t<links>` per line."""
    if not lines_field:
        return []
    out: list[str] = []
    for raw in lines_field.split("\n"):
        if not raw:
            continue
        parts = raw.split("\t")
        if len(parts) >= 2:
            out.append(parts[1].strip())
        else:
            out.append("")
    return out


# ── Claim row normalisation ───────────────────────────────────────────

def _flatten_claim_evidence(claim_row: dict) -> list[dict]:
    """Yield one row per (annotator, doc, sent) evidence pointer for a single claim.

    The FEVER claim JSONL stores `evidence` as a nested list:
        evidence: list[annotation_set: list[evidence: [annot_id, evidence_id, doc, sent]]]

    We flatten this to the same flat-row shape that `_build_claims` consumes
    (one row per evidence pointer).
    """
    rows: list[dict] = []
    cid = claim_row["id"]
    claim = claim_row.get("claim", "")
    label = claim_row.get("label", "NOT ENOUGH INFO")

    nested = claim_row.get("evidence") or []
    seen_any = False
    for annotation_set in nested:
        if not annotation_set:
            continue
        for ev in annotation_set:
            # ev = [annotation_id, evidence_id, doc_title, sentence_id]
            if not isinstance(ev, list) or len(ev) < 4:
                continue
            annot_id, ev_id, doc, sent = ev[0], ev[1], ev[2], ev[3]
            seen_any = True
            rows.append({
                "id": cid,
                "claim": claim,
                "label": label,
                "evidence_annotation_id": annot_id if annot_id is not None else 0,
                "evidence_id": ev_id if ev_id is not None else 0,
                "evidence_wiki_url": doc or "",
                "evidence_sentence_id": sent if sent is not None else -1,
            })

    if not seen_any:
        # Emit a single row so NEI claims (no evidence) still appear.
        rows.append({
            "id": cid, "claim": claim, "label": label,
            "evidence_annotation_id": 0, "evidence_id": 0,
            "evidence_wiki_url": "", "evidence_sentence_id": -1,
        })
    return rows


def _iter_claims_jsonl(path: Path) -> Iterator[dict]:
    """Yield flat rows (one per evidence pointer) from a FEVER claims JSONL file."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield from _flatten_claim_evidence(json.loads(line))


def _iter_wiki_pages(wiki_dir: Path) -> Iterator[dict]:
    """Yield {id, text, lines} dicts streaming through all wiki-*.jsonl files."""
    files = sorted(wiki_dir.glob("wiki-*.jsonl"))
    if not files:
        raise RuntimeError(f"No wiki-*.jsonl files in {wiki_dir}")
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


# ── Fever class ───────────────────────────────────────────────────────

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
        include_train: bool = False,
        max_train_claims: int = 5_000,
        urls: Optional[dict[str, str]] = None,
    ) -> "Fever":
        """Load FEVER `paper_dev` claims (with evidence) and a subset of wiki pages.

        Args:
            max_dev_claims: cap on dev claims with evidence (SUPPORTS/REFUTES only).
            max_corpus_docs: cap on total wiki pages indexed (gold + sampled).
            cache_dir: where to cache downloads + extracted wiki pages.
                       Defaults to /space_mounts/pars/fever_data.
            seed: RNG seed for reservoir sampling of negative docs.
            include_train: also load train claims (used by KL fine-tuning).
            max_train_claims: cap on train claims when include_train=True.
            urls: override default fever.ai download URLs.
        """
        cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        url_map = {**DEFAULT_URLS, **(urls or {})}

        # ── Download claim files ────────────────────────────────────
        dev_path = _download(url_map["paper_dev"], cache_dir / "paper_dev.jsonl")

        train_path = None
        if include_train:
            train_path = _download(url_map["train"], cache_dir / "train.jsonl")

        # ── Download + extract wiki pages ───────────────────────────
        zip_path = _download(url_map["wiki_pages"], cache_dir / "wiki-pages.zip")
        wiki_dir = _extract_wiki_pages(zip_path, cache_dir / "wiki-pages")

        # ── Build claim objects ─────────────────────────────────────
        fv = cls()

        dev_rows = _iter_claims_jsonl(dev_path)
        fv.dev_claims, dev_titles = cls._build_claims(
            dev_rows, max_claims=max_dev_claims, require_evidence=True,
        )
        needed_titles: set[str] = set(dev_titles)

        if include_train and train_path is not None:
            train_rows = _iter_claims_jsonl(train_path)
            fv.train_claims, train_titles = cls._build_claims(
                train_rows, max_claims=max_train_claims, require_evidence=True,
            )
            needed_titles.update(train_titles)

        # ── Build corpus (gold + reservoir-sampled negatives) ───────
        rng = random.Random(seed)
        n_neg_target = max(0, max_corpus_docs - len(needed_titles))
        reservoir: list[Abstract] = []
        seen_neg = 0
        gold_seen: set[str] = set()

        from tqdm import tqdm

        for row in tqdm(_iter_wiki_pages(wiki_dir), desc="scan wiki pages"):
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
                continue

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
    def _build_claims(
        rows: Iterable[dict], max_claims: int, require_evidence: bool,
    ) -> tuple[list[Claim], set[str]]:
        """Group flat FEVER rows into Claim objects. Returns (claims, doc_titles_used).

        The cap is applied lazily — we stop reading rows once we've finalised
        `max_claims` distinct claims, so passing a generator is cheap even on
        a large train file.
        """
        # We can't finalise a claim until we've seen all its rows, so buffer by id.
        by_id: dict[int, dict] = {}
        finalised_order: list[int] = []
        finalised: set[int] = set()

        def finalise_one(cid: int) -> Claim | None:
            entry = by_id.get(cid)
            if entry is None:
                return None
            mapped = LABEL_MAP.get(entry["label"], "NEI")
            if require_evidence and mapped == "NEI":
                return None
            if require_evidence and not entry["groups"]:
                return None
            evidence: dict[str, list[dict]] = defaultdict(list)
            for (_, doc), sents in entry["groups"].items():
                evidence[doc].append({"label": mapped, "sentences": sorted(sents)})
            return Claim(
                claim_id=str(cid),
                text=entry["claim"],
                evidence=dict(evidence),
            )

        claims: list[Claim] = []
        used_titles: set[str] = set()

        for row in rows:
            cid = int(row["id"])
            if cid in finalised:
                continue

            entry = by_id.setdefault(cid, {
                "claim": row.get("claim", ""),
                "label": row.get("label") or "NOT ENOUGH INFO",
                "groups": defaultdict(set),
            })
            doc = row.get("evidence_wiki_url") or ""
            sent = row.get("evidence_sentence_id")
            annot = row.get("evidence_annotation_id")
            if annot is None:
                annot = row.get("evidence_id", 0)
            if doc and sent is not None and int(sent) >= 0:
                entry["groups"][(int(annot), doc)].add(int(sent))

            if cid not in finalised_order:
                finalised_order.append(cid)

        # JSONL is sorted by claim id, so once we've seen all rows we can finalise in id order.
        for cid in finalised_order:
            built = finalise_one(cid)
            if built is None:
                continue
            claims.append(built)
            for doc in built.evidence:
                used_titles.add(doc)
            if len(claims) >= max_claims:
                break

        return claims, used_titles

    # ── SciFact-compatible accessors ────────────────────────────────────

    def get_corpus_texts(self) -> dict[str, str]:
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
