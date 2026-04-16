"""Joint sentence-level rationale selection and verdict prediction.

A single model classifies (claim, sentence) pairs into three classes:
  - SUPPORT:   this sentence is evidence supporting the claim
  - CONTRADICT: this sentence is evidence contradicting the claim
  - NEI:       this sentence is not relevant evidence

This jointly solves rationale selection (anything not NEI) and verdict
prediction, avoiding the error cascade of separate cosine-similarity
selection followed by abstract-level NLI.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


JOINT_LABEL2ID = {"SUPPORT": 0, "CONTRADICT": 1, "NEI": 2}
JOINT_ID2LABEL = {0: "SUPPORT", 1: "CONTRADICT", 2: "NEI"}


@dataclass
class SentenceVerdict:
    """Result for a single (claim, sentence) pair."""
    doc_id: str
    sentence_idx: int
    text: str
    label: str          # SUPPORT, CONTRADICT, or NEI
    confidence: float
    logits: dict[str, float]

    @property
    def is_rationale(self) -> bool:
        return self.label != "NEI"


@dataclass
class JointDocResult:
    """Aggregated result for one document."""
    doc_id: str
    label: str
    confidence: float
    rationale_indices: list[int]
    sentence_verdicts: list[SentenceVerdict]


class JointSentenceModel:
    """Joint rationale selection + verdict prediction at sentence level.

    Replaces both the cosine-similarity RationaleSelector and the
    abstract-level VerdictPredictor with a single fine-tuned model that
    classifies each (claim, sentence) pair.
    """

    def __init__(
        self,
        model_name: str = "models/joint-scifact",
        device: str | None = None,
        nei_threshold: float = 0.5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.nei_threshold = nei_threshold

        self.label_map = JOINT_ID2LABEL

    @torch.no_grad()
    def predict_sentences(
        self,
        claim: str,
        doc_id: str,
        sentences: list[str],
        batch_size: int = 32,
    ) -> list[SentenceVerdict]:
        """Classify all sentences in a document against the claim."""
        if not sentences:
            return []

        results = []
        for start in range(0, len(sentences), batch_size):
            batch_sents = sentences[start : start + batch_size]
            batch_idxs = list(range(start, start + len(batch_sents)))

            inputs = self.tokenizer(
                [claim] * len(batch_sents),
                batch_sents,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

            for i, (sent_idx, sent_text) in enumerate(zip(batch_idxs, batch_sents)):
                pred_idx = int(torch.argmax(probs[i]))
                label = self.label_map[pred_idx]
                confidence = float(probs[i][pred_idx])
                logits = {
                    self.label_map[j]: float(probs[i][j])
                    for j in range(probs.shape[1])
                }

                results.append(SentenceVerdict(
                    doc_id=doc_id,
                    sentence_idx=sent_idx,
                    text=sent_text,
                    label=label,
                    confidence=confidence,
                    logits=logits,
                ))

        return results

    def predict_document(
        self,
        claim: str,
        doc_id: str,
        sentences: list[str],
    ) -> JointDocResult:
        """Predict rationale sentences and aggregate verdict for one document."""
        sentence_verdicts = self.predict_sentences(claim, doc_id, sentences)

        rationale_verdicts = [sv for sv in sentence_verdicts if sv.is_rationale]

        if not rationale_verdicts:
            return JointDocResult(
                doc_id=doc_id,
                label="NEI",
                confidence=0.0,
                rationale_indices=[],
                sentence_verdicts=sentence_verdicts,
            )

        # Aggregate: confidence-weighted vote across rationale sentences
        support_score = sum(sv.logits.get("SUPPORT", 0.0) for sv in rationale_verdicts)
        contradict_score = sum(sv.logits.get("CONTRADICT", 0.0) for sv in rationale_verdicts)
        n = len(rationale_verdicts)
        support_score /= n
        contradict_score /= n

        if support_score >= contradict_score:
            label = "SUPPORT"
            confidence = support_score
        else:
            label = "CONTRADICT"
            confidence = contradict_score

        rationale_indices = sorted(sv.sentence_idx for sv in rationale_verdicts)

        return JointDocResult(
            doc_id=doc_id,
            label=label,
            confidence=confidence,
            rationale_indices=rationale_indices,
            sentence_verdicts=sentence_verdicts,
        )

    def predict_documents(
        self,
        claim: str,
        docs: dict[str, list[str]],
    ) -> dict[str, JointDocResult]:
        """Predict rationale + verdict for multiple retrieved documents."""
        return {
            doc_id: self.predict_document(claim, doc_id, sentences)
            for doc_id, sentences in docs.items()
        }
