"""NLI-based verdict prediction for claim-evidence pairs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABEL_MAP = {0: "CONTRADICT", 1: "NOT_ENOUGH_INFO", 2: "SUPPORT"}


@dataclass
class VerdictResult:
    label: str
    confidence: float
    logits: dict[str, float]


class VerdictPredictor:
    """Predicts SUPPORT / CONTRADICT / NOT_ENOUGH_INFO for a claim-evidence pair.

    Uses an NLI model fine-tuned on scientific text. The default model
    (SciFact RoBERTa) maps the NLI entailment/contradiction/neutral labels
    to our verdict labels.
    """

    def __init__(
        self,
        model_name: str = "roberta-large-mnli",
        device: str | None = None,
        label_map: dict[int, str] | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # MNLI label order: contradiction=0, neutral=1, entailment=2
        self.label_map = label_map or LABEL_MAP

    @torch.no_grad()
    def predict(self, claim: str, evidence: str) -> VerdictResult:
        """Predict verdict for a single claim-evidence pair."""
        inputs = self.tokenizer(
            claim,
            evidence,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

        pred_idx = int(torch.argmax(probs))
        label = self.label_map[pred_idx]
        confidence = float(probs[pred_idx])

        logits = {self.label_map[i]: float(probs[i]) for i in range(len(probs))}

        return VerdictResult(label=label, confidence=confidence, logits=logits)

    @torch.no_grad()
    def predict_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[VerdictResult]:
        """Predict verdicts for multiple (claim, evidence) pairs."""
        if not pairs:
            return []

        claims, evidences = zip(*pairs)
        inputs = self.tokenizer(
            list(claims),
            list(evidences),
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)

        results = []
        for i in range(len(pairs)):
            pred_idx = int(torch.argmax(probs[i]))
            label = self.label_map[pred_idx]
            confidence = float(probs[i][pred_idx])
            logits = {self.label_map[j]: float(probs[i][j]) for j in range(probs.shape[1])}
            results.append(VerdictResult(label=label, confidence=confidence, logits=logits))

        return results
