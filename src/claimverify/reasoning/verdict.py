"""NLI-based verdict prediction for claim-evidence pairs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NLI_LABEL_MAPS = {
    "roberta-large-mnli": {0: "CONTRADICT", 1: "NOT_ENOUGH_INFO", 2: "SUPPORT"},
    "default": {0: "SUPPORT", 1: "NOT_ENOUGH_INFO", 2: "CONTRADICT"},
}


@dataclass
class VerdictResult:
    label: str
    confidence: float
    logits: dict[str, float]


class VerdictPredictor:
    """Predicts SUPPORT / CONTRADICT / NOT_ENOUGH_INFO for a claim-evidence pair.

    Reads the model's id2label config to automatically map NLI labels
    (entailment/contradiction/neutral) to verdict labels.
    """

    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        device: str | None = None,
        label_map: dict[int, str] | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        if label_map:
            self.label_map = label_map
        elif model_name in NLI_LABEL_MAPS:
            self.label_map = NLI_LABEL_MAPS[model_name]
        else:
            self.label_map = self._build_label_map()

    def _build_label_map(self) -> dict[int, str]:
        """Infer label map from the model's config."""
        nli_to_verdict = {
            "entailment": "SUPPORT",
            "contradiction": "CONTRADICT",
            "neutral": "NOT_ENOUGH_INFO",
        }
        id2label = self.model.config.id2label
        return {int(k): nli_to_verdict.get(v.lower(), v.upper()) for k, v in id2label.items()}

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
