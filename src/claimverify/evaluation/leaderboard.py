"""SciFact leaderboard prediction formatting and evaluation.

Converts pipeline outputs to the AllenAI SciFact leaderboard format:
    {"id": <claim_id>, "evidence": {<doc_id>: {"sentences": [...], "label": "..."}}}
"""

from __future__ import annotations

import json
from pathlib import Path


def format_prediction(
    claim_id: int | str,
    doc_verdicts: dict[str, str],
    doc_rationales: dict[str, list[int]],
    max_sentences: int = 3,
) -> dict:
    """Format a single claim prediction for leaderboard submission.

    Args:
        claim_id: The claim ID (integer).
        doc_verdicts: {doc_id: "SUPPORT" | "CONTRADICT"} per document.
        doc_rationales: {doc_id: [sentence_indices]} per document.
        max_sentences: Max rationale sentences per doc (leaderboard only uses first 3).
    """
    evidence = {}
    for doc_id, label in doc_verdicts.items():
        if label == "NOT_ENOUGH_INFO":
            continue
        sentences = doc_rationales.get(doc_id, [])[:max_sentences]
        if not sentences:
            continue
        evidence[doc_id] = {
            "sentences": sorted(sentences),
            "label": label,
        }

    return {"id": int(claim_id), "evidence": evidence}


def write_predictions(predictions: list[dict], path: str | Path) -> None:
    """Write predictions in JSONL format for leaderboard submission."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for pred in sorted(predictions, key=lambda p: p["id"]):
            f.write(json.dumps(pred) + "\n")


def load_predictions(path: str | Path) -> list[dict]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(path) as f:
        for line in f:
            predictions.append(json.loads(line.strip()))
    return predictions


def evaluate_against_gold(
    predictions: list[dict],
    gold: list[dict],
) -> dict[str, float]:
    """Evaluate predictions against gold labels (abstract-level).

    Implements the SciFact leaderboard evaluation:
    - Abstract is correct if label matches AND at least one gold rationale
      set is a subset of predicted sentences.
    - Reports precision, recall, F1 at abstract level.
    """
    gold_by_id = {}
    for g in gold:
        cid = g["id"] if "id" in g else int(g.get("claim_id", 0))
        gold_by_id[cid] = g.get("evidence", {})

    pred_by_id = {p["id"]: p.get("evidence", {}) for p in predictions}

    tp = 0
    total_pred = 0
    total_gold = 0

    for cid, gold_evidence in gold_by_id.items():
        pred_evidence = pred_by_id.get(cid, {})

        for doc_id, gold_annotations in gold_evidence.items():
            total_gold += 1

            if doc_id not in pred_evidence:
                continue

            pred_doc = pred_evidence[doc_id]
            gold_label = gold_annotations["label"] if isinstance(gold_annotations, dict) else gold_annotations[0]["label"]
            pred_label = pred_doc["label"]

            if pred_label != gold_label:
                continue

            # Check if predicted sentences contain a gold rationale set
            pred_sents = set(pred_doc["sentences"])
            if isinstance(gold_annotations, dict):
                gold_sent_sets = [set(gold_annotations["sentences"])]
            else:
                gold_sent_sets = [set(ann["sentences"]) for ann in gold_annotations]

            if any(gs.issubset(pred_sents) for gs in gold_sent_sets):
                tp += 1

        for doc_id in pred_evidence:
            total_pred += 1

    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gold if total_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "abstract_precision": precision,
        "abstract_recall": recall,
        "abstract_f1": f1,
        "true_positives": tp,
        "total_predicted": total_pred,
        "total_gold": total_gold,
    }
