#!/usr/bin/env python3
"""Abstention evaluation: run the full pipeline with calibrated abstention.

Retrieves evidence, predicts verdicts, computes uncertainty signals,
and evaluates the coverage-vs-accuracy trade-off.

Run:
    python scripts/05_abstention.py
    python scripts/05_abstention.py --threshold 0.35
    python scripts/05_abstention.py --mode bm25 --no-conflict-override
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from claimverify.data.scifact import SciFact
from claimverify.retrieval.bm25 import BM25Retriever
from claimverify.retrieval.dense import DenseRetriever
from claimverify.retrieval.fusion import ReciprocalRankFusion
from claimverify.retrieval.disagreement import jaccard_at_k
from claimverify.reasoning.rationale import RationaleSelector
from claimverify.reasoning.verdict import VerdictPredictor
from claimverify.reasoning.aggregation import aggregate_verdicts
from claimverify.calibration.signals import extract_signals
from claimverify.calibration.gate import AbstentionGate
from claimverify.calibration.tuning import coverage_risk_curve, find_optimal_threshold, auc_coverage_risk
from claimverify.evaluation.metrics import macro_f1

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Abstention evaluation")
    parser.add_argument("--mode", default="hybrid", choices=["bm25", "dense", "hybrid"])
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--nli-model", default="roberta-large-mnli")
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--no-conflict-override", action="store_true")
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    console.print("\n[bold blue]Abstention Evaluation on SciFact Dev[/bold blue]\n")

    # Load data
    console.print("Loading SciFact...")
    sf = SciFact.load()
    corpus = sf.get_corpus_texts()

    # Build retrievers
    console.print("Building BM25 index...")
    bm25 = BM25Retriever(k1=1.2, b=0.75)
    bm25.build_index(corpus)

    dense = None
    if args.mode != "bm25":
        console.print(f"Building dense index ({args.dense_model})...")
        dense = DenseRetriever(model_name=args.dense_model)
        if args.index_path:
            try:
                dense.load_index(args.index_path)
            except (FileNotFoundError, RuntimeError):
                dense.build_index(corpus, save_path=args.index_path)
        else:
            dense.build_index(corpus)

    console.print("Loading rationale selector...")
    selector = RationaleSelector(max_sentences_per_doc=3)

    console.print(f"Loading NLI model ({args.nli_model})...")
    predictor = VerdictPredictor(model_name=args.nli_model)

    gate = AbstentionGate(
        threshold=args.threshold,
        conflict_override=not args.no_conflict_override,
    )

    rrf = ReciprocalRankFusion(k=60)

    # Run pipeline
    claims_with_evidence = [c for c in sf.dev_claims if c.evidence]
    console.print(f"\nEvaluating {len(claims_with_evidence)} claims...\n")

    all_signals = []
    all_predictions = []
    all_gold = []
    per_claim = []

    for claim in tqdm(claims_with_evidence, desc="Pipeline"):
        # Retrieve
        bm25_results = bm25.retrieve(claim.text, top_k=100)
        if args.mode == "bm25":
            retrieved = bm25_results[:args.top_k]
            agreement = 1.0
        elif args.mode == "dense":
            retrieved = dense.retrieve(claim.text, top_k=args.top_k)
            agreement = 1.0
        else:
            dense_results = dense.retrieve(claim.text, top_k=100)
            retrieved = rrf.fuse(bm25_results, dense_results, top_k=args.top_k)
            agreement = jaccard_at_k(bm25_results, dense_results, k=10)

        retrieved_ids = [doc_id for doc_id, _ in retrieved]
        top_score = retrieved[0][1] if retrieved else 0.0

        # Select rationales
        doc_sentences = {}
        for doc_id in retrieved_ids:
            if doc_id in sf.abstracts:
                doc_sentences[doc_id] = sf.abstracts[doc_id].sentences
        rationales = selector.select_from_docs(claim.text, doc_sentences)

        # Predict verdicts per doc
        doc_verdicts = {}
        for doc_id, sents in rationales.items():
            evidence_text = " ".join(s.text for s in sents)
            verdict = predictor.predict(claim.text, evidence_text)
            doc_verdicts[doc_id] = verdict

        # Aggregate
        agg = aggregate_verdicts(doc_verdicts)

        # Extract uncertainty signals
        signals = extract_signals(
            claim_id=claim.claim_id,
            nli_logits={"SUPPORT": agg.support_score, "CONTRADICT": agg.contradict_score,
                        "NOT_ENOUGH_INFO": agg.nei_score},
            retrieval_score=top_score,
            retriever_agreement=agreement,
            evidence_count=agg.evidence_count,
            has_conflict=agg.has_conflict,
        )
        decision = gate.decide(signals)

        # Match to gold
        for doc_id, annotations in claim.evidence.items():
            for ann in annotations:
                gold_label = "SUPPORT" if ann["label"] == "SUPPORT" else "CONTRADICT"
                if doc_id in doc_verdicts:
                    pred_label = doc_verdicts[doc_id].label
                else:
                    pred_label = "NOT_ENOUGH_INFO"

                all_signals.append(signals)
                all_predictions.append(pred_label)
                all_gold.append(gold_label)

        per_claim.append({
            "claim_id": claim.claim_id,
            "aggregated_label": agg.label,
            "confidence_score": round(signals.combined_score, 4),
            "gate_decision": decision.action,
            "gate_reason": decision.reason,
            "retriever_agreement": round(agreement, 4),
            "has_conflict": agg.has_conflict,
            "evidence_count": agg.evidence_count,
        })

    # Overall metrics (no abstention)
    eval_all = macro_f1(all_predictions, all_gold,
                        classes=["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"])

    # Metrics with abstention
    answered_preds = []
    answered_gold = []
    for s, p, g in zip(all_signals, all_predictions, all_gold):
        d = gate.decide(s)
        if d.action == "answer":
            answered_preds.append(p)
            answered_gold.append(g)

    eval_answered = macro_f1(answered_preds, answered_gold,
                             classes=["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]) if answered_preds else {}

    # Coverage-risk curve
    curve = coverage_risk_curve(all_signals, all_gold, all_predictions, n_thresholds=50)
    optimal = find_optimal_threshold(curve, min_coverage=0.5)
    auc = auc_coverage_risk(curve)

    # Print results
    table = Table(title="Abstention Evaluation Results")
    table.add_column("Metric", style="bold")
    table.add_column("No Abstention", justify="right")
    table.add_column("With Abstention", justify="right")

    table.add_row("Accuracy",
                  f"{eval_all['accuracy']:.3f}",
                  f"{eval_answered.get('accuracy', 0):.3f}")
    table.add_row("Macro-F1",
                  f"{eval_all['macro_f1']:.3f}",
                  f"{eval_answered.get('macro_f1', 0):.3f}")
    table.add_row("Coverage",
                  "100%",
                  f"{len(answered_preds)/len(all_predictions)*100:.1f}%")
    table.add_row("", "", "")
    table.add_row("AUC (coverage-risk)", f"{auc}", "")
    table.add_row("Optimal threshold (≥50% cov.)",
                  f"{optimal.get('threshold', '-')}", "")
    table.add_row("Optimal accuracy",
                  f"{optimal.get('accuracy', '-')}", "")
    table.add_row("Optimal coverage",
                  f"{optimal.get('coverage', '-')}", "")

    n_answer = sum(1 for r in per_claim if r["gate_decision"] == "answer")
    n_abstain = sum(1 for r in per_claim if r["gate_decision"] == "abstain")
    n_conflict = sum(1 for r in per_claim if r["gate_decision"] == "flag_conflict")

    table.add_row("", "", "")
    table.add_row("Claims answered", f"{n_answer}", "")
    table.add_row("Claims abstained", f"{n_abstain}", "")
    table.add_row("Claims flagged (conflict)", f"{n_conflict}", "")

    console.print(table)

    # Save
    output = {
        "mode": args.mode,
        "nli_model": args.nli_model,
        "threshold": args.threshold,
        "n_claims": len(per_claim),
        "metrics_all": {
            "accuracy": eval_all["accuracy"],
            "macro_f1": eval_all["macro_f1"],
            "per_class": eval_all["per_class"],
        },
        "metrics_with_abstention": {
            "accuracy": eval_answered.get("accuracy", 0),
            "macro_f1": eval_answered.get("macro_f1", 0),
            "coverage": len(answered_preds) / len(all_predictions) if all_predictions else 0,
        },
        "gate_stats": {
            "n_answer": n_answer,
            "n_abstain": n_abstain,
            "n_conflict": n_conflict,
        },
        "coverage_risk_curve": curve,
        "optimal_threshold": optimal,
        "auc_coverage_risk": auc,
        "per_claim": per_claim,
    }
    out_path = results_dir / "05_abstention.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
