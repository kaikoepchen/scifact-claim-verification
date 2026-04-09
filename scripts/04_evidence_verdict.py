#!/usr/bin/env python3
"""Phase 4: Evidence selection + verdict prediction on SciFact dev.

Runs the full pipeline: retrieve -> select rationale sentences ->
predict NLI verdict -> aggregate per-claim -> evaluate against gold labels.

Run:
    python scripts/04_evidence_verdict.py
    python scripts/04_evidence_verdict.py --nli-model roberta-large-mnli
    python scripts/04_evidence_verdict.py --mode bm25
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
from claimverify.reasoning.rationale import RationaleSelector
from claimverify.reasoning.verdict import VerdictPredictor
from claimverify.reasoning.aggregation import aggregate_verdicts
from claimverify.evaluation.metrics import macro_f1

console = Console()


def build_retriever(args, corpus):
    """Build the retrieval pipeline based on mode."""
    bm25 = BM25Retriever(k1=1.2, b=0.75)
    bm25.build_index(corpus)

    if args.mode == "bm25":
        return bm25, None

    dense = DenseRetriever(model_name=args.dense_model)
    if args.index_path:
        try:
            dense.load_index(args.index_path)
            console.print("  Loaded cached dense index.")
        except (FileNotFoundError, RuntimeError):
            dense.build_index(corpus, save_path=args.index_path)
    else:
        dense.build_index(corpus)

    return bm25, dense


def retrieve_for_claim(bm25, dense, query, mode, top_k=10):
    """Retrieve documents for a single claim."""
    if mode == "bm25":
        return bm25.retrieve(query, top_k=top_k)
    elif mode == "dense":
        return dense.retrieve(query, top_k=top_k)
    else:
        bm25_results = bm25.retrieve(query, top_k=100)
        dense_results = dense.retrieve(query, top_k=100)
        rrf = ReciprocalRankFusion(k=60)
        return rrf.fuse(bm25_results, dense_results, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Evidence + Verdict")
    parser.add_argument("--mode", default="hybrid", choices=["bm25", "dense", "hybrid"])
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--nli-model", default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--top-k", type=int, default=5, help="Documents to retrieve per claim")
    parser.add_argument("--max-sents", type=int, default=3, help="Max rationale sentences per doc")
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    console.print("\n[bold blue]Phase 4: Evidence Selection + Verdict Prediction[/bold blue]\n")

    # Load data
    console.print("Loading SciFact...")
    sf = SciFact.load()
    corpus = sf.get_corpus_texts()
    gold_labels = sf.get_verdict_labels(split="dev")
    console.print(f"  {sf.corpus_size} abstracts, {len(sf.dev_claims)} dev claims, "
                  f"{len(gold_labels)} gold verdict labels\n")

    # Build retriever
    console.print(f"Building retriever ({args.mode})...")
    bm25, dense = build_retriever(args, corpus)

    # Load models
    console.print("Loading rationale selector...")
    selector = RationaleSelector(max_sentences_per_doc=args.max_sents)

    console.print(f"Loading NLI model ({args.nli_model})...")
    predictor = VerdictPredictor(model_name=args.nli_model)

    # Build gold lookup: (claim_id, doc_id) -> label
    gold_lookup = {}
    for g in gold_labels:
        key = (g["claim_id"], g["doc_id"])
        gold_lookup[key] = g["label"]

    # Run pipeline on claims that have gold evidence
    claims_with_evidence = [c for c in sf.dev_claims if c.evidence]
    console.print(f"\nRunning on {len(claims_with_evidence)} claims with gold evidence...\n")

    predictions = []
    gold_list = []
    per_claim_results = []

    for claim in tqdm(claims_with_evidence, desc="Evaluating"):
        retrieved = retrieve_for_claim(bm25, dense, claim.text, args.mode, top_k=args.top_k)
        retrieved_doc_ids = [doc_id for doc_id, _ in retrieved]

        # Collect sentences from retrieved docs
        doc_sentences = {}
        for doc_id in retrieved_doc_ids:
            if doc_id in sf.abstracts:
                doc_sentences[doc_id] = sf.abstracts[doc_id].sentences

        # Select rationale sentences
        rationales = selector.select_from_docs(claim.text, doc_sentences)

        # Predict verdict per document using full abstract text
        doc_verdicts = {}
        for doc_id in retrieved_doc_ids:
            if doc_id not in sf.abstracts:
                continue
            evidence_text = sf.abstracts[doc_id].text
            verdict = predictor.predict(claim.text, evidence_text)
            doc_verdicts[doc_id] = verdict

        # Aggregate
        agg = aggregate_verdicts(doc_verdicts)

        # Match against gold labels for this claim
        for doc_id, annotations in claim.evidence.items():
            for ann in annotations:
                gold_label = ann["label"]
                # Map SciFact labels to our labels
                if gold_label == "SUPPORT":
                    gold_mapped = "SUPPORT"
                elif gold_label == "CONTRADICT":
                    gold_mapped = "CONTRADICT"
                else:
                    gold_mapped = "NOT_ENOUGH_INFO"

                # Did we retrieve this doc?
                if doc_id in doc_verdicts:
                    pred_label = doc_verdicts[doc_id].label
                else:
                    pred_label = "NOT_ENOUGH_INFO"

                predictions.append(pred_label)
                gold_list.append(gold_mapped)

        per_claim_results.append({
            "claim_id": claim.claim_id,
            "claim_text": claim.text,
            "retrieved_docs": retrieved_doc_ids,
            "n_rationales": sum(len(v) for v in rationales.values()),
            "aggregated_label": agg.label,
            "aggregated_confidence": round(agg.confidence, 4),
            "has_conflict": agg.has_conflict,
            "per_doc": {
                doc_id: {
                    "label": v.label,
                    "confidence": round(v.confidence, 4),
                    "logits": {k: round(val, 4) for k, val in v.logits.items()},
                }
                for doc_id, v in doc_verdicts.items()
            },
        })

    # Evaluate
    console.print()
    eval_result = macro_f1(predictions, gold_list, classes=["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"])

    table = Table(title="Verdict Prediction Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Accuracy", f"{eval_result['accuracy']:.3f}")
    table.add_row("Macro-F1", f"{eval_result['macro_f1']:.3f}")
    table.add_row("", "")

    for cls in ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]:
        if cls in eval_result["per_class"]:
            pc = eval_result["per_class"][cls]
            table.add_row(f"  {cls} precision", f"{pc['precision']:.3f}")
            table.add_row(f"  {cls} recall", f"{pc['recall']:.3f}")
            table.add_row(f"  {cls} F1", f"{pc['f1']:.3f}")
            table.add_row("", "")

    n_conflicts = sum(1 for r in per_claim_results if r["has_conflict"])
    table.add_row("Claims evaluated", f"{len(per_claim_results)}")
    table.add_row("Evidence conflicts detected", f"{n_conflicts}")

    console.print(table)

    # Save
    output = {
        "phase": "04_evidence_verdict",
        "mode": args.mode,
        "nli_model": args.nli_model,
        "dense_model": args.dense_model,
        "top_k": args.top_k,
        "max_sents_per_doc": args.max_sents,
        "n_claims": len(per_claim_results),
        "n_predictions": len(predictions),
        "metrics": {
            "accuracy": eval_result["accuracy"],
            "macro_f1": eval_result["macro_f1"],
            "per_class": eval_result["per_class"],
        },
        "n_conflicts": n_conflicts,
        "per_claim": per_claim_results,
    }
    out_path = results_dir / "04_evidence_verdict.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
