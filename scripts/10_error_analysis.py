#!/usr/bin/env python3
"""Error analysis on SciFact dev predictions.

Breaks down errors by type: retrieval failures, verdict errors,
rationale misses. Analyzes how retriever disagreement correlates
with different failure modes.

Run:
    python scripts/10_error_analysis.py
    python scripts/10_error_analysis.py --nli-model models/verdict-scifact
"""

import argparse
import json
import sys
from collections import Counter
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

console = Console()


def classify_error(gold_doc_retrieved, verdict_correct, rationale_overlap):
    """Classify the type of error for a single claim-doc pair."""
    if not gold_doc_retrieved:
        return "retrieval_miss"
    if not verdict_correct:
        return "wrong_verdict"
    if rationale_overlap == 0:
        return "rationale_miss"
    return "correct"


def main():
    parser = argparse.ArgumentParser(description="Error analysis")
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--nli-model", default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    console.print("\n[bold blue]Error Analysis on SciFact Dev[/bold blue]\n")

    sf = SciFact.load()
    corpus = sf.get_corpus_texts()

    console.print("Building retrievers...")
    bm25 = BM25Retriever(k1=1.2, b=0.75)
    bm25.build_index(corpus)

    dense = DenseRetriever(model_name=args.dense_model)
    if args.index_path:
        try:
            dense.load_index(args.index_path)
        except (FileNotFoundError, RuntimeError):
            dense.build_index(corpus, save_path=args.index_path)
    else:
        dense.build_index(corpus)

    rrf = ReciprocalRankFusion(k=60)
    selector = RationaleSelector(max_sentences_per_doc=3)

    console.print(f"Loading NLI model ({args.nli_model})...")
    predictor = VerdictPredictor(model_name=args.nli_model)

    claims_with_evidence = [c for c in sf.dev_claims if c.evidence]
    console.print(f"\nAnalyzing {len(claims_with_evidence)} claims...\n")

    error_types = Counter()
    errors_by_label = {"SUPPORT": Counter(), "CONTRADICT": Counter()}
    disagreement_by_error = {"correct": [], "retrieval_miss": [], "wrong_verdict": [], "rationale_miss": []}
    claim_length_by_error = {"correct": [], "retrieval_miss": [], "wrong_verdict": [], "rationale_miss": []}
    detailed_errors = []

    for claim in tqdm(claims_with_evidence, desc="Analyzing"):
        bm25_results = bm25.retrieve(claim.text, top_k=100)
        dense_results = dense.retrieve(claim.text, top_k=100)
        retrieved = rrf.fuse(bm25_results, dense_results, top_k=args.top_k)
        retrieved_ids = set(doc_id for doc_id, _ in retrieved)

        agreement = jaccard_at_k(bm25_results, dense_results, k=10)
        claim_words = len(claim.text.split())

        for doc_id, annotations in claim.evidence.items():
            for ann in annotations:
                gold_label = ann["label"]
                gold_sents = set(ann["sentences"])

                gold_doc_retrieved = doc_id in retrieved_ids

                # Predict verdict
                verdict_correct = False
                rationale_overlap = 0

                if gold_doc_retrieved and doc_id in sf.abstracts:
                    v = predictor.predict(claim.text, sf.abstracts[doc_id].text)
                    pred_label = v.label
                    verdict_correct = (pred_label == gold_label)

                    # Check rationale overlap
                    doc_sentences = sf.abstracts[doc_id].sentences
                    rationales = selector.select(claim.text, doc_id, doc_sentences)
                    pred_sents = set(s.sentence_idx for s in rationales)
                    rationale_overlap = len(pred_sents & gold_sents)

                error_type = classify_error(gold_doc_retrieved, verdict_correct, rationale_overlap)
                error_types[error_type] += 1
                errors_by_label[gold_label][error_type] += 1
                disagreement_by_error[error_type].append(agreement)
                claim_length_by_error[error_type].append(claim_words)

                if error_type != "correct" and len(detailed_errors) < 30:
                    detailed_errors.append({
                        "claim_id": claim.claim_id,
                        "claim_text": claim.text[:100],
                        "gold_doc": doc_id,
                        "gold_label": gold_label,
                        "error_type": error_type,
                        "retriever_agreement": round(agreement, 3),
                    })

    # Print error breakdown
    total = sum(error_types.values())
    table = Table(title="Error Breakdown")
    table.add_column("Error Type", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("Fraction", justify="right")

    for etype in ["correct", "retrieval_miss", "wrong_verdict", "rationale_miss"]:
        count = error_types[etype]
        table.add_row(etype, f"{count}", f"{count/total:.1%}")
    table.add_row("", "", "")
    table.add_row("[bold]Total[/bold]", f"{total}", "100%")
    console.print(table)

    # Error by gold label
    label_table = Table(title="Errors by Gold Label")
    label_table.add_column("Error Type", style="bold")
    label_table.add_column("SUPPORT", justify="right")
    label_table.add_column("CONTRADICT", justify="right")

    for etype in ["correct", "retrieval_miss", "wrong_verdict", "rationale_miss"]:
        label_table.add_row(
            etype,
            f"{errors_by_label['SUPPORT'][etype]}",
            f"{errors_by_label['CONTRADICT'][etype]}",
        )
    console.print(label_table)

    # Disagreement vs error type
    disagree_table = Table(title="Retriever Agreement by Error Type")
    disagree_table.add_column("Error Type", style="bold")
    disagree_table.add_column("Mean Jaccard@10", justify="right")
    disagree_table.add_column("N", justify="right")

    for etype in ["correct", "retrieval_miss", "wrong_verdict", "rationale_miss"]:
        vals = disagreement_by_error[etype]
        if vals:
            disagree_table.add_row(etype, f"{sum(vals)/len(vals):.3f}", f"{len(vals)}")
        else:
            disagree_table.add_row(etype, "-", "0")
    console.print(disagree_table)

    # Claim length vs error type
    length_table = Table(title="Claim Length by Error Type")
    length_table.add_column("Error Type", style="bold")
    length_table.add_column("Mean Words", justify="right")

    for etype in ["correct", "retrieval_miss", "wrong_verdict", "rationale_miss"]:
        vals = claim_length_by_error[etype]
        if vals:
            length_table.add_row(etype, f"{sum(vals)/len(vals):.1f}")
        else:
            length_table.add_row(etype, "-")
    console.print(length_table)

    # Print example errors
    console.print("\n[bold]Example errors:[/bold]")
    for ex in detailed_errors[:10]:
        console.print(
            f"\n  [{ex['error_type']}] {ex['claim_text']}"
            f"\n    Gold: {ex['gold_label']} (doc {ex['gold_doc']}) | Agreement: {ex['retriever_agreement']}"
        )

    # Save
    output = {
        "top_k": args.top_k,
        "n_claims": len(claims_with_evidence),
        "total_annotations": total,
        "error_breakdown": dict(error_types),
        "error_by_label": {
            label: dict(counts) for label, counts in errors_by_label.items()
        },
        "disagreement_by_error": {
            etype: {
                "mean": sum(vals) / len(vals) if vals else 0,
                "n": len(vals),
            }
            for etype, vals in disagreement_by_error.items()
        },
        "claim_length_by_error": {
            etype: {
                "mean": sum(vals) / len(vals) if vals else 0,
                "n": len(vals),
            }
            for etype, vals in claim_length_by_error.items()
        },
        "detailed_errors": detailed_errors,
    }
    out_path = results_dir / "10_error_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
