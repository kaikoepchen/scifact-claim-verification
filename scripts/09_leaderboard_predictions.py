#!/usr/bin/env python3
"""Generate SciFact leaderboard predictions.

Runs the full pipeline and outputs predictions in the AllenAI
SciFact evaluator format (JSONL). Can target dev or test split.

Supports two modes:
  --pipeline separate  (default legacy): cosine rationale selector + NLI verdict
  --pipeline joint:     joint sentence-level model for both tasks

Run:
    python scripts/09_leaderboard_predictions.py
    python scripts/09_leaderboard_predictions.py --pipeline joint --joint-model models/joint-scifact
    python scripts/09_leaderboard_predictions.py --split test --pipeline joint
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
from claimverify.reasoning.joint import JointSentenceModel
from claimverify.evaluation.leaderboard import (
    format_prediction,
    write_predictions,
    evaluate_against_gold,
)

console = Console()


def predict_separate(claim, retrieved_ids, sf, selector, predictor):
    """Legacy pipeline: separate rationale selection + verdict prediction."""
    doc_sentences = {}
    for doc_id in retrieved_ids:
        if doc_id in sf.abstracts:
            doc_sentences[doc_id] = sf.abstracts[doc_id].sentences
    rationales = selector.select_from_docs(claim.text, doc_sentences)

    doc_verdicts = {}
    for doc_id in retrieved_ids:
        if doc_id not in sf.abstracts:
            continue
        verdict = predictor.predict(claim.text, sf.abstracts[doc_id].text)
        doc_verdicts[doc_id] = verdict.label

    doc_rationale_idxs = {}
    for doc_id, scored_sents in rationales.items():
        doc_rationale_idxs[doc_id] = [s.sentence_idx for s in scored_sents]

    return doc_verdicts, doc_rationale_idxs


def predict_joint(claim, retrieved_ids, sf, joint_model):
    """Joint pipeline: single model does rationale selection + verdict."""
    doc_sentences = {}
    for doc_id in retrieved_ids:
        if doc_id in sf.abstracts:
            doc_sentences[doc_id] = sf.abstracts[doc_id].sentences

    doc_results = joint_model.predict_documents(claim.text, doc_sentences)

    doc_verdicts = {}
    doc_rationale_idxs = {}
    for doc_id, result in doc_results.items():
        if result.label != "NEI" and result.rationale_indices:
            doc_verdicts[doc_id] = result.label
            doc_rationale_idxs[doc_id] = result.rationale_indices

    return doc_verdicts, doc_rationale_idxs


def main():
    parser = argparse.ArgumentParser(description="Generate leaderboard predictions")
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--mode", default="hybrid", choices=["bm25", "dense", "hybrid"])
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default=None)

    # Pipeline selection
    parser.add_argument("--pipeline", default="joint", choices=["separate", "joint"])

    # Separate pipeline args
    parser.add_argument("--nli-model", default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")

    # Joint pipeline args
    parser.add_argument("--joint-model", default="models/joint-scifact")

    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_path = args.output or str(results_dir / f"predictions_{args.split}.jsonl")

    console.print(f"\n[bold blue]Generating Leaderboard Predictions ({args.split})[/bold blue]")
    console.print(f"  Pipeline: {args.pipeline}\n")

    # Load data
    console.print("Loading SciFact...")
    sf = SciFact.load()
    corpus = sf.get_corpus_texts()

    claims = {"dev": sf.dev_claims, "test": sf.test_claims}[args.split]
    console.print(f"  {len(claims)} {args.split} claims\n")

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

    rrf = ReciprocalRankFusion(k=60)

    # Load reasoning models
    selector = None
    predictor = None
    joint_model = None

    if args.pipeline == "separate":
        console.print("Loading rationale selector...")
        selector = RationaleSelector(max_sentences_per_doc=3)
        console.print(f"Loading NLI model ({args.nli_model})...")
        predictor = VerdictPredictor(model_name=args.nli_model)
    else:
        console.print(f"Loading joint model ({args.joint_model})...")
        joint_model = JointSentenceModel(model_name=args.joint_model)

    # Generate predictions
    console.print(f"\nRunning pipeline on {len(claims)} claims...\n")
    predictions = []

    for claim in tqdm(claims, desc="Predicting"):
        # Retrieve
        if args.mode == "bm25":
            retrieved = bm25.retrieve(claim.text, top_k=args.top_k)
        elif args.mode == "dense":
            retrieved = dense.retrieve(claim.text, top_k=args.top_k)
        else:
            bm25_r = bm25.retrieve(claim.text, top_k=100)
            dense_r = dense.retrieve(claim.text, top_k=100)
            retrieved = rrf.fuse(bm25_r, dense_r, top_k=args.top_k)

        retrieved_ids = [doc_id for doc_id, _ in retrieved]

        # Run selected pipeline
        if args.pipeline == "separate":
            doc_verdicts, doc_rationale_idxs = predict_separate(
                claim, retrieved_ids, sf, selector, predictor,
            )
        else:
            doc_verdicts, doc_rationale_idxs = predict_joint(
                claim, retrieved_ids, sf, joint_model,
            )

        pred = format_prediction(claim.claim_id, doc_verdicts, doc_rationale_idxs)
        predictions.append(pred)

    # Write predictions
    write_predictions(predictions, output_path)
    console.print(f"\nPredictions written to {output_path}")

    # Stats
    n_with_evidence = sum(1 for p in predictions if p["evidence"])
    n_support = sum(
        1 for p in predictions
        for d in p["evidence"].values()
        if d["label"] == "SUPPORT"
    )
    n_contradict = sum(
        1 for p in predictions
        for d in p["evidence"].values()
        if d["label"] == "CONTRADICT"
    )

    table = Table(title="Prediction Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total claims", f"{len(predictions)}")
    table.add_row("Claims with evidence", f"{n_with_evidence}")
    table.add_row("Claims without evidence", f"{len(predictions) - n_with_evidence}")
    table.add_row("SUPPORT predictions", f"{n_support}")
    table.add_row("CONTRADICT predictions", f"{n_contradict}")
    console.print(table)

    # If dev split, evaluate against gold
    if args.split == "dev":
        gold = []
        for claim in sf.dev_claims:
            evidence = {}
            for doc_id, anns in claim.evidence.items():
                for ann in anns:
                    evidence[doc_id] = {
                        "sentences": ann["sentences"],
                        "label": ann["label"],
                    }
            gold.append({"id": int(claim.claim_id), "evidence": evidence})

        metrics = evaluate_against_gold(predictions, gold)

        eval_table = Table(title="Dev Evaluation (Abstract-Level)")
        eval_table.add_column("Metric", style="bold")
        eval_table.add_column("Value", justify="right")
        eval_table.add_row("Precision", f"{metrics['abstract_precision']:.3f}")
        eval_table.add_row("Recall", f"{metrics['abstract_recall']:.3f}")
        eval_table.add_row("F1", f"{metrics['abstract_f1']:.3f}")
        eval_table.add_row("True positives", f"{metrics['true_positives']}")
        eval_table.add_row("Total predicted", f"{metrics['total_predicted']}")
        eval_table.add_row("Total gold", f"{metrics['total_gold']}")
        console.print(eval_table)

        # Save metrics
        metrics_path = results_dir / "09_leaderboard_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump({**metrics, "pipeline": args.pipeline}, f, indent=2)
        console.print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
