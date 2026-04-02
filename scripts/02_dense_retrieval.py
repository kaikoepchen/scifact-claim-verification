#!/usr/bin/env python3
"""
Phase 2: Dense retrieval + hybrid fusion on SciFact.

This script:
  1. Loads SciFact and builds dense index with BGE-M3 (or E5)
  2. Evaluates dense-only retrieval
  3. Runs RRF hybrid fusion (BM25 + dense)
  4. Optionally applies cross-encoder reranking
  5. Produces the 5-condition ablation table

Conditions:
  A. BM25 alone
  B. Dense alone
  C. Hybrid (BM25 + Dense, RRF)
  D. Hybrid + Cross-Encoder Reranker
  E. (Later) SPLADE + Dense + RRF

Run:
    python scripts/02_dense_retrieval.py
    python scripts/02_dense_retrieval.py --model intfloat/e5-base-v2
    python scripts/02_dense_retrieval.py --no-reranker
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table

from claimverify.data.scifact import SciFact
from claimverify.evaluation.metrics import evaluate_retrieval
from claimverify.retrieval.pipeline import RetrievalConfig, RetrievalPipeline

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Dense retrieval + hybrid fusion on SciFact")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Dense embedding model")
    parser.add_argument("--no-reranker", action="store_true", help="Skip cross-encoder reranking")
    parser.add_argument("--index-path", default="data/processed/dense_index", help="FAISS index path")
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    console.print("\n[bold blue]Phase 2: Dense Retrieval + Hybrid Fusion[/bold blue]\n")

    # --- Load data ---
    console.print("Loading SciFact...")
    sf = SciFact.load()
    corpus = sf.get_corpus_texts()
    queries = {c.claim_id: c.text for c in sf.dev_claims}
    qrels = sf.get_qrels(split="dev")
    console.print(f"  {sf.corpus_size} abstracts, {len(queries)} dev queries\n")

    # --- Build pipeline ---
    config = RetrievalConfig(
        dense_model=args.model,
        dense_index_path=args.index_path,
        reranker_enabled=not args.no_reranker,
    )
    pipeline = RetrievalPipeline(config)
    console.print(f"Building indexes (dense model: {args.model})...")
    pipeline.build(corpus)
    console.print("  Done.\n")

    # --- Run ablation conditions ---
    conditions = {
        "A: BM25 only": "bm25",
        "B: Dense only": "dense",
        "C: Hybrid (RRF)": "hybrid",
    }

    all_metrics = {}
    for label, mode in conditions.items():
        console.print(f"Evaluating {label}...")

        # For fair comparison, disable reranker for A/B/C
        orig_reranker = pipeline.reranker
        if mode != "hybrid" or True:  # We'll add D separately
            pipeline.reranker = None
        results = pipeline.batch_retrieve(queries, mode=mode, top_k=10)
        pipeline.reranker = orig_reranker

        metrics = evaluate_retrieval(results, qrels, ks=[5, 10, 20])
        all_metrics[label] = metrics

    # Condition D: Hybrid + Reranker
    if not args.no_reranker:
        console.print("Evaluating D: Hybrid + Reranker...")
        results = pipeline.batch_retrieve(queries, mode="hybrid", top_k=10)
        metrics = evaluate_retrieval(results, qrels, ks=[5, 10, 20])
        all_metrics["D: Hybrid + Reranker"] = metrics

    # --- Display ablation table ---
    console.print()
    table = Table(title="Retrieval Ablation on SciFact Dev")
    table.add_column("Condition", style="bold")
    metric_names = sorted(next(iter(all_metrics.values())).keys())
    for m in metric_names:
        table.add_column(m, justify="right")

    for label, metrics in all_metrics.items():
        row = [label] + [f"{metrics[m]:.4f}" for m in metric_names]
        table.add_row(*row)

    console.print(table)

    # --- Save ---
    output = {
        "phase": "02_dense_retrieval",
        "dense_model": args.model,
        "reranker": not args.no_reranker,
        "ablation": all_metrics,
    }
    out_path = results_dir / "02_dense_retrieval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
