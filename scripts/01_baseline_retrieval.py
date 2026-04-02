#!/usr/bin/env python3
"""
Phase 1: BM25 baseline on SciFact.

This is the first script to run. It:
  1. Loads SciFact corpus and claims from HuggingFace
  2. Builds BM25 index on the 5,183-abstract corpus
  3. Evaluates BM25 on dev claims (Recall@5, Recall@10, nDCG@10, MRR)
  4. Compares against BEIR published baseline (nDCG@10 ≈ 0.665)
  5. Saves results to results/01_bm25_baseline.json

Run:
    python scripts/01_baseline_retrieval.py
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table

from claimverify.data.scifact import SciFact
from claimverify.evaluation.metrics import evaluate_retrieval
from claimverify.retrieval.bm25 import BM25Retriever

console = Console()

BEIR_BM25_NDCG10 = 0.665  # Published BEIR BM25 baseline on SciFact


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # --- Load SciFact ---
    console.print("\n[bold blue]Phase 1: BM25 Baseline on SciFact[/bold blue]\n")
    console.print("Loading SciFact dataset...")
    sf = SciFact.load()
    console.print(
        f"  Corpus: {sf.corpus_size} abstracts | "
        f"Claims: {sf.num_claims}"
    )

    # --- Build BM25 Index ---
    console.print("\nBuilding BM25 index...")
    corpus = sf.get_corpus_texts()
    bm25 = BM25Retriever(k1=1.2, b=0.75)
    bm25.build_index(corpus)
    console.print(f"  Indexed {bm25.N} documents, avg length {bm25.avgdl:.1f} tokens")

    # --- Prepare queries ---
    queries = {c.claim_id: c.text for c in sf.dev_claims}
    qrels = sf.get_qrels(split="dev")
    console.print(f"  Evaluating on {len(queries)} dev claims ({len(qrels)} with evidence)\n")

    # --- Retrieve ---
    console.print("Running BM25 retrieval...")
    results = bm25.batch_retrieve(queries, top_k=100)

    # --- Evaluate ---
    metrics = evaluate_retrieval(results, qrels, ks=[5, 10, 20])

    # --- Display ---
    table = Table(title="BM25 Retrieval Results on SciFact Dev")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("BEIR Baseline", justify="right", style="dim")
    table.add_column("Delta", justify="right")

    for name, value in sorted(metrics.items()):
        beir_ref = f"{BEIR_BM25_NDCG10:.3f}" if name == "ndcg@10" else "—"
        delta = ""
        if name == "ndcg@10":
            d = value - BEIR_BM25_NDCG10
            color = "green" if d >= 0 else "red"
            delta = f"[{color}]{d:+.3f}[/{color}]"
        table.add_row(name, f"{value:.4f}", beir_ref, delta)

    console.print(table)

    # --- Save ---
    output = {
        "phase": "01_bm25_baseline",
        "dataset": "SciFact",
        "split": "dev",
        "num_queries": len(queries),
        "num_queries_with_qrels": len(qrels),
        "corpus_size": sf.corpus_size,
        "bm25_params": {"k1": 1.2, "b": 0.75},
        "metrics": metrics,
        "beir_bm25_ndcg10": BEIR_BM25_NDCG10,
    }
    out_path = results_dir / "01_bm25_baseline.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
