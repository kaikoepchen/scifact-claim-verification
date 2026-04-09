#!/usr/bin/env python3
"""Claim decomposition analysis on SciFact dev.

Compares retrieval quality with and without claim decomposition.
For compound claims, sub-claims are retrieved independently and
results are merged via RRF.

Run:
    python scripts/06_claim_decomposition.py
    python scripts/06_claim_decomposition.py --top-k 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table

from claimverify.data.scifact import SciFact
from claimverify.retrieval.bm25 import BM25Retriever
from claimverify.retrieval.dense import DenseRetriever
from claimverify.retrieval.fusion import ReciprocalRankFusion
from claimverify.preprocessing.decompose import ClaimDecomposer
from claimverify.evaluation.metrics import evaluate_retrieval

console = Console()


def retrieve_with_decomposition(
    claim_text, decomposer, bm25, dense, rrf, top_k=10,
):
    """Retrieve using decomposed sub-claims, merge with RRF."""
    decomposed = decomposer.decompose(claim_text)

    if not decomposed.is_compound:
        bm25_results = bm25.retrieve(claim_text, top_k=100)
        dense_results = dense.retrieve(claim_text, top_k=100)
        return rrf.fuse(bm25_results, dense_results, top_k=top_k), decomposed

    # Retrieve per sub-claim, then merge all results
    all_results = []
    for sub in decomposed.sub_claims:
        bm25_results = bm25.retrieve(sub, top_k=100)
        dense_results = dense.retrieve(sub, top_k=100)
        fused = rrf.fuse(bm25_results, dense_results, top_k=50)
        all_results.append(fused)

    # Merge sub-claim results with another round of RRF
    if len(all_results) == 1:
        return all_results[0][:top_k], decomposed

    merged = all_results[0]
    for other in all_results[1:]:
        merged = rrf.fuse(merged, other, top_k=top_k)

    return merged[:top_k], decomposed


def main():
    parser = argparse.ArgumentParser(description="Claim decomposition analysis")
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    console.print("\n[bold blue]Claim Decomposition Analysis on SciFact Dev[/bold blue]\n")

    # Load
    console.print("Loading SciFact...")
    sf = SciFact.load()
    corpus = sf.get_corpus_texts()
    qrels = sf.get_qrels(split="dev")

    # Build retrievers
    console.print("Building BM25 index...")
    bm25 = BM25Retriever(k1=1.2, b=0.75)
    bm25.build_index(corpus)

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
    decomposer = ClaimDecomposer()

    claims_with_evidence = [c for c in sf.dev_claims if c.evidence]
    queries = {c.claim_id: c.text for c in claims_with_evidence}

    # Baseline: retrieve without decomposition
    console.print("\nRetrieving without decomposition...")
    baseline_results = {}
    for cid, text in queries.items():
        bm25_r = bm25.retrieve(text, top_k=100)
        dense_r = dense.retrieve(text, top_k=100)
        baseline_results[cid] = rrf.fuse(bm25_r, dense_r, top_k=args.top_k)

    # With decomposition
    console.print("Retrieving with decomposition...")
    decomp_results = {}
    decomp_stats = {"compound": 0, "atomic": 0, "total_subclaims": 0}
    examples = []

    for cid, text in queries.items():
        results, decomposed = retrieve_with_decomposition(
            text, decomposer, bm25, dense, rrf, top_k=args.top_k
        )
        decomp_results[cid] = results

        if decomposed.is_compound:
            decomp_stats["compound"] += 1
            decomp_stats["total_subclaims"] += decomposed.n_parts
            if len(examples) < 10:
                examples.append({
                    "claim_id": cid,
                    "original": decomposed.original,
                    "sub_claims": decomposed.sub_claims,
                })
        else:
            decomp_stats["atomic"] += 1

    # Evaluate both
    baseline_metrics = evaluate_retrieval(baseline_results, qrels, ks=[5, 10])
    decomp_metrics = evaluate_retrieval(decomp_results, qrels, ks=[5, 10])

    # Print comparison
    table = Table(title="Retrieval: Baseline vs. Decomposed")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Decomposed", justify="right")
    table.add_column("Delta", justify="right")

    for metric in sorted(baseline_metrics.keys()):
        b = baseline_metrics[metric]
        d = decomp_metrics[metric]
        delta = d - b
        color = "green" if delta > 0.001 else ("red" if delta < -0.001 else "")
        table.add_row(
            metric,
            f"{b:.4f}",
            f"{d:.4f}",
            f"[{color}]{delta:+.4f}[/{color}]" if color else f"{delta:+.4f}",
        )

    console.print(table)

    # Decomposition stats
    console.print(f"\n  Claims evaluated: {len(queries)}")
    console.print(f"  Compound claims: {decomp_stats['compound']}")
    console.print(f"  Atomic claims: {decomp_stats['atomic']}")
    if decomp_stats["compound"] > 0:
        avg = decomp_stats["total_subclaims"] / decomp_stats["compound"]
        console.print(f"  Avg sub-claims per compound: {avg:.1f}")

    # Print examples
    if examples:
        console.print("\n[bold]Decomposition examples:[/bold]")
        for ex in examples[:5]:
            console.print(f"\n  [dim]{ex['claim_id']}[/dim] {ex['original'][:80]}")
            for i, sub in enumerate(ex["sub_claims"]):
                console.print(f"    -> [{i+1}] {sub[:80]}")

    # Save
    output = {
        "top_k": args.top_k,
        "n_claims": len(queries),
        "decomposition_stats": decomp_stats,
        "baseline_metrics": baseline_metrics,
        "decomposed_metrics": decomp_metrics,
        "delta": {k: decomp_metrics[k] - baseline_metrics[k] for k in baseline_metrics},
        "examples": examples,
    }
    out_path = results_dir / "06_claim_decomposition.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
