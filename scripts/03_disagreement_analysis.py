#!/usr/bin/env python3
"""Retriever disagreement analysis on SciFact dev.

Computes sparse-dense disagreement signals and tests whether
disagreement predicts retrieval failures. This is the core
validation for the paper's main hypothesis.

Run:
    python scripts/03_disagreement_analysis.py
    python scripts/03_disagreement_analysis.py --model intfloat/e5-base-v2
    python scripts/03_disagreement_analysis.py --k 5
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
from claimverify.retrieval.disagreement import (
    DisagreementSignals,
    compute_disagreement,
    disagreement_retrieval_correlation,
)

console = Console()


def print_distribution(signals: list[DisagreementSignals]) -> None:
    """Print summary statistics for disagreement signals."""
    table = Table(title="Disagreement Signal Distribution")
    table.add_column("Signal", style="bold")
    table.add_column("Mean", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Q25", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Q75", justify="right")
    table.add_column("Max", justify="right")

    def quantiles(vals):
        s = sorted(vals)
        n = len(s)
        return {
            "mean": sum(s) / n,
            "min": s[0],
            "q25": s[n // 4],
            "median": s[n // 2],
            "q75": s[3 * n // 4],
            "max": s[-1],
        }

    for name, getter in [
        ("Jaccard@k", lambda s: s.jaccard_at_k),
        ("Rank correlation", lambda s: s.rank_correlation),
        ("Overlap count", lambda s: float(s.overlap_count)),
        ("Score margin (sparse)", lambda s: s.score_margin_sparse),
        ("Score margin (dense)", lambda s: s.score_margin_dense),
    ]:
        vals = [getter(s) for s in signals]
        q = quantiles(vals)
        table.add_row(
            name,
            f"{q['mean']:.3f}", f"{q['min']:.3f}", f"{q['q25']:.3f}",
            f"{q['median']:.3f}", f"{q['q75']:.3f}", f"{q['max']:.3f}",
        )

    console.print(table)

    top1_rate = sum(1 for s in signals if s.top1_same) / len(signals)
    console.print(f"\n  Top-1 agreement rate: {top1_rate:.1%}")


def print_correlation(corr: dict[str, float]) -> None:
    """Print the disagreement-failure correlation analysis."""
    table = Table(title="Disagreement vs. Retrieval Success")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Queries evaluated", f"{corr['n_evaluated']}")
    table.add_row("Median Jaccard@k", f"{corr['median_jaccard']:.3f}")
    table.add_row("Mean Jaccard@k", f"{corr['mean_jaccard']:.3f}")
    table.add_row("", "")
    table.add_row("[bold]High agreement group[/bold]", f"n={corr['n_high_agreement']}")
    table.add_row("  Retrieval success rate", f"{corr['success_rate_high_agreement']:.1%}")
    table.add_row("  Only-one-retriever rate", f"{corr['one_retriever_only_rate_agree']:.1%}")
    table.add_row("[bold]High disagreement group[/bold]", f"n={corr['n_high_disagreement']}")
    table.add_row("  Retrieval success rate", f"{corr['success_rate_high_disagreement']:.1%}")
    table.add_row("  Only-one-retriever rate", f"{corr['one_retriever_only_rate_disagree']:.1%}")
    table.add_row("", "")
    table.add_row("[bold]Success gap (agree - disagree)[/bold]",
                  f"[{'green' if corr['success_gap'] > 0 else 'red'}]"
                  f"{corr['success_gap']:+.1%}[/]")
    table.add_row("Pearson(jaccard, success)", f"{corr['pearson_jaccard_success']:.3f}")
    table.add_row("Top-1 agreement rate", f"{corr['top1_agreement_rate']:.1%}")

    console.print(table)


def print_examples(
    signals: list[DisagreementSignals],
    qrels: dict[str, dict[str, int]],
    queries: dict[str, str],
    n: int = 5,
) -> None:
    """Print example claims with highest and lowest disagreement."""
    evaluated = [s for s in signals if s.query_id in qrels]
    by_jaccard = sorted(evaluated, key=lambda s: s.jaccard_at_k)

    console.print("\n[bold red]Highest disagreement claims (lowest Jaccard):[/bold red]")
    for s in by_jaccard[:n]:
        claim_text = queries.get(s.query_id, "?")[:80].encode("ascii", "replace").decode()
        console.print(f"  [dim]{s.query_id}[/dim] J={s.jaccard_at_k:.3f} rho={s.rank_correlation:.3f} "
                      f"overlap={s.overlap_count} | {claim_text}")

    console.print("\n[bold green]Highest agreement claims (highest Jaccard):[/bold green]")
    for s in by_jaccard[-n:]:
        claim_text = queries.get(s.query_id, "?")[:80].encode("ascii", "replace").decode()
        console.print(f"  [dim]{s.query_id}[/dim] J={s.jaccard_at_k:.3f} rho={s.rank_correlation:.3f} "
                      f"overlap={s.overlap_count} | {claim_text}")


def main():
    parser = argparse.ArgumentParser(description="Retriever disagreement analysis")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Dense embedding model")
    parser.add_argument("--k", type=int, default=10, help="Top-k for disagreement computation")
    parser.add_argument("--index-path", default=None, help="FAISS index path")
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    console.print("\n[bold blue]Retriever Disagreement Analysis on SciFact Dev[/bold blue]\n")

    # --- Load data ---
    console.print("Loading SciFact...")
    sf = SciFact.load()
    corpus = sf.get_corpus_texts()
    queries = {c.claim_id: c.text for c in sf.dev_claims}
    qrels = sf.get_qrels(split="dev")
    console.print(f"  {sf.corpus_size} abstracts, {len(queries)} dev queries, "
                  f"{len(qrels)} with relevance judgments\n")

    # --- Build retrievers ---
    console.print("Building BM25 index...")
    bm25 = BM25Retriever(k1=1.2, b=0.75)
    bm25.build_index(corpus)

    console.print(f"Building dense index ({args.model})...")
    dense = DenseRetriever(model_name=args.model)
    if args.index_path:
        try:
            dense.load_index(args.index_path)
            console.print("  Loaded cached index.")
        except (FileNotFoundError, RuntimeError):
            dense.build_index(corpus, save_path=args.index_path)
    else:
        dense.build_index(corpus)

    # --- Retrieve ---
    top_k = max(args.k, 100)  # retrieve more than k for fair scoring
    console.print(f"\nRetrieving top-{top_k} with both retrievers...")
    sparse_results = bm25.batch_retrieve(queries, top_k=top_k)
    dense_results = dense.batch_retrieve(queries, top_k=top_k)

    # --- Compute disagreement ---
    console.print(f"\nComputing disagreement signals (k={args.k})...\n")
    signals = compute_disagreement(sparse_results, dense_results, qrels, k=args.k)

    print_distribution(signals)

    # --- Correlation analysis ---
    console.print()
    corr = disagreement_retrieval_correlation(
        signals, qrels, sparse_results, dense_results, k=args.k
    )
    print_correlation(corr)

    # --- Examples ---
    print_examples(signals, qrels, queries)

    # --- Save ---
    output = {
        "phase": "03_disagreement_analysis",
        "dense_model": args.model,
        "k": args.k,
        "n_queries": len(queries),
        "n_with_qrels": len(qrels),
        "distribution": {
            "mean_jaccard": corr["mean_jaccard"],
            "median_jaccard": corr["median_jaccard"],
            "top1_agreement_rate": corr["top1_agreement_rate"],
        },
        "correlation": corr,
        "per_query": [
            {
                "query_id": s.query_id,
                "jaccard": s.jaccard_at_k,
                "rank_correlation": s.rank_correlation,
                "overlap_count": s.overlap_count,
                "top1_same": s.top1_same,
                "score_margin_sparse": s.score_margin_sparse,
                "score_margin_dense": s.score_margin_dense,
                "rr_sparse": s.min_rr_sparse,
                "rr_dense": s.min_rr_dense,
            }
            for s in signals
        ],
    }
    out_path = results_dir / "03_disagreement_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\nResults saved to {out_path}")

    # --- Verdict ---
    console.print("\n[bold]Interpretation:[/bold]")
    if corr["success_gap"] > 0.02:
        console.print("[green]  Signal is promising: high-agreement queries have higher "
                      "retrieval success than high-disagreement queries.[/green]")
    elif corr["success_gap"] > -0.02:
        console.print("[yellow]  Signal is weak: no clear difference between agreement groups.[/yellow]")
    else:
        console.print("[red]  Signal is inverted: disagreement correlates with BETTER retrieval. "
                      "This may indicate complementary retrieval rather than uncertainty.[/red]")


if __name__ == "__main__":
    main()
