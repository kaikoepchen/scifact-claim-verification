#!/usr/bin/env python3
"""Explanation generation evaluation on SciFact dev.

Runs the full pipeline: retrieve -> select rationales -> predict verdict
-> generate cited explanation -> evaluate citation fidelity.

Run:
    python scripts/08_generation.py
    python scripts/08_generation.py --method template
    python scripts/08_generation.py --method extractive --top-k 5
    python scripts/08_generation.py --method llm
    python scripts/08_generation.py --method llm --llm-model google/gemma-4-E4B
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
from claimverify.generation.citation import build_citation_context
from claimverify.generation.generator import ExplanationGenerator
from claimverify.evaluation.metrics import citation_quality

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Explanation generation evaluation")
    parser.add_argument("--method", default="extractive",
                        choices=["template", "extractive", "llm"])
    parser.add_argument("--llm-model", default="google/gemma-4-E4B",
                        help="LLM model for --method llm")
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--nli-model", default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    console.print("\n[bold blue]Explanation Generation Evaluation[/bold blue]\n")

    # Load data
    console.print("Loading SciFact...")
    sf = SciFact.load()
    corpus = sf.get_corpus_texts()

    # Build retriever
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

    console.print("Loading rationale selector...")
    selector = RationaleSelector(max_sentences_per_doc=3)

    console.print(f"Loading NLI model ({args.nli_model})...")
    predictor = VerdictPredictor(model_name=args.nli_model)

    llm_model = args.llm_model if args.method == "llm" else None
    generator = ExplanationGenerator(method=args.method, llm_model=llm_model)

    # Run pipeline
    claims_with_evidence = [c for c in sf.dev_claims if c.evidence]
    console.print(f"\nGenerating explanations for {len(claims_with_evidence)} claims...\n")

    citation_data = []
    examples = []

    for claim in tqdm(claims_with_evidence, desc="Generating"):
        bm25_results = bm25.retrieve(claim.text, top_k=100)
        dense_results = dense.retrieve(claim.text, top_k=100)
        retrieved = rrf.fuse(bm25_results, dense_results, top_k=args.top_k)
        retrieved_ids = [doc_id for doc_id, _ in retrieved]

        # Select rationales
        doc_sentences = {}
        for doc_id in retrieved_ids:
            if doc_id in sf.abstracts:
                doc_sentences[doc_id] = sf.abstracts[doc_id].sentences
        rationales = selector.select_from_docs(claim.text, doc_sentences)

        # Predict verdict using full abstract
        doc_verdicts = {}
        for doc_id in retrieved_ids:
            if doc_id not in sf.abstracts:
                continue
            verdict = predictor.predict(claim.text, sf.abstracts[doc_id].text)
            doc_verdicts[doc_id] = verdict

        agg = aggregate_verdicts(doc_verdicts)

        # Build citation context from rationale sentences
        doc_sents_for_citation = {}
        for doc_id, scored_sents in rationales.items():
            doc_sents_for_citation[doc_id] = [
                (s.sentence_idx, s.text, s.score) for s in scored_sents
            ]

        ctx = build_citation_context(claim.text, agg.label, doc_sents_for_citation)
        explanation = generator.generate(ctx)

        citation_data.append({
            "cited_refs": explanation.cited_refs,
            "available_refs": ctx.get_ref_ids(),
            "verdict": explanation.verdict,
        })

        if len(examples) < 10:
            examples.append({
                "claim_id": claim.claim_id,
                "claim": claim.text,
                "verdict": explanation.verdict,
                "explanation": explanation.text,
                "n_evidence": len(ctx.evidence),
                "n_cited": len(explanation.cited_refs),
            })

    # Evaluate citation quality
    cq = citation_quality(citation_data)

    table = Table(title="Citation Quality Metrics")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Citation precision", f"{cq['citation_precision']:.3f}")
    table.add_row("Citation recall", f"{cq['citation_recall']:.3f}")
    table.add_row("Unsupported citation rate", f"{cq['unsupported_rate']:.3f}")
    table.add_row("Empty citation rate (non-NEI)", f"{cq['empty_citation_rate']:.3f}")
    table.add_row("", "")
    table.add_row("Claims evaluated", f"{len(claims_with_evidence)}")
    table.add_row("Generation method", args.method)

    console.print(table)

    # Print examples
    console.print("\n[bold]Example explanations:[/bold]")
    for ex in examples[:5]:
        console.print(f"\n  [dim]{ex['claim_id']}[/dim] {ex['claim'][:80]}")
        console.print(f"  Verdict: {ex['verdict']} | Evidence: {ex['n_evidence']} | Cited: {ex['n_cited']}")
        for line in ex["explanation"].split("\n")[:4]:
            console.print(f"    {line[:100]}")

    # Save
    output = {
        "method": args.method,
        "llm_model": args.llm_model if args.method == "llm" else None,
        "nli_model": args.nli_model,
        "top_k": args.top_k,
        "n_claims": len(claims_with_evidence),
        "citation_quality": cq,
        "examples": examples,
    }
    out_path = results_dir / "08_generation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
