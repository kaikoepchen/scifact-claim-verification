#!/usr/bin/env python3
"""Adaptive retrieval depth on SciFact.

We compare three retrieval strategies feeding the joint sentence model:
  a) FIXED-3   — always rerank to top-3
  b) FIXED-10  — always rerank to top-10
  c) ADAPTIVE  — rerank to top-3 when BM25/dense agree (Jaccard@10 >= t),
                 top-10 when they disagree

The adaptive policy spends extra compute (and possibly extra noise) only on
the claims where the two retrievers don't agree — the ones where the right
doc is more likely sitting outside top-3.

We sweep the disagreement threshold from 0.05 to 0.5 in 10 steps, and
report doc/sentence recall, verdict accuracy, and the average number of
docs retrieved per claim.

Run:
    python scripts/17_adaptive_retrieval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from claimverify.data.scifact import SciFact
from claimverify.retrieval.bm25 import BM25Retriever
from claimverify.retrieval.dense import DenseRetriever
from claimverify.retrieval.fusion import CrossEncoderReranker, ReciprocalRankFusion
from claimverify.retrieval.disagreement import jaccard_at_k
from claimverify.reasoning.joint import JointSentenceModel

console = Console()


BASE_K = 3
EXPANDED_K = 10


def joint_predict(joint, claim_text, retrieved, abstracts):
    """Run joint model on retrieved docs; return (verdicts, rationale_indices)."""
    doc_ids = [d for d, _ in retrieved]
    doc_sentences = {d: abstracts[d].sentences for d in doc_ids if d in abstracts}
    if not doc_sentences:
        return {}, {}
    doc_results = joint.predict_documents(claim_text, doc_sentences)
    verdicts: dict[str, str] = {}
    rationales: dict[str, list[int]] = {}
    for doc_id, result in doc_results.items():
        if result.label != "NEI" and result.rationale_indices:
            verdicts[doc_id] = result.label
            rationales[doc_id] = result.rationale_indices
    return verdicts, rationales


def compute_metrics(per_claim, pick):
    """Compute strategy metrics. `pick(c)` -> (retrieved_ids, verdicts, rationales)."""
    n_with_evidence = 0
    doc_hits = 0
    sent_tp = 0
    sent_fn = 0
    verdict_correct = 0
    n_docs_total = 0

    for c in per_claim:
        retrieved_ids, verdicts, rationales = pick(c)
        n_docs_total += len(retrieved_ids)

        if not c["evidence"]:
            continue
        n_with_evidence += 1

        retrieved_set = set(retrieved_ids)
        gold_doc_ids = set(c["evidence"].keys())
        if retrieved_set & gold_doc_ids:
            doc_hits += 1

        for doc_id, anns in c["evidence"].items():
            for ann in anns:
                gold = set(ann["sentences"])
                if doc_id in retrieved_set:
                    pred = set(rationales.get(doc_id, []))
                    sent_tp += len(gold & pred)
                    sent_fn += len(gold - pred)
                else:
                    sent_fn += len(gold)

        is_correct = False
        for doc_id, anns in c["evidence"].items():
            if doc_id in verdicts:
                pred_sents = set(rationales.get(doc_id, []))
                for ann in anns:
                    if verdicts[doc_id] == ann["label"] and set(ann["sentences"]).issubset(pred_sents):
                        is_correct = True
        if is_correct:
            verdict_correct += 1

    n_claims = len(per_claim)
    return {
        "n_claims": n_claims,
        "n_with_evidence": n_with_evidence,
        "doc_recall": round(doc_hits / n_with_evidence, 4) if n_with_evidence else 0.0,
        "sentence_recall": round(sent_tp / (sent_tp + sent_fn), 4) if (sent_tp + sent_fn) else 0.0,
        "verdict_accuracy": round(verdict_correct / n_with_evidence, 4) if n_with_evidence else 0.0,
        "avg_docs_per_claim": round(n_docs_total / n_claims, 4) if n_claims else 0.0,
    }


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    console.print("\n[bold blue]Adaptive Retrieval — SciFact[/bold blue]\n")

    console.print("Loading SciFact...")
    sf = SciFact.load()
    corpus = sf.get_corpus_texts()

    console.print("Building BM25 index...")
    bm25 = BM25Retriever(k1=1.2, b=0.75)
    bm25.build_index(corpus)

    console.print("Building dense index...")
    dense = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
    dense.build_index(corpus)

    rrf = ReciprocalRankFusion(k=60)

    console.print("Loading cross-encoder reranker...")
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    console.print("Loading joint model...")
    joint = JointSentenceModel(model_name="models/joint-scifact")

    console.print(f"\nRunning on {len(sf.dev_claims)} dev claims...\n")

    per_claim = []
    for claim in tqdm(sf.dev_claims, desc="Pipeline"):
        bm25_results = bm25.retrieve(claim.text, top_k=100)
        dense_results = dense.retrieve(claim.text, top_k=100)
        agreement = jaccard_at_k(bm25_results, dense_results, k=10)

        cand_3 = rrf.fuse(bm25_results, dense_results, top_k=BASE_K)
        ranked_3 = reranker.rerank(claim.text, cand_3, corpus, top_k=BASE_K)

        cand_10 = rrf.fuse(bm25_results, dense_results, top_k=EXPANDED_K)
        ranked_10 = reranker.rerank(claim.text, cand_10, corpus, top_k=EXPANDED_K)

        v3, r3 = joint_predict(joint, claim.text, ranked_3, sf.abstracts)
        v10, r10 = joint_predict(joint, claim.text, ranked_10, sf.abstracts)

        per_claim.append({
            "claim_id": claim.claim_id,
            "agreement": float(agreement),
            "evidence": claim.evidence,
            "retrieved_3": [d for d, _ in ranked_3],
            "retrieved_10": [d for d, _ in ranked_10],
            "verdicts_3": v3,
            "rationales_3": r3,
            "verdicts_10": v10,
            "rationales_10": r10,
        })

    fixed_3 = compute_metrics(
        per_claim,
        lambda c: (c["retrieved_3"], c["verdicts_3"], c["rationales_3"]),
    )
    fixed_10 = compute_metrics(
        per_claim,
        lambda c: (c["retrieved_10"], c["verdicts_10"], c["rationales_10"]),
    )

    def adaptive_pick(t):
        def pick(c):
            if c["agreement"] >= t:
                return c["retrieved_3"], c["verdicts_3"], c["rationales_3"]
            return c["retrieved_10"], c["verdicts_10"], c["rationales_10"]
        return pick

    thresholds = np.linspace(0.05, 0.5, 10)
    sweep = []
    for t in thresholds:
        m = compute_metrics(per_claim, adaptive_pick(float(t)))
        n_expanded = sum(1 for c in per_claim if c["agreement"] < t)
        m["threshold"] = round(float(t), 4)
        m["n_expanded"] = n_expanded
        m["frac_expanded"] = round(n_expanded / len(per_claim), 4)
        sweep.append(m)

    best = max(sweep, key=lambda r: r["verdict_accuracy"])

    sweep_table = Table(title="Adaptive — Threshold Sweep")
    sweep_table.add_column("Threshold", justify="right")
    sweep_table.add_column("Expanded", justify="right")
    sweep_table.add_column("Frac Exp", justify="right")
    sweep_table.add_column("Doc Rec", justify="right")
    sweep_table.add_column("Sent Rec", justify="right")
    sweep_table.add_column("Verdict Acc", justify="right")
    sweep_table.add_column("Avg Docs", justify="right")
    for row in sweep:
        marker = "  *" if row["threshold"] == best["threshold"] else "   "
        sweep_table.add_row(
            f"{row['threshold']:.3f}{marker}",
            f"{row['n_expanded']}",
            f"{row['frac_expanded']:.1%}",
            f"{row['doc_recall']:.4f}",
            f"{row['sentence_recall']:.4f}",
            f"{row['verdict_accuracy']:.4f}",
            f"{row['avg_docs_per_claim']:.2f}",
        )
    console.print(sweep_table)
    console.print(f"\n[bold]Best adaptive threshold (by verdict accuracy):[/bold] {best['threshold']:.3f}\n")

    cmp_table = Table(title="Strategy Comparison (SciFact dev)")
    cmp_table.add_column("Strategy", style="bold")
    cmp_table.add_column("Doc Recall", justify="right")
    cmp_table.add_column("Sent Recall", justify="right")
    cmp_table.add_column("Verdict Acc", justify="right")
    cmp_table.add_column("Avg Docs", justify="right")

    for name, m in [
        ("FIXED-3 (baseline)", fixed_3),
        ("FIXED-10 (upper bound)", fixed_10),
        (f"ADAPTIVE (t={best['threshold']:.3f})", best),
    ]:
        cmp_table.add_row(
            name,
            f"{m['doc_recall']:.4f}",
            f"{m['sentence_recall']:.4f}",
            f"{m['verdict_accuracy']:.4f}",
            f"{m['avg_docs_per_claim']:.2f}",
        )
    console.print(cmp_table)

    output = {
        "dataset": "scifact",
        "n_claims": len(per_claim),
        "base_top_k": BASE_K,
        "expanded_top_k": EXPANDED_K,
        "fixed_3": fixed_3,
        "fixed_10": fixed_10,
        "adaptive_sweep": sweep,
        "adaptive_best_threshold": best["threshold"],
        "adaptive_best": best,
    }
    out_path = results_dir / "17_adaptive_retrieval.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
