#!/usr/bin/env python3
"""Evaluate abstention with the joint sentence model.

The core question: does retriever disagreement help us know when to
abstain, and does abstaining actually improve accuracy on answered claims?

We measure:
1. Accuracy on ALL claims vs. accuracy on ANSWERED claims (with abstention)
2. Whether retriever disagreement correlates with prediction correctness
3. Ablation: abstention with vs. without the disagreement signal
4. The coverage-accuracy trade-off curve

Run:
    python scripts/11_abstention_joint.py
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
from claimverify.retrieval.fusion import ReciprocalRankFusion
from claimverify.retrieval.disagreement import jaccard_at_k
from claimverify.reasoning.joint import JointSentenceModel
from claimverify.calibration.signals import UncertaintySignals, extract_signals
from claimverify.calibration.tuning import coverage_risk_curve, find_optimal_threshold, auc_coverage_risk
from claimverify.evaluation.leaderboard import format_prediction, evaluate_against_gold

console = Console()


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    console.print("\n[bold blue]Abstention Evaluation with Joint Model[/bold blue]\n")

    # Load data
    console.print("Loading SciFact...")
    sf = SciFact.load()
    corpus = sf.get_corpus_texts()

    # Build both retrievers (needed for disagreement signal)
    console.print("Building BM25 index...")
    bm25 = BM25Retriever(k1=1.2, b=0.75)
    bm25.build_index(corpus)

    console.print("Building dense index...")
    dense = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
    dense.build_index(corpus)

    rrf = ReciprocalRankFusion(k=60)

    console.print("Loading joint model...")
    joint = JointSentenceModel(model_name="models/joint-scifact")

    top_k = 3

    # Build gold labels for leaderboard eval
    gold = []
    for claim in sf.dev_claims:
        evidence = {}
        for doc_id, anns in claim.evidence.items():
            for ann in anns:
                evidence[doc_id] = {"sentences": ann["sentences"], "label": ann["label"]}
        gold.append({"id": int(claim.claim_id), "evidence": evidence})

    # ── Run pipeline on all dev claims ──────────────────────────────
    console.print(f"\nRunning on {len(sf.dev_claims)} dev claims...\n")

    per_claim = []

    for claim in tqdm(sf.dev_claims, desc="Pipeline"):
        # Retrieve with both retrievers
        bm25_results = bm25.retrieve(claim.text, top_k=100)
        dense_results = dense.retrieve(claim.text, top_k=100)
        retrieved = rrf.fuse(bm25_results, dense_results, top_k=top_k)
        agreement = jaccard_at_k(bm25_results, dense_results, k=10)

        retrieved_ids = [doc_id for doc_id, _ in retrieved]
        top_score = retrieved[0][1] if retrieved else 0.0

        # Joint prediction
        doc_sentences = {}
        for doc_id in retrieved_ids:
            if doc_id in sf.abstracts:
                doc_sentences[doc_id] = sf.abstracts[doc_id].sentences

        doc_results = joint.predict_documents(claim.text, doc_sentences)

        # Build prediction in leaderboard format
        doc_verdicts = {}
        doc_rationale_idxs = {}
        all_rationale_verdicts = []
        for doc_id, result in doc_results.items():
            if result.label != "NEI" and result.rationale_indices:
                doc_verdicts[doc_id] = result.label
                doc_rationale_idxs[doc_id] = result.rationale_indices
            for sv in result.sentence_verdicts:
                if sv.is_rationale:
                    all_rationale_verdicts.append(sv)

        pred = format_prediction(claim.claim_id, doc_verdicts, doc_rationale_idxs)

        # Compute NLI-like confidence from sentence verdicts
        if all_rationale_verdicts:
            support_score = np.mean([sv.logits.get("SUPPORT", 0) for sv in all_rationale_verdicts])
            contradict_score = np.mean([sv.logits.get("CONTRADICT", 0) for sv in all_rationale_verdicts])
            nei_score = np.mean([sv.logits.get("NEI", 0) for sv in all_rationale_verdicts])
        else:
            support_score, contradict_score, nei_score = 0.0, 0.0, 1.0

        signals = extract_signals(
            claim_id=claim.claim_id,
            nli_logits={"SUPPORT": support_score, "CONTRADICT": contradict_score,
                        "NOT_ENOUGH_INFO": nei_score},
            retrieval_score=top_score,
            retriever_agreement=agreement,
            evidence_count=len(doc_verdicts),
            has_conflict=False,
        )

        # Check correctness against gold (per-document)
        is_correct = False
        gold_docs = claim.evidence
        for doc_id, anns in gold_docs.items():
            if doc_id in doc_verdicts:
                for ann in anns:
                    if doc_verdicts[doc_id] == ann["label"]:
                        pred_sents = set(doc_rationale_idxs.get(doc_id, []))
                        gold_sents = set(ann["sentences"])
                        if gold_sents.issubset(pred_sents):
                            is_correct = True

        per_claim.append({
            "claim_id": claim.claim_id,
            "prediction": pred,
            "combined_score": signals.combined_score,
            "retriever_agreement": agreement,
            "nli_confidence": signals.nli_confidence,
            "nli_margin": signals.nli_margin,
            "evidence_count": len(doc_verdicts),
            "is_correct": is_correct,
            "has_evidence": bool(claim.evidence),
            "signals": signals,
        })

    # ── Evaluate: All claims vs. Abstained ──────────────────────────
    claims_with_evidence = [c for c in per_claim if c["has_evidence"]]
    total = len(claims_with_evidence)
    correct_all = sum(1 for c in claims_with_evidence if c["is_correct"])
    acc_all = correct_all / total if total > 0 else 0

    console.print(f"\n[bold]Baseline (no abstention):[/bold]")
    console.print(f"  Claims with evidence: {total}")
    console.print(f"  Correct: {correct_all}")
    console.print(f"  Accuracy: {acc_all:.3f}\n")

    # ── Sweep thresholds on combined_score ───────────────────────────
    scores = np.array([c["combined_score"] for c in claims_with_evidence])
    correctness = np.array([c["is_correct"] for c in claims_with_evidence])

    console.print("[bold]Coverage-Accuracy Curve (combined score with disagreement):[/bold]")
    table = Table()
    table.add_column("Threshold", justify="right")
    table.add_column("Answered", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Acc Gain", justify="right")

    curve_data = []
    best_gain = 0
    best_row = None

    for threshold in np.linspace(0, float(scores.max()) + 0.01, 30):
        mask = scores >= threshold
        n_answered = int(mask.sum())
        if n_answered == 0:
            continue
        n_correct = int(correctness[mask].sum())
        acc = n_correct / n_answered
        coverage = n_answered / total
        gain = acc - acc_all

        point = {
            "threshold": round(float(threshold), 4),
            "n_answered": n_answered,
            "n_correct": n_correct,
            "accuracy": round(acc, 4),
            "coverage": round(coverage, 4),
            "accuracy_gain": round(gain, 4),
        }
        curve_data.append(point)

        if gain > best_gain and coverage >= 0.5:
            best_gain = gain
            best_row = point

        table.add_row(
            f"{threshold:.3f}", f"{n_answered}", f"{n_correct}",
            f"{acc:.3f}", f"{coverage:.1%}",
            f"{'+' if gain > 0 else ''}{gain:.3f}",
        )

    console.print(table)

    # ── Ablation: disagreement signal vs. no disagreement ────────────
    console.print(f"\n[bold]Ablation: Retriever Disagreement Signal[/bold]\n")

    # Recompute scores WITHOUT retriever agreement (set to 1.0 = "always agree")
    scores_no_disagree = []
    for c in claims_with_evidence:
        s = c["signals"]
        fake_signals = UncertaintySignals(
            claim_id=s.claim_id,
            nli_confidence=s.nli_confidence,
            nli_margin=s.nli_margin,
            retrieval_score=s.retrieval_score,
            retriever_agreement=1.0,  # Pretend retrievers always agree
            evidence_count=s.evidence_count,
            has_conflict=s.has_conflict,
        )
        scores_no_disagree.append(fake_signals.combined_score)

    scores_no_disagree = np.array(scores_no_disagree)

    # Find best threshold for each variant at >=50% coverage
    def best_accuracy_at_coverage(score_arr, correct_arr, min_cov=0.5):
        best_acc = 0
        best_t = 0
        best_cov = 1.0
        for t in np.linspace(0, float(score_arr.max()) + 0.01, 50):
            mask = score_arr >= t
            n = int(mask.sum())
            if n == 0:
                continue
            cov = n / len(score_arr)
            if cov < min_cov:
                continue
            acc = int(correct_arr[mask].sum()) / n
            if acc > best_acc:
                best_acc = acc
                best_t = float(t)
                best_cov = cov
        return {"accuracy": round(best_acc, 4), "threshold": round(best_t, 4),
                "coverage": round(best_cov, 4)}

    best_with = best_accuracy_at_coverage(scores, correctness)
    best_without = best_accuracy_at_coverage(scores_no_disagree, correctness)

    ablation_table = Table(title="Ablation: Effect of Retriever Disagreement on Abstention")
    ablation_table.add_column("Variant", style="bold")
    ablation_table.add_column("Best Acc (≥50% cov)", justify="right")
    ablation_table.add_column("Coverage", justify="right")
    ablation_table.add_column("Threshold", justify="right")

    ablation_table.add_row("No abstention", f"{acc_all:.4f}", "100%", "—")
    ablation_table.add_row(
        "With disagreement signal",
        f"{best_with['accuracy']:.4f}",
        f"{best_with['coverage']:.1%}",
        f"{best_with['threshold']:.3f}",
    )
    ablation_table.add_row(
        "Without disagreement signal",
        f"{best_without['accuracy']:.4f}",
        f"{best_without['coverage']:.1%}",
        f"{best_without['threshold']:.3f}",
    )
    console.print(ablation_table)

    delta = best_with["accuracy"] - best_without["accuracy"]
    console.print(f"\n  Disagreement signal contribution: {'+' if delta > 0 else ''}{delta:.4f} accuracy")

    # ── Correlation: disagreement vs. correctness ────────────────────
    agreements = np.array([c["retriever_agreement"] for c in claims_with_evidence])
    corr = np.corrcoef(agreements, correctness.astype(float))[0, 1]

    # Split into high/low agreement
    median_agr = np.median(agreements)
    high_agr = correctness[agreements >= median_agr]
    low_agr = correctness[agreements < median_agr]

    console.print(f"\n[bold]Disagreement-Correctness Correlation:[/bold]")
    console.print(f"  Pearson(agreement, correct): {corr:.4f}")
    console.print(f"  Median agreement: {median_agr:.3f}")
    console.print(f"  Accuracy (high agreement, n={len(high_agr)}): {high_agr.mean():.3f}")
    console.print(f"  Accuracy (low agreement, n={len(low_agr)}): {low_agr.mean():.3f}")
    console.print(f"  Gap: {high_agr.mean() - low_agr.mean():+.3f}")

    # ── Leaderboard F1 with abstention ──────────────────────────────
    console.print(f"\n[bold]Leaderboard F1 (abstract-level) with abstention:[/bold]\n")

    # At best threshold, only submit predictions for answered claims
    if best_row:
        t = best_row["threshold"]
    else:
        t = best_with["threshold"]

    preds_all = [c["prediction"] for c in per_claim]
    preds_answered = [
        c["prediction"] for c in per_claim
        if c["combined_score"] >= t or not c["has_evidence"]
    ]
    # For abstained claims, submit empty evidence
    preds_with_abstention = []
    for c in per_claim:
        if c["combined_score"] >= t:
            preds_with_abstention.append(c["prediction"])
        else:
            preds_with_abstention.append({"id": int(c["claim_id"]), "evidence": {}})

    metrics_all = evaluate_against_gold(preds_all, gold)
    metrics_abstain = evaluate_against_gold(preds_with_abstention, gold)

    f1_table = Table(title="Leaderboard F1 Impact")
    f1_table.add_column("Variant", style="bold")
    f1_table.add_column("Precision", justify="right")
    f1_table.add_column("Recall", justify="right")
    f1_table.add_column("F1", justify="right")

    f1_table.add_row(
        "No abstention",
        f"{metrics_all['abstract_precision']:.3f}",
        f"{metrics_all['abstract_recall']:.3f}",
        f"{metrics_all['abstract_f1']:.3f}",
    )
    f1_table.add_row(
        f"With abstention (t={t:.3f})",
        f"{metrics_abstain['abstract_precision']:.3f}",
        f"{metrics_abstain['abstract_recall']:.3f}",
        f"{metrics_abstain['abstract_f1']:.3f}",
    )
    console.print(f1_table)

    # ── Save results ────────────────────────────────────────────────
    output = {
        "baseline_accuracy": acc_all,
        "n_claims_with_evidence": total,
        "n_correct_all": correct_all,
        "best_with_disagreement": best_with,
        "best_without_disagreement": best_without,
        "disagreement_contribution": round(delta, 4),
        "correlation_agreement_correct": round(float(corr), 4),
        "accuracy_high_agreement": round(float(high_agr.mean()), 4),
        "accuracy_low_agreement": round(float(low_agr.mean()), 4),
        "agreement_gap": round(float(high_agr.mean() - low_agr.mean()), 4),
        "leaderboard_f1_no_abstention": metrics_all,
        "leaderboard_f1_with_abstention": metrics_abstain,
        "abstention_threshold": t,
        "coverage_accuracy_curve": curve_data,
    }
    out_path = results_dir / "11_abstention_joint.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
