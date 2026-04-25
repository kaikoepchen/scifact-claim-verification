#!/usr/bin/env python3
"""FEVER evaluation: feature ablation + hand-tuned vs learned abstention gate.

Mirrors scripts/14_learned_gate.py but on FEVER. Also reports the
high/low-agreement accuracy split that scripts/11_abstention_joint.py shows
on SciFact, so the disagreement-signal claim is testable on a second dataset.

Run:
    python scripts/15_fever_evaluation.py
    python scripts/15_fever_evaluation.py --n-seeds 5 --max-dev-claims 1000
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from claimverify.data.fever import Fever
from claimverify.retrieval.bm25 import BM25Retriever
from claimverify.retrieval.dense import DenseRetriever
from claimverify.retrieval.fusion import ReciprocalRankFusion
from claimverify.retrieval.disagreement import jaccard_at_k
from claimverify.reasoning.joint import JointSentenceModel

console = Console()


ABLATION_CONFIGS = {
    "full":              {"cols": [0, 1, 2], "desc": "confidence + margin + agreement"},
    "no_disagreement":   {"cols": [0, 1],    "desc": "confidence + margin"},
    "disagreement_only": {"cols": [2],       "desc": "agreement only"},
    "confidence_only":   {"cols": [0],       "desc": "confidence only"},
}


# ── Metrics ───────────────────────────────────────────────────────────

def compute_auarc(y_true: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    n = len(y_sorted)
    if n == 0:
        return 0.0
    cumsum = np.cumsum(y_sorted)
    accuracies = cumsum / np.arange(1, n + 1)
    return float(np.mean(accuracies))


def accuracy_at_coverage(y_true: np.ndarray, scores: np.ndarray,
                         target_coverage: float) -> dict:
    n = len(y_true)
    k = max(1, int(np.ceil(target_coverage * n)))
    top_k_idx = np.argsort(-scores)[:k]
    return {
        "accuracy": round(float(y_true[top_k_idx].mean()), 4),
        "coverage": round(k / n, 4),
    }


# ── Feature collection ───────────────────────────────────────────────

def collect_features(fv, bm25, dense, rrf, joint, top_k=3):
    features = []  # [nli_confidence, nli_margin, retriever_agreement]
    labels = []    # 1 = correct, 0 = incorrect

    for claim in tqdm(fv.dev_claims, desc="Collecting features"):
        if not claim.evidence:
            continue

        bm25_results = bm25.retrieve(claim.text, top_k=100)
        dense_results = dense.retrieve(claim.text, top_k=100)
        retrieved = rrf.fuse(bm25_results, dense_results, top_k=top_k)
        agreement = jaccard_at_k(bm25_results, dense_results, k=10)

        retrieved_ids = [doc_id for doc_id, _ in retrieved]

        doc_sentences = {}
        for doc_id in retrieved_ids:
            if doc_id in fv.abstracts:
                doc_sentences[doc_id] = fv.abstracts[doc_id].sentences

        if not doc_sentences:
            features.append([0.0, 0.0, agreement])
            labels.append(0)
            continue

        doc_results = joint.predict_documents(claim.text, doc_sentences)

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

        if all_rationale_verdicts:
            probs = [
                np.mean([sv.logits.get("SUPPORT", 0) for sv in all_rationale_verdicts]),
                np.mean([sv.logits.get("CONTRADICT", 0) for sv in all_rationale_verdicts]),
                np.mean([sv.logits.get("NEI", 0) for sv in all_rationale_verdicts]),
            ]
        else:
            probs = [0.0, 0.0, 1.0]

        sorted_probs = sorted(probs, reverse=True)
        nli_confidence = sorted_probs[0]
        nli_margin = sorted_probs[0] - sorted_probs[1]

        # Correctness: any predicted (doc, label, rationale ⊇ gold) match.
        is_correct = False
        for doc_id, anns in claim.evidence.items():
            if doc_id in doc_verdicts:
                for ann in anns:
                    if doc_verdicts[doc_id] == ann["label"]:
                        pred_sents = set(doc_rationale_idxs.get(doc_id, []))
                        gold_sents = set(ann["sentences"])
                        if gold_sents.issubset(pred_sents):
                            is_correct = True

        features.append([float(nli_confidence), float(nli_margin), float(agreement)])
        labels.append(1 if is_correct else 0)

    return np.array(features), np.array(labels)


# ── Cross-validated learned gate ──────────────────────────────────────

def run_ablation(X: np.ndarray, y: np.ndarray, n_seeds: int = 3) -> dict:
    results = {}
    for config_name, config in ABLATION_CONFIGS.items():
        cols = config["cols"]
        X_sub = X[:, cols]

        seed_auarcs, seed_acc80, seed_acc90 = [], [], []
        all_coefs = []

        for seed in range(n_seeds):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + seed)
            oof_scores = np.zeros(len(y))

            for train_idx, val_idx in cv.split(X_sub, y):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_sub[train_idx])
                X_val = scaler.transform(X_sub[val_idx])

                model = LogisticRegressionCV(
                    Cs=10, cv=3, scoring="accuracy",
                    random_state=42 + seed, max_iter=1000,
                )
                model.fit(X_train, y[train_idx])
                oof_scores[val_idx] = model.predict_proba(X_val)[:, 1]
                all_coefs.append(model.coef_[0].tolist())

            seed_auarcs.append(compute_auarc(y, oof_scores))
            seed_acc80.append(accuracy_at_coverage(y, oof_scores, 0.80)["accuracy"])
            seed_acc90.append(accuracy_at_coverage(y, oof_scores, 0.90)["accuracy"])

        mean_coefs = np.mean(all_coefs, axis=0).tolist()
        feature_names = ["nli_confidence", "nli_margin", "retriever_agreement"]
        coef_dict = {feature_names[c]: round(v, 4) for c, v in zip(cols, mean_coefs)}

        results[config_name] = {
            "description": config["desc"],
            "features": [feature_names[c] for c in cols],
            "auarc_mean": round(float(np.mean(seed_auarcs)), 4),
            "auarc_std": round(float(np.std(seed_auarcs)), 4),
            "acc_at_80_mean": round(float(np.mean(seed_acc80)), 4),
            "acc_at_80_std": round(float(np.std(seed_acc80)), 4),
            "acc_at_90_mean": round(float(np.mean(seed_acc90)), 4),
            "acc_at_90_std": round(float(np.std(seed_acc90)), 4),
            "coefficients": coef_dict,
            "n_seeds": n_seeds,
        }

    return results


def hand_tuned_scores(X: np.ndarray) -> np.ndarray:
    return 0.45 * X[:, 0] + 0.25 * X[:, 1] + 0.30 * X[:, 2]


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FEVER abstention gate evaluation")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--max-dev-claims", type=int, default=1000)
    parser.add_argument("--max-corpus-docs", type=int, default=50_000)
    parser.add_argument("--joint-model", default="models/joint-scifact",
                        help="Path or HF id of the joint sentence-level model.")
    parser.add_argument("--features-cache", type=str, default=None,
                        help="Path to cached features.npz (skip pipeline if exists)")
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    console.print("\n[bold blue]FEVER — Abstention Gate Evaluation[/bold blue]\n")

    # ── Collect features (or load cache) ─────────────────────────────
    if args.features_cache and Path(args.features_cache).exists():
        console.print(f"Loading cached features from {args.features_cache}")
        data = np.load(args.features_cache)
        X, y = data["X"], data["y"]
    else:
        console.print(f"Loading FEVER (max_dev_claims={args.max_dev_claims}, "
                      f"max_corpus_docs={args.max_corpus_docs})...")
        fv = Fever.load(
            max_dev_claims=args.max_dev_claims,
            max_corpus_docs=args.max_corpus_docs,
        )
        console.print(f"  Dev claims: {len(fv.dev_claims)}  Corpus: {fv.corpus_size}")
        console.print(f"  Label dist: {fv.label_distribution('dev')}")

        corpus = fv.get_corpus_texts()

        console.print("Building BM25 index...")
        bm25 = BM25Retriever(k1=1.2, b=0.75)
        bm25.build_index(corpus)

        console.print("Building dense index...")
        dense = DenseRetriever(model_name="sentence-transformers/all-MiniLM-L6-v2")
        dense.build_index(corpus)

        rrf = ReciprocalRankFusion(k=60)

        console.print(f"Loading joint model from {args.joint_model}...")
        joint = JointSentenceModel(model_name=args.joint_model)

        X, y = collect_features(fv, bm25, dense, rrf, joint)

        cache_path = results_dir / "fever_gate_features.npz"
        np.savez(cache_path, X=X, y=y)
        console.print(f"Features cached to {cache_path}")

    if len(y) == 0:
        console.print("[red]No features collected — aborting.[/red]")
        return

    console.print(f"Dataset: {len(y)} claims, {y.sum()} correct ({y.mean():.1%})\n")

    # ── Hand-tuned baseline ──────────────────────────────────────────
    ht_scores = hand_tuned_scores(X)
    ht_auarc = compute_auarc(y, ht_scores)
    ht_acc80 = accuracy_at_coverage(y, ht_scores, 0.80)
    ht_acc90 = accuracy_at_coverage(y, ht_scores, 0.90)

    # ── Learned ablation ─────────────────────────────────────────────
    console.print(f"Running ablation with {args.n_seeds} seeds...\n")
    ablation = run_ablation(X, y, n_seeds=args.n_seeds)

    # ── High/low agreement split (mirrors script 11) ─────────────────
    agreements = X[:, 2]
    median_agr = float(np.median(agreements))
    high_mask = agreements >= median_agr
    low_mask = ~high_mask
    high_acc = float(y[high_mask].mean()) if high_mask.any() else 0.0
    low_acc = float(y[low_mask].mean()) if low_mask.any() else 0.0
    corr = float(np.corrcoef(agreements, y.astype(float))[0, 1]) if len(y) > 1 else 0.0

    # ── Display ──────────────────────────────────────────────────────
    table = Table(title="FEVER — Abstention Gate Feature Ablation")
    table.add_column("Config", style="bold")
    table.add_column("Features")
    table.add_column("AUARC", justify="right")
    table.add_column("Acc@80%", justify="right")
    table.add_column("Acc@90%", justify="right")

    table.add_row(
        "Hand-tuned", "conf + margin + agree (fixed w)",
        f"{ht_auarc:.4f}",
        f"{ht_acc80['accuracy']:.4f}",
        f"{ht_acc90['accuracy']:.4f}",
    )
    table.add_row("", "", "", "", "")

    for name, res in ablation.items():
        table.add_row(
            name, res["description"],
            f"{res['auarc_mean']:.4f} ± {res['auarc_std']:.4f}",
            f"{res['acc_at_80_mean']:.4f} ± {res['acc_at_80_std']:.4f}",
            f"{res['acc_at_90_mean']:.4f} ± {res['acc_at_90_std']:.4f}",
        )

    console.print(table)

    full = ablation["full"]
    no_dis = ablation["no_disagreement"]
    delta_auarc = full["auarc_mean"] - no_dis["auarc_mean"]
    delta_acc80 = full["acc_at_80_mean"] - no_dis["acc_at_80_mean"]

    console.print(f"\n[bold]Disagreement signal contribution (learned):[/bold]")
    console.print(f"  AUARC:  {delta_auarc:+.4f} (full - no_disagreement)")
    console.print(f"  Acc@80: {delta_acc80:+.4f} (full - no_disagreement)")

    console.print(f"\n[bold]High/low-agreement split:[/bold]")
    console.print(f"  Median agreement: {median_agr:.3f}")
    console.print(f"  Acc (high agreement, n={int(high_mask.sum())}): {high_acc:.3f}")
    console.print(f"  Acc (low agreement,  n={int(low_mask.sum())}): {low_acc:.3f}")
    console.print(f"  Gap: {high_acc - low_acc:+.3f}")
    console.print(f"  Pearson(agreement, correct): {corr:+.4f}")

    console.print(f"\n[bold]Learned coefficients (full):[/bold]")
    for feat, coef in full["coefficients"].items():
        console.print(f"  {feat}: {coef:+.4f}")

    # ── Save ─────────────────────────────────────────────────────────
    output = {
        "dataset": "fever",
        "n_claims": int(len(y)),
        "n_correct": int(y.sum()),
        "baseline_accuracy": round(float(y.mean()), 4),
        "hand_tuned": {
            "auarc": round(ht_auarc, 4),
            "acc_at_80": ht_acc80,
            "acc_at_90": ht_acc90,
        },
        "ablation": ablation,
        "disagreement_contribution": {
            "auarc_delta": round(delta_auarc, 4),
            "acc_at_80_delta": round(delta_acc80, 4),
        },
        "agreement_split": {
            "median_agreement": round(median_agr, 4),
            "accuracy_high_agreement": round(high_acc, 4),
            "accuracy_low_agreement": round(low_acc, 4),
            "agreement_gap": round(high_acc - low_acc, 4),
            "correlation_agreement_correct": round(corr, 4),
        },
    }
    out_path = results_dir / "15_fever_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
