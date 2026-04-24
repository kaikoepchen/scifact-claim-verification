#!/usr/bin/env python3
"""Learned abstention gate with feature ablation.

Replaces hand-tuned weights with a logistic regression trained to predict
whether the system will be correct, given the three uncertainty signals:
  1. NLI confidence (max class probability)
  2. NLI margin (gap between top two classes)
  3. Retriever agreement (Jaccard@10 between BM25 and dense)

The key experiment is the feature ablation: train the model in four
configurations and compare AUARC to isolate the contribution of each signal.

  | Config             | Features                          |
  |--------------------|-----------------------------------|
  | Full               | confidence + margin + agreement   |
  | No disagreement    | confidence + margin               |
  | Disagreement only  | agreement                         |
  | Confidence only    | confidence                        |

Run:
    python scripts/14_learned_gate.py
    python scripts/14_learned_gate.py --n-seeds 5
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

from claimverify.data.scifact import SciFact
from claimverify.retrieval.bm25 import BM25Retriever
from claimverify.retrieval.dense import DenseRetriever
from claimverify.retrieval.fusion import ReciprocalRankFusion
from claimverify.retrieval.disagreement import jaccard_at_k
from claimverify.reasoning.joint import JointSentenceModel

console = Console()


# ── Feature ablation configs ──────────────────────────────────────────

ABLATION_CONFIGS = {
    "full":              {"cols": [0, 1, 2], "desc": "confidence + margin + agreement"},
    "no_disagreement":   {"cols": [0, 1],    "desc": "confidence + margin"},
    "disagreement_only": {"cols": [2],       "desc": "agreement only"},
    "confidence_only":   {"cols": [0],       "desc": "confidence only"},
}


# ── Metrics ───────────────────────────────────────────────────────────

def compute_auarc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area Under the Accuracy-Rejection Curve.

    Sort by descending confidence. At each rejection threshold, compute
    accuracy on the remaining (non-rejected) examples. AUARC is the
    area under this curve, normalized to [0, 1].
    """
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    n = len(y_sorted)
    if n == 0:
        return 0.0

    cumsum = np.cumsum(y_sorted)
    accuracies = cumsum / np.arange(1, n + 1)

    # Trapezoidal integration over coverage [1/n, 1]
    auarc = np.mean(accuracies)
    return float(auarc)


def accuracy_at_coverage(y_true: np.ndarray, scores: np.ndarray,
                         target_coverage: float) -> dict:
    """Accuracy when answering only the top-k% most confident claims."""
    n = len(y_true)
    k = max(1, int(np.ceil(target_coverage * n)))
    top_k_idx = np.argsort(-scores)[:k]
    acc = float(y_true[top_k_idx].mean())
    actual_cov = k / n
    return {"accuracy": round(acc, 4), "coverage": round(actual_cov, 4)}


# ── Pipeline: collect features ────────────────────────────────────────

def collect_features(sf, bm25, dense, rrf, joint, top_k=3):
    """Run the pipeline on dev claims and collect per-claim features + labels."""
    features = []  # Each row: [nli_confidence, nli_margin, retriever_agreement]
    labels = []    # 1 = correct, 0 = incorrect

    for claim in tqdm(sf.dev_claims, desc="Collecting features"):
        if not claim.evidence:
            continue

        # Retrieve
        bm25_results = bm25.retrieve(claim.text, top_k=100)
        dense_results = dense.retrieve(claim.text, top_k=100)
        retrieved = rrf.fuse(bm25_results, dense_results, top_k=top_k)
        agreement = jaccard_at_k(bm25_results, dense_results, k=10)

        retrieved_ids = [doc_id for doc_id, _ in retrieved]

        # Joint prediction
        doc_sentences = {}
        for doc_id in retrieved_ids:
            if doc_id in sf.abstracts:
                doc_sentences[doc_id] = sf.abstracts[doc_id].sentences

        doc_results = joint.predict_documents(claim.text, doc_sentences)

        # Extract verdicts and rationale logits
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

        # NLI confidence and margin
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

        # Correctness label
        is_correct = False
        for doc_id, anns in claim.evidence.items():
            if doc_id in doc_verdicts:
                for ann in anns:
                    if doc_verdicts[doc_id] == ann["label"]:
                        pred_sents = set(doc_rationale_idxs.get(doc_id, []))
                        gold_sents = set(ann["sentences"])
                        if gold_sents.issubset(pred_sents):
                            is_correct = True

        features.append([nli_confidence, nli_margin, agreement])
        labels.append(1 if is_correct else 0)

    return np.array(features), np.array(labels)


# ── Cross-validated learned gate ──────────────────────────────────────

def run_ablation(X: np.ndarray, y: np.ndarray, n_seeds: int = 3) -> dict:
    """Run cross-validated logistic regression for each ablation config.

    Returns {config_name: {auarc, acc@80, acc@90, std, coefficients}}.
    """
    results = {}

    for config_name, config in ABLATION_CONFIGS.items():
        cols = config["cols"]
        X_sub = X[:, cols]

        seed_auarcs = []
        seed_acc80 = []
        seed_acc90 = []
        all_coefs = []

        for seed in range(n_seeds):
            # Cross-validated predictions (each example scored out-of-fold)
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

            auarc = compute_auarc(y, oof_scores)
            a80 = accuracy_at_coverage(y, oof_scores, 0.80)
            a90 = accuracy_at_coverage(y, oof_scores, 0.90)

            seed_auarcs.append(auarc)
            seed_acc80.append(a80["accuracy"])
            seed_acc90.append(a90["accuracy"])

        # Average coefficients across all folds and seeds
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


# ── Hand-tuned baseline ──────────────────────────────────────────────

def hand_tuned_scores(X: np.ndarray) -> np.ndarray:
    """Replicate the hand-tuned gate: 0.45*conf + 0.25*margin + 0.30*agree."""
    return 0.45 * X[:, 0] + 0.25 * X[:, 1] + 0.30 * X[:, 2]


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Learned abstention gate with feature ablation")
    parser.add_argument("--n-seeds", type=int, default=3, help="Seeds for cross-validation")
    parser.add_argument("--features-cache", type=str, default=None,
                        help="Path to cached features.npz (skip pipeline if exists)")
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    console.print("\n[bold blue]Learned Abstention Gate — Feature Ablation[/bold blue]\n")

    # ── Collect features (or load cache) ─────────────────────────────
    if args.features_cache and Path(args.features_cache).exists():
        console.print(f"Loading cached features from {args.features_cache}")
        data = np.load(args.features_cache)
        X, y = data["X"], data["y"]
    else:
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

        console.print("Loading joint model...")
        joint = JointSentenceModel(model_name="models/joint-scifact")

        X, y = collect_features(sf, bm25, dense, rrf, joint)

        # Cache for future runs
        cache_path = results_dir / "gate_features.npz"
        np.savez(cache_path, X=X, y=y)
        console.print(f"Features cached to {cache_path}")

    console.print(f"Dataset: {len(y)} claims, {y.sum()} correct ({y.mean():.1%})\n")

    # ── Hand-tuned baseline ──────────────────────────────────────────
    ht_scores = hand_tuned_scores(X)
    ht_auarc = compute_auarc(y, ht_scores)
    ht_acc80 = accuracy_at_coverage(y, ht_scores, 0.80)
    ht_acc90 = accuracy_at_coverage(y, ht_scores, 0.90)

    # ── Learned ablation ─────────────────────────────────────────────
    console.print(f"Running ablation with {args.n_seeds} seeds...\n")
    ablation = run_ablation(X, y, n_seeds=args.n_seeds)

    # ── Display results ──────────────────────────────────────────────
    table = Table(title="Abstention Gate — Feature Ablation")
    table.add_column("Config", style="bold")
    table.add_column("Features")
    table.add_column("AUARC", justify="right")
    table.add_column("Acc@80%", justify="right")
    table.add_column("Acc@90%", justify="right")

    # Hand-tuned row
    table.add_row(
        "Hand-tuned",
        "conf + margin + agree (fixed w)",
        f"{ht_auarc:.4f}",
        f"{ht_acc80['accuracy']:.4f}",
        f"{ht_acc90['accuracy']:.4f}",
    )
    table.add_row("", "", "", "", "")

    for name, res in ablation.items():
        auarc_str = f"{res['auarc_mean']:.4f} ± {res['auarc_std']:.4f}"
        acc80_str = f"{res['acc_at_80_mean']:.4f} ± {res['acc_at_80_std']:.4f}"
        acc90_str = f"{res['acc_at_90_mean']:.4f} ± {res['acc_at_90_std']:.4f}"
        table.add_row(name, res["description"], auarc_str, acc80_str, acc90_str)

    console.print(table)

    # ── Key comparison ───────────────────────────────────────────────
    full = ablation["full"]
    no_dis = ablation["no_disagreement"]
    delta_auarc = full["auarc_mean"] - no_dis["auarc_mean"]
    delta_acc80 = full["acc_at_80_mean"] - no_dis["acc_at_80_mean"]

    console.print(f"\n[bold]Disagreement signal contribution (learned):[/bold]")
    console.print(f"  AUARC:  {delta_auarc:+.4f} (full - no_disagreement)")
    console.print(f"  Acc@80: {delta_acc80:+.4f} (full - no_disagreement)")

    # Coefficients
    console.print(f"\n[bold]Learned coefficients (full model, averaged):[/bold]")
    for feat, coef in full["coefficients"].items():
        console.print(f"  {feat}: {coef:+.4f}")

    # ── Save ─────────────────────────────────────────────────────────
    output = {
        "n_claims": len(y),
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
    }
    out_path = results_dir / "14_learned_gate.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
