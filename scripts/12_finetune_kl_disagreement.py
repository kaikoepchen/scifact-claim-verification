#!/usr/bin/env python3
"""Fine-tune joint model with KL-divergence retriever consistency regularization.

Core idea: for each training claim, retrieve evidence with BM25 and dense
separately, run the model on both views, and regularize with symmetric KL
divergence between the two prediction distributions. This teaches the model
to produce consistent verdicts regardless of which retrieval method provided
the evidence — making it robust to retrieval variance.

Loss = task_loss + lambda_kl * symmetric_KL(pred_bm25, pred_dense)

The task loss trains on gold sentence labels (as before). The KL term is
an additional signal that doesn't require gold labels — it only asks that
the model be self-consistent across retrieval views.

Run:
    python scripts/12_finetune_kl_disagreement.py
    python scripts/12_finetune_kl_disagreement.py --lambda-kl 0.5
    python scripts/12_finetune_kl_disagreement.py --lambda-kl 0.0  # ablation: no KL
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from claimverify.data.scifact import SciFact
from claimverify.retrieval.bm25 import BM25Retriever
from claimverify.retrieval.dense import DenseRetriever
from claimverify.retrieval.disagreement import jaccard_at_k
from claimverify.evaluation.metrics import macro_f1

console = Console()

LABEL2ID = {"SUPPORT": 0, "CONTRADICT": 1, "NEI": 2}
ID2LABEL = {0: "SUPPORT", 1: "CONTRADICT", 2: "NEI"}


def symmetric_kl(p_logits, q_logits):
    """Symmetric KL divergence between two logit distributions.

    SKL(P || Q) = 0.5 * (KL(P || Q) + KL(Q || P))
    """
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)

    kl_pq = F.kl_div(q, p_probs, reduction="batchmean", log_target=False)
    kl_qp = F.kl_div(p, q_probs, reduction="batchmean", log_target=False)
    return 0.5 * (kl_pq + kl_qp)


def precompute_retrieval(claims, corpus, abstracts, bm25, dense, top_k=5):
    """Pre-compute BM25 and dense retrieval for all claims.

    Returns dict: claim_id -> {
        "bm25_docs": {doc_id: [sentences]},
        "dense_docs": {doc_id: [sentences]},
        "jaccard": float,
    }
    """
    results = {}
    for claim in tqdm(claims, desc="Pre-computing retrieval"):
        bm25_results = bm25.retrieve(claim.text, top_k=top_k)
        dense_results = dense.retrieve(claim.text, top_k=top_k)
        jaccard = jaccard_at_k(
            bm25.retrieve(claim.text, top_k=100),
            dense.retrieve(claim.text, top_k=100),
            k=10,
        )

        bm25_docs = {}
        for doc_id, _ in bm25_results:
            if doc_id in abstracts:
                bm25_docs[doc_id] = abstracts[doc_id].sentences

        dense_docs = {}
        for doc_id, _ in dense_results:
            if doc_id in abstracts:
                dense_docs[doc_id] = abstracts[doc_id].sentences

        results[claim.claim_id] = {
            "bm25_docs": bm25_docs,
            "dense_docs": dense_docs,
            "jaccard": jaccard,
        }
    return results


class SentenceLevelDataset(Dataset):
    """Sentence-level training data with claim grouping for KL computation."""

    def __init__(self, claims, abstracts, tokenizer, max_length=256, neg_ratio=3.0, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        rng = random.Random(seed)

        for claim in claims:
            for doc_id, annotations in claim.evidence.items():
                if doc_id not in abstracts:
                    continue
                sentences = abstracts[doc_id].sentences
                if not sentences:
                    continue

                gold_indices = set()
                label_for_doc = None
                for ann in annotations:
                    label = ann["label"]
                    if label not in ("SUPPORT", "CONTRADICT"):
                        continue
                    label_for_doc = label
                    for sent_idx in ann["sentences"]:
                        if sent_idx < len(sentences):
                            gold_indices.add(sent_idx)

                if not gold_indices or label_for_doc is None:
                    continue

                for sent_idx in gold_indices:
                    self.examples.append({
                        "claim": claim.text,
                        "claim_id": claim.claim_id,
                        "sentence": sentences[sent_idx],
                        "label": LABEL2ID[label_for_doc],
                    })

                non_rationale = [i for i in range(len(sentences)) if i not in gold_indices]
                n_neg = min(len(non_rationale), int(neg_ratio * len(gold_indices)))
                for sent_idx in rng.sample(non_rationale, n_neg):
                    self.examples.append({
                        "claim": claim.text,
                        "claim_id": claim.claim_id,
                        "sentence": sentences[sent_idx],
                        "label": LABEL2ID["NEI"],
                    })

        rng.shuffle(self.examples)
        self.label_counts = {ID2LABEL[i]: 0 for i in range(3)}
        for ex in self.examples:
            self.label_counts[ID2LABEL[ex["label"]]] += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex["claim"], ex["sentence"],
            truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
            "claim_id": ex["claim_id"],
        }


def compute_view_logits(model, tokenizer, claim_text, doc_sentences, device,
                        max_length=256, no_grad=False, max_sents=10):
    """Run model on sentences from retrieved documents, return mean logits.

    This gives a claim-level prediction distribution for one retrieval view.
    Limits to max_sents sentences to control memory usage.
    """
    all_sents = []
    for doc_id, sentences in doc_sentences.items():
        all_sents.extend(sentences)

    if not all_sents:
        return None

    # Limit sentences to control memory
    if len(all_sents) > max_sents:
        all_sents = all_sents[:max_sents]

    inputs = tokenizer(
        [claim_text] * len(all_sents),
        all_sents,
        truncation=True, max_length=max_length,
        padding=True, return_tensors="pt",
    ).to(device)

    if no_grad:
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                outputs = model(**inputs)
        return outputs.logits.mean(dim=0, keepdim=True).detach()
    else:
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            outputs = model(**inputs)
        return outputs.logits.mean(dim=0, keepdim=True)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    use_amp = device == "cuda"
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend([ID2LABEL[p.item()] for p in preds])
            all_labels.extend([ID2LABEL[l.item()] for l in labels])
    return macro_f1(all_preds, all_labels, classes=["SUPPORT", "CONTRADICT", "NEI"])


def main():
    parser = argparse.ArgumentParser(description="Fine-tune with KL disagreement regularization")
    parser.add_argument("--base-model", default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--neg-ratio", type=float, default=3.0)
    parser.add_argument("--lambda-kl", type=float, default=0.3, help="Weight for KL regularization")
    parser.add_argument("--kl-top-k", type=int, default=3, help="Top-k docs per retriever for KL views")
    parser.add_argument("--kl-every-n", type=int, default=4, help="Compute KL term every N gradient steps")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--output-dir", default="models/joint-kl-scifact")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-transformer / HF model for the dense retrieval view used in KL")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(f"\n[bold blue]Fine-tuning with KL Disagreement Regularization[/bold blue]")
    console.print(f"  Device: {device}")
    console.print(f"  Base: {args.base_model}")
    console.print(f"  lambda_kl: {args.lambda_kl}")
    console.print(f"  KL top-k: {args.kl_top_k}, every {args.kl_every_n} steps")
    console.print(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}x{args.grad_accum}\n")

    # ── Load data ────────────────────────────────────────────────────
    console.print("Loading SciFact...")
    sf = SciFact.load()
    corpus = sf.get_corpus_texts()

    # ── Build retrievers and pre-compute retrieval ───────────────────
    console.print("Building BM25 index...")
    bm25 = BM25Retriever(k1=1.2, b=0.75)
    bm25.build_index(corpus)

    console.print(f"Building dense index [{args.dense_model}]...")
    dense_retriever = DenseRetriever(model_name=args.dense_model)
    dense_retriever.build_index(corpus)

    console.print("Pre-computing retrieval for training claims...")
    train_retrieval = precompute_retrieval(
        sf.train_claims, corpus, sf.abstracts, bm25, dense_retriever, top_k=args.kl_top_k,
    )

    # Collect unique training claim texts for KL computation
    train_claim_map = {c.claim_id: c.text for c in sf.train_claims}
    # Only claims that have retrieval results from both retrievers
    kl_claim_ids = [
        cid for cid, r in train_retrieval.items()
        if r["bm25_docs"] and r["dense_docs"]
    ]
    console.print(f"  Claims with both retrieval views: {len(kl_claim_ids)}")

    # ── Load model ───────────────────────────────────────────────────
    console.print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=3, ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,
    )
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID
    model.to(device)

    # ── Build datasets ───────────────────────────────────────────────
    train_dataset = SentenceLevelDataset(
        sf.train_claims, sf.abstracts, tokenizer,
        max_length=args.max_length, neg_ratio=args.neg_ratio, seed=args.seed,
    )
    dev_dataset = SentenceLevelDataset(
        sf.dev_claims, sf.abstracts, tokenizer,
        max_length=args.max_length, neg_ratio=args.neg_ratio, seed=args.seed + 1,
    )

    console.print(f"  Train: {len(train_dataset)} examples")
    for lbl, cnt in train_dataset.label_counts.items():
        console.print(f"    {lbl}: {cnt}")
    console.print(f"  Dev: {len(dev_dataset)} examples\n")

    # Weighted sampling
    label_counts = train_dataset.label_counts
    total = sum(label_counts.values())
    class_weights = {
        LABEL2ID[lbl]: total / (3 * cnt) if cnt > 0 else 0.0
        for lbl, cnt in label_counts.items()
    }
    sample_weights = [class_weights[ex["label"]] for ex in train_dataset.examples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

    # ── Optimizer, scheduler, AMP ────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps,
    )
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    weight_tensor = torch.tensor(
        [class_weights[i] for i in range(3)], dtype=torch.float
    ).to(device)
    task_loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    # ── Training loop ────────────────────────────────────────────────
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kl_rng = random.Random(args.seed + 100)
    epoch_kl_losses = []

    console.print(f"[bold]Training (lambda_kl={args.lambda_kl})...[/bold]\n")

    for epoch in range(args.epochs):
        model.train()
        total_task_loss = 0.0
        total_kl_loss = 0.0
        n_kl_steps = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # ── Task loss (standard cross-entropy on gold labels) ────
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                task_loss = task_loss_fn(outputs.logits, labels) / args.grad_accum

            # ── KL regularization (every N grad steps to save compute) ──
            kl_loss_value = 0.0
            compute_kl = (
                args.lambda_kl > 0
                and (step + 1) % args.kl_every_n == 0
                and len(kl_claim_ids) > 0
            )

            if compute_kl:
                # Sample a random claim for KL computation
                cid = kl_rng.choice(kl_claim_ids)
                claim_text = train_claim_map[cid]
                retrieval = train_retrieval[cid]

                # BM25 view: with gradients (model learns from this)
                # Dense view: no_grad + detached (target distribution)
                # This halves memory vs. computing gradients through both views
                with torch.amp.autocast("cuda", enabled=use_amp):
                    bm25_logits = compute_view_logits(
                        model, tokenizer, claim_text,
                        retrieval["bm25_docs"], device, args.max_length,
                        no_grad=False, max_sents=8,
                    )
                    dense_logits = compute_view_logits(
                        model, tokenizer, claim_text,
                        retrieval["dense_docs"], device, args.max_length,
                        no_grad=True, max_sents=8,
                    )

                    if bm25_logits is not None and dense_logits is not None:
                        kl_term = symmetric_kl(bm25_logits, dense_logits)
                        kl_loss = args.lambda_kl * kl_term / args.grad_accum
                        kl_loss_value = kl_term.item()
                    else:
                        kl_loss = torch.tensor(0.0, device=device)
            else:
                kl_loss = torch.tensor(0.0, device=device)

            # ── Combined backward ────────────────────────────────────
            combined = task_loss + kl_loss
            scaler.scale(combined).backward()

            total_task_loss += task_loss.item() * args.grad_accum
            if compute_kl and kl_loss_value > 0:
                total_kl_loss += kl_loss_value
                n_kl_steps += 1

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            avg_kl = total_kl_loss / max(n_kl_steps, 1)
            pbar.set_postfix(task=f"{task_loss.item() * args.grad_accum:.3f}", kl=f"{avg_kl:.3f}")

        # Flush remaining gradients
        if (step + 1) % args.grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        avg_task = total_task_loss / len(train_loader)
        avg_kl = total_kl_loss / max(n_kl_steps, 1)
        epoch_kl_losses.append(avg_kl)

        # ── Evaluate ─────────────────────────────────────────────────
        dev_metrics = evaluate(model, dev_loader, device)
        f1 = dev_metrics["macro_f1"]

        console.print(f"  Epoch {epoch+1}: task_loss={avg_task:.4f}, kl_loss={avg_kl:.4f}, "
                      f"dev_acc={dev_metrics['accuracy']:.3f}, dev_f1={f1:.3f}")
        for cls in ["SUPPORT", "CONTRADICT", "NEI"]:
            if cls in dev_metrics["per_class"]:
                pc = dev_metrics["per_class"][cls]
                console.print(f"    {cls}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            patience_counter = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            console.print(f"  -> Saved best model (F1={f1:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                console.print(f"\n  Early stopping at epoch {epoch+1}")
                break

    console.print(f"\n[bold green]Best: epoch {best_epoch}, macro-F1={best_f1:.3f}[/bold green]")

    # ── Final evaluation ─────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    final_metrics = evaluate(model, dev_loader, device)

    console.print(f"\n[bold]Final dev results:[/bold]")
    console.print(f"  Accuracy: {final_metrics['accuracy']:.3f}")
    console.print(f"  Macro-F1: {final_metrics['macro_f1']:.3f}")
    for cls in ["SUPPORT", "CONTRADICT", "NEI"]:
        if cls in final_metrics["per_class"]:
            pc = final_metrics["per_class"][cls]
            console.print(f"  {cls}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f}")

    # ── Save log ─────────────────────────────────────────────────────
    log = {
        "base_model": args.base_model,
        "dense_model": args.dense_model,
        "lambda_kl": args.lambda_kl,
        "kl_top_k": args.kl_top_k,
        "kl_every_n": args.kl_every_n,
        "epochs_run": min(epoch + 1, args.epochs),
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "neg_ratio": args.neg_ratio,
        "train_examples": len(train_dataset),
        "dev_examples": len(dev_dataset),
        "kl_eligible_claims": len(kl_claim_ids),
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "epoch_avg_kl_losses": epoch_kl_losses,
        "final_metrics": final_metrics,
    }
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    console.print(f"Training log saved to {output_dir / 'training_log.json'}")


if __name__ == "__main__":
    main()
