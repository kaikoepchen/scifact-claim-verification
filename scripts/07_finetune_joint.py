#!/usr/bin/env python3
"""Fine-tune a joint sentence-level rationale selector + verdict predictor.

Trains a single model on (claim, sentence) pairs with three classes:
  SUPPORT (0):    sentence is evidence supporting the claim
  CONTRADICT (1): sentence is evidence contradicting the claim
  NEI (2):        sentence is not relevant evidence

Training data comes from SciFact gold annotations:
  - Positive: gold rationale sentences with their verdict label
  - In-doc negatives: non-rationale sentences from evidence documents
  - Hard negatives: sentences from top-retrieved non-evidence documents

Uses DeBERTa-v3-large as base (pre-trained on MNLI+FEVER+ANLI+WANLI),
which gives strong NLI priors that transfer well to scientific claims.

Run:
    python scripts/07_finetune_joint.py
    python scripts/07_finetune_joint.py --epochs 8 --lr 1e-5
    python scripts/07_finetune_joint.py --base-model microsoft/deberta-v3-base
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from claimverify.data.scifact import SciFact
from claimverify.evaluation.metrics import macro_f1

console = Console()

LABEL2ID = {"SUPPORT": 0, "CONTRADICT": 1, "NEI": 2}
ID2LABEL = {0: "SUPPORT", 1: "CONTRADICT", 2: "NEI"}


class SentenceLevelDataset(Dataset):
    """Sentence-level training data for joint rationale+verdict prediction."""

    def __init__(
        self,
        claims,
        abstracts,
        tokenizer,
        max_length=512,
        neg_ratio=3.0,
        seed=42,
    ):
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

                # Collect gold rationale sentence indices and their label
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

                # Positive examples: gold rationale sentences
                for sent_idx in gold_indices:
                    self.examples.append({
                        "claim": claim.text,
                        "sentence": sentences[sent_idx],
                        "label": LABEL2ID[label_for_doc],
                    })

                # In-doc negative examples: non-rationale sentences
                non_rationale = [
                    i for i in range(len(sentences)) if i not in gold_indices
                ]
                # Sample up to neg_ratio * |positives| negatives from this doc
                n_neg = min(
                    len(non_rationale),
                    int(neg_ratio * len(gold_indices)),
                )
                sampled_neg = rng.sample(non_rationale, n_neg)
                for sent_idx in sampled_neg:
                    self.examples.append({
                        "claim": claim.text,
                        "sentence": sentences[sent_idx],
                        "label": LABEL2ID["NEI"],
                    })

        rng.shuffle(self.examples)

        # Count labels for logging
        self.label_counts = {ID2LABEL[i]: 0 for i in range(3)}
        for ex in self.examples:
            self.label_counts[ID2LABEL[ex["label"]]] += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex["claim"],
            ex["sentence"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
        }


def evaluate(model, dataloader, device):
    """Evaluate on dev set, return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
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
    parser = argparse.ArgumentParser(description="Fine-tune joint sentence-level model")
    parser.add_argument(
        "--base-model",
        default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        help="Base model with NLI pretraining",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=256, help="Shorter since sentence-level")
    parser.add_argument("--neg-ratio", type=float, default=3.0, help="Negative:positive ratio per doc")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--output-dir", default="models/joint-scifact")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"\n[bold blue]Fine-tuning Joint Sentence Model[/bold blue]")
    console.print(f"  Device: {device}")
    console.print(f"  Base model: {args.base_model}")
    console.print(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}x{args.grad_accum}")
    console.print(f"  Neg ratio: {args.neg_ratio}, Max length: {args.max_length}")
    console.print(f"  Patience: {args.patience}\n")

    # Load data
    console.print("Loading SciFact...")
    sf = SciFact.load()

    console.print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=3, ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,
    )
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID
    model.to(device)

    # Build datasets
    console.print("Building sentence-level datasets...")
    train_dataset = SentenceLevelDataset(
        sf.train_claims, sf.abstracts, tokenizer,
        max_length=args.max_length, neg_ratio=args.neg_ratio, seed=args.seed,
    )
    dev_dataset = SentenceLevelDataset(
        sf.dev_claims, sf.abstracts, tokenizer,
        max_length=args.max_length, neg_ratio=args.neg_ratio, seed=args.seed + 1,
    )

    console.print(f"  Train: {len(train_dataset)} examples")
    for label, count in train_dataset.label_counts.items():
        console.print(f"    {label}: {count}")
    console.print(f"  Dev: {len(dev_dataset)} examples")
    for label, count in dev_dataset.label_counts.items():
        console.print(f"    {label}: {count}")

    # Weighted sampling to handle class imbalance
    label_counts = train_dataset.label_counts
    total = sum(label_counts.values())
    class_weights = {
        LABEL2ID[label]: total / (3 * count) if count > 0 else 0.0
        for label, count in label_counts.items()
    }
    sample_weights = [class_weights[ex["label"]] for ex in train_dataset.examples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    # Mixed precision for memory efficiency on T4
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Class-weighted loss
    weight_tensor = torch.tensor(
        [class_weights[i] for i in range(3)], dtype=torch.float
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)

    # Train
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]Training (mixed precision: {use_amp})...[/bold]\n")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels) / args.grad_accum

            scaler.scale(loss).backward()
            total_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{loss.item() * args.grad_accum:.4f}")

        # Flush remaining gradients
        if (step + 1) % args.grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        dev_metrics = evaluate(model, dev_loader, device)
        f1 = dev_metrics["macro_f1"]

        console.print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, "
                      f"dev_acc={dev_metrics['accuracy']:.3f}, dev_macro_f1={f1:.3f}")
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
                console.print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
                break

    console.print(f"\n[bold green]Best: epoch {best_epoch}, macro-F1={best_f1:.3f}[/bold green]")
    console.print(f"Model saved to {output_dir}")

    # Final eval with best model
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    final_metrics = evaluate(model, dev_loader, device)

    console.print(f"\n[bold]Final dev results (best checkpoint):[/bold]")
    console.print(f"  Accuracy: {final_metrics['accuracy']:.3f}")
    console.print(f"  Macro-F1: {final_metrics['macro_f1']:.3f}")
    for cls in ["SUPPORT", "CONTRADICT", "NEI"]:
        if cls in final_metrics["per_class"]:
            pc = final_metrics["per_class"][cls]
            console.print(f"  {cls}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f}")

    # Save training log
    log = {
        "base_model": args.base_model,
        "epochs_run": min(epoch + 1, args.epochs),
        "epochs_max": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_length": args.max_length,
        "neg_ratio": args.neg_ratio,
        "train_examples": len(train_dataset),
        "train_label_counts": train_dataset.label_counts,
        "dev_examples": len(dev_dataset),
        "dev_label_counts": dev_dataset.label_counts,
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "final_metrics": final_metrics,
    }
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    console.print(f"Training log saved to {output_dir / 'training_log.json'}")


if __name__ == "__main__":
    main()
