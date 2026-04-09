#!/usr/bin/env python3
"""Fine-tune DeBERTa on SciFact training data for verdict prediction.

Standard NLI fine-tuning on (claim, abstract) pairs. Nothing novel here —
just building a working predictor so we can evaluate the disagreement
signal properly.

Run:
    python scripts/07_finetune_verdict.py
    python scripts/07_finetune_verdict.py --epochs 5 --lr 1e-5
    python scripts/07_finetune_verdict.py --base-model microsoft/deberta-v3-base
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from rich.console import Console
from tqdm import tqdm

from claimverify.data.scifact import SciFact
from claimverify.evaluation.metrics import macro_f1

console = Console()

LABEL2ID = {"SUPPORT": 0, "CONTRADICT": 1}
ID2LABEL = {0: "SUPPORT", 1: "CONTRADICT"}


class SciFactNLIDataset(Dataset):
    def __init__(self, claims, abstracts, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for claim in claims:
            for doc_id, annotations in claim.evidence.items():
                if doc_id not in abstracts:
                    continue
                abstract_text = abstracts[doc_id].text
                for ann in annotations:
                    label = ann["label"]
                    if label not in LABEL2ID:
                        continue
                    self.examples.append({
                        "claim": claim.text,
                        "evidence": abstract_text,
                        "label": LABEL2ID[label],
                    })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = self.tokenizer(
            ex["claim"],
            ex["evidence"],
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
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)

            all_preds.extend([ID2LABEL[p.item()] for p in preds])
            all_labels.extend([ID2LABEL[l.item()] for l in labels])

    return macro_f1(all_preds, all_labels, classes=["SUPPORT", "CONTRADICT"])


def main():
    parser = argparse.ArgumentParser(description="Fine-tune verdict model on SciFact")
    parser.add_argument("--base-model", default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output-dir", default="models/verdict-scifact")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"\n[bold blue]Fine-tuning Verdict Model[/bold blue]")
    console.print(f"  Device: {device}")
    console.print(f"  Base model: {args.base_model}")
    console.print(f"  Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}\n")

    # Load data
    console.print("Loading SciFact...")
    sf = SciFact.load()

    console.print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=2, ignore_mismatched_sizes=True,
    )
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID
    model.to(device)

    # Build datasets
    train_dataset = SciFactNLIDataset(sf.train_claims, sf.abstracts, tokenizer, args.max_length)
    dev_dataset = SciFactNLIDataset(sf.dev_claims, sf.abstracts, tokenizer, args.max_length)

    console.print(f"  Train examples: {len(train_dataset)}")
    console.print(f"  Dev examples: {len(dev_dataset)}\n")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps,
    )

    # Train
    best_f1 = 0.0
    best_epoch = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate
        dev_metrics = evaluate(model, dev_loader, device)
        f1 = dev_metrics["macro_f1"]

        console.print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, "
                      f"dev_acc={dev_metrics['accuracy']:.3f}, dev_f1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            console.print(f"  -> Saved best model (F1={f1:.3f})")

    console.print(f"\nBest: epoch {best_epoch}, F1={best_f1:.3f}")
    console.print(f"Model saved to {output_dir}")

    # Final eval with best model
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    final_metrics = evaluate(model, dev_loader, device)

    console.print(f"\nFinal dev results:")
    console.print(f"  Accuracy: {final_metrics['accuracy']:.3f}")
    console.print(f"  Macro-F1: {final_metrics['macro_f1']:.3f}")
    for cls in ["SUPPORT", "CONTRADICT"]:
        pc = final_metrics["per_class"][cls]
        console.print(f"  {cls}: P={pc['precision']:.3f} R={pc['recall']:.3f} F1={pc['f1']:.3f}")

    # Save training log
    log = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "train_examples": len(train_dataset),
        "dev_examples": len(dev_dataset),
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "final_metrics": final_metrics,
    }
    with open(output_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
