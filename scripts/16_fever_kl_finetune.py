#!/usr/bin/env python3
"""KL-disagreement fine-tuning on FEVER, with lambda ablation.

Mirrors scripts/12_finetune_kl_disagreement.py but loads FEVER instead of
SciFact and runs the full lambda ablation in one invocation by default
(0.0, 0.1, 0.3, 0.5). The key question: does the CONTRADICT F1 lift from
KL training on SciFact replicate on FEVER?

Loss = task_loss + lambda_kl * symmetric_KL(pred_bm25, pred_dense)

Run:
    python scripts/16_fever_kl_finetune.py
    python scripts/16_fever_kl_finetune.py --lambdas 0.0 0.1
    python scripts/16_fever_kl_finetune.py --lambda-kl 0.3   # single run
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from rich.console import Console
from tqdm import tqdm

from claimverify.data.fever import Fever
from claimverify.retrieval.bm25 import BM25Retriever
from claimverify.retrieval.dense import DenseRetriever
from claimverify.retrieval.disagreement import jaccard_at_k
from claimverify.evaluation.metrics import macro_f1

console = Console()

LABEL2ID = {"SUPPORT": 0, "CONTRADICT": 1, "NEI": 2}
ID2LABEL = {0: "SUPPORT", 1: "CONTRADICT", 2: "NEI"}


def symmetric_kl(p_logits, q_logits):
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)
    kl_pq = F.kl_div(q, p_probs, reduction="batchmean", log_target=False)
    kl_qp = F.kl_div(p, q_probs, reduction="batchmean", log_target=False)
    return 0.5 * (kl_pq + kl_qp)


def precompute_retrieval(claims, abstracts, bm25, dense, top_k=5):
    results = {}
    for claim in tqdm(claims, desc="Pre-computing retrieval"):
        bm25_top = bm25.retrieve(claim.text, top_k=top_k)
        dense_top = dense.retrieve(claim.text, top_k=top_k)
        bm25_100 = bm25.retrieve(claim.text, top_k=100)
        dense_100 = dense.retrieve(claim.text, top_k=100)
        jaccard = jaccard_at_k(bm25_100, dense_100, k=10)

        bm25_docs = {d: abstracts[d].sentences for d, _ in bm25_top if d in abstracts}
        dense_docs = {d: abstracts[d].sentences for d, _ in dense_top if d in abstracts}

        results[claim.claim_id] = {
            "bm25_docs": bm25_docs,
            "dense_docs": dense_docs,
            "jaccard": jaccard,
        }
    return results


class SentenceLevelDataset(Dataset):
    def __init__(self, claims, abstracts, tokenizer, max_length=256, neg_ratio=3.0, seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        rng = random.Random(seed)

        for claim in claims:
            for doc_id, anns in claim.evidence.items():
                if doc_id not in abstracts:
                    continue
                sentences = abstracts[doc_id].sentences
                if not sentences:
                    continue

                gold_indices: set[int] = set()
                label_for_doc = None
                for ann in anns:
                    label = ann["label"]
                    if label not in ("SUPPORT", "CONTRADICT"):
                        continue
                    label_for_doc = label
                    for s in ann["sentences"]:
                        if 0 <= s < len(sentences):
                            gold_indices.add(s)

                if not gold_indices or label_for_doc is None:
                    continue

                for s in gold_indices:
                    self.examples.append({
                        "claim": claim.text,
                        "claim_id": claim.claim_id,
                        "sentence": sentences[s],
                        "label": LABEL2ID[label_for_doc],
                    })

                non_rationale = [i for i in range(len(sentences)) if i not in gold_indices]
                n_neg = min(len(non_rationale), int(neg_ratio * len(gold_indices)))
                for s in rng.sample(non_rationale, n_neg):
                    self.examples.append({
                        "claim": claim.text,
                        "claim_id": claim.claim_id,
                        "sentence": sentences[s],
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
                        max_length=256, no_grad=False, max_sents=8):
    all_sents = []
    for _, sentences in doc_sentences.items():
        all_sents.extend(sentences)
    if not all_sents:
        return None
    if len(all_sents) > max_sents:
        all_sents = all_sents[:max_sents]

    inputs = tokenizer(
        [claim_text] * len(all_sents),
        all_sents,
        truncation=True, max_length=max_length,
        padding=True, return_tensors="pt",
    ).to(device)

    use_amp = device == "cuda"
    if no_grad:
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(**inputs)
        return outputs.logits.mean(dim=0, keepdim=True).detach()
    else:
        with torch.amp.autocast("cuda", enabled=use_amp):
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


def train_one_lambda(args, lambda_kl, fv, train_retrieval, kl_claim_ids,
                     train_claim_map, device):
    """Run one fine-tuning trial at a specific lambda. Returns metrics dict."""
    console.print(f"\n[bold blue]── lambda_kl = {lambda_kl} ──[/bold blue]")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=3, ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,
    )
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID
    model.to(device)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    train_dataset = SentenceLevelDataset(
        fv.train_claims, fv.abstracts, tokenizer,
        max_length=args.max_length, neg_ratio=args.neg_ratio, seed=args.seed,
    )
    dev_dataset = SentenceLevelDataset(
        fv.dev_claims, fv.abstracts, tokenizer,
        max_length=args.max_length, neg_ratio=args.neg_ratio, seed=args.seed + 1,
    )
    console.print(f"  Train: {len(train_dataset)} examples  Dev: {len(dev_dataset)} examples")
    for lbl, cnt in train_dataset.label_counts.items():
        console.print(f"    {lbl}: {cnt}")

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

    output_dir = Path(args.output_dir_template.format(lambda_kl=lambda_kl))
    output_dir.mkdir(parents=True, exist_ok=True)

    kl_rng = random.Random(args.seed + 100)
    epoch_kl_losses = []
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    last_epoch = 0

    for epoch in range(args.epochs):
        last_epoch = epoch
        model.train()
        total_task_loss = 0.0
        total_kl_loss = 0.0
        n_kl_steps = 0
        optimizer.zero_grad()
        step = 0

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                task_loss = task_loss_fn(outputs.logits, labels) / args.grad_accum

            kl_loss_value = 0.0
            compute_kl = (
                lambda_kl > 0
                and (step + 1) % args.kl_every_n == 0
                and len(kl_claim_ids) > 0
            )

            if compute_kl:
                cid = kl_rng.choice(kl_claim_ids)
                claim_text = train_claim_map[cid]
                retrieval = train_retrieval[cid]

                with torch.amp.autocast("cuda", enabled=use_amp):
                    bm25_logits = compute_view_logits(
                        model, tokenizer, claim_text, retrieval["bm25_docs"],
                        device, args.max_length, no_grad=False, max_sents=8,
                    )
                    dense_logits = compute_view_logits(
                        model, tokenizer, claim_text, retrieval["dense_docs"],
                        device, args.max_length, no_grad=True, max_sents=8,
                    )
                    if bm25_logits is not None and dense_logits is not None:
                        kl_term = symmetric_kl(bm25_logits, dense_logits)
                        kl_loss = lambda_kl * kl_term / args.grad_accum
                        kl_loss_value = kl_term.item()
                    else:
                        kl_loss = torch.tensor(0.0, device=device)
            else:
                kl_loss = torch.tensor(0.0, device=device)

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

            pbar.set_postfix(
                task=f"{task_loss.item() * args.grad_accum:.3f}",
                kl=f"{(total_kl_loss / max(n_kl_steps, 1)):.3f}",
            )

        if (step + 1) % args.grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        avg_task = total_task_loss / max(len(train_loader), 1)
        avg_kl = total_kl_loss / max(n_kl_steps, 1)
        epoch_kl_losses.append(avg_kl)

        dev_metrics = evaluate(model, dev_loader, device)
        f1 = dev_metrics["macro_f1"]
        console.print(f"    Epoch {epoch+1}: task={avg_task:.4f} kl={avg_kl:.4f} "
                      f"acc={dev_metrics['accuracy']:.3f} f1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            patience_counter = 0
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                console.print(f"    Early stop at epoch {epoch+1}")
                break

    # Final eval on best checkpoint.
    model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    final_metrics = evaluate(model, dev_loader, device)

    return {
        "lambda_kl": lambda_kl,
        "epochs_run": last_epoch + 1,
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "final_metrics": final_metrics,
        "epoch_avg_kl_losses": epoch_kl_losses,
        "output_dir": str(output_dir),
        "train_examples": len(train_dataset),
        "dev_examples": len(dev_dataset),
    }


def main():
    parser = argparse.ArgumentParser(description="FEVER KL-disagreement fine-tuning + lambda ablation")
    parser.add_argument("--base-model", default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--neg-ratio", type=float, default=3.0)
    parser.add_argument("--kl-top-k", type=int, default=3)
    parser.add_argument("--kl-every-n", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--output-dir-template", default="models/joint-kl-fever-lambda{lambda_kl}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-dev-claims", type=int, default=1000)
    parser.add_argument("--max-train-claims", type=int, default=5000)
    parser.add_argument("--max-corpus-docs", type=int, default=50_000)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.1, 0.3, 0.5],
                        help="Lambda values to sweep. Default ablation: 0.0 0.1 0.3 0.5.")
    parser.add_argument("--lambda-kl", type=float, default=None,
                        help="Run a single lambda (overrides --lambdas).")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lambdas = [args.lambda_kl] if args.lambda_kl is not None else args.lambdas

    console.print(f"\n[bold blue]FEVER KL Fine-tuning — Lambda Ablation[/bold blue]")
    console.print(f"  Device: {device}  Base: {args.base_model}")
    console.print(f"  Lambdas: {lambdas}")
    console.print(f"  Epochs: {args.epochs}  LR: {args.lr}  Batch: {args.batch_size}x{args.grad_accum}\n")

    # ── Load FEVER (shared across all lambda runs) ───────────────────
    console.print("Loading FEVER...")
    fv = Fever.load(
        max_dev_claims=args.max_dev_claims,
        max_corpus_docs=args.max_corpus_docs,
        include_train=True,
        max_train_claims=args.max_train_claims,
    )
    console.print(f"  Train claims: {len(fv.train_claims)}  Dev claims: {len(fv.dev_claims)}")
    console.print(f"  Corpus: {fv.corpus_size}")
    console.print(f"  Dev label dist: {fv.label_distribution('dev')}")

    corpus = fv.get_corpus_texts()

    console.print("Building BM25 index...")
    bm25 = BM25Retriever(k1=1.2, b=0.75)
    bm25.build_index(corpus)

    console.print(f"Building dense index [{args.dense_model}]...")
    dense_retriever = DenseRetriever(model_name=args.dense_model)
    dense_retriever.build_index(corpus)

    console.print("Pre-computing retrieval for training claims...")
    train_retrieval = precompute_retrieval(
        fv.train_claims, fv.abstracts, bm25, dense_retriever, top_k=args.kl_top_k,
    )
    train_claim_map = {c.claim_id: c.text for c in fv.train_claims}
    kl_claim_ids = [
        cid for cid, r in train_retrieval.items()
        if r["bm25_docs"] and r["dense_docs"]
    ]
    console.print(f"  Claims with both retrieval views: {len(kl_claim_ids)}")

    # ── Run each lambda ──────────────────────────────────────────────
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    runs = []
    for lam in lambdas:
        run_metrics = train_one_lambda(
            args, lam, fv, train_retrieval, kl_claim_ids,
            train_claim_map, device,
        )
        runs.append(run_metrics)

        # Persist incrementally so a crash in one lambda doesn't lose earlier ones.
        partial = {
            "dataset": "fever",
            "base_model": args.base_model,
            "dense_model": args.dense_model,
            "config": {
                "epochs": args.epochs, "batch_size": args.batch_size,
                "grad_accum": args.grad_accum, "lr": args.lr,
                "max_length": args.max_length, "neg_ratio": args.neg_ratio,
                "kl_top_k": args.kl_top_k, "kl_every_n": args.kl_every_n,
                "max_dev_claims": args.max_dev_claims,
                "max_train_claims": args.max_train_claims,
                "max_corpus_docs": args.max_corpus_docs,
                "kl_eligible_claims": len(kl_claim_ids),
            },
            "runs": runs,
        }
        with open(results_dir / "16_fever_kl.json", "w") as f:
            json.dump(partial, f, indent=2, default=str)

    # ── Summary ──────────────────────────────────────────────────────
    console.print(f"\n[bold]Lambda ablation summary (FEVER):[/bold]")
    for r in runs:
        fm = r["final_metrics"]
        contradict = fm["per_class"].get("CONTRADICT", {})
        console.print(
            f"  λ={r['lambda_kl']:>4}  "
            f"acc={fm['accuracy']:.3f}  f1={fm['macro_f1']:.3f}  "
            f"CONTRADICT_F1={contradict.get('f1', 0.0):.3f}"
        )

    console.print(f"\nResults saved to {results_dir / '16_fever_kl.json'}")


if __name__ == "__main__":
    main()
