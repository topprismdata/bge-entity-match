#!/usr/bin/env python3
"""
Phase 2: LoRA fine-tuning of BGE-M3 Dense encoder for entity matching.

Uses sentence-transformers + PEFT (LoRA) with CachedMultipleNegativesRankingLoss.
Only modifies Dense encoder, Sparse and ColBERT are frozen.
"""
import argparse
import json
import os
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator


def load_jsonl(path: str) -> list[dict]:
    samples = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


class EntityMatchingDataset(torch.utils.data.Dataset):
    """Dataset for MultipleNegativesRankingLoss with hard negatives."""

    def __init__(self, samples: list[dict]):
        # Include 1 hard negative per sample (3rd text in InputExample)
        self.examples = [
            InputExample(texts=[s['query'], s['positive'], s['negatives'][0]])
            for s in samples if s.get('negatives')
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune BGE-M3 Dense")
    parser.add_argument('--training-data', default='match_results/m3_cache/training_data.jsonl')
    parser.add_argument('--output-dir', default='match_results/m3_cache/lora_weights')
    parser.add_argument('--model', default='BAAI/bge-m3')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup-ratio', type=float, default=0.1)
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--mini-batch-size', type=int, default=32)
    args = parser.parse_args()

    t0 = time.time()

    # ── Load training data ──
    print(f"[1/5] Loading training data from {args.training_data}...", file=sys.stderr)
    samples = load_jsonl(args.training_data)
    print(f"  Loaded {len(samples)} samples", file=sys.stderr)

    # ── Train/val split ──
    import random
    random.seed(42)
    random.shuffle(samples)
    n_val = int(len(samples) * args.val_split)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}", file=sys.stderr)

    # ── Load model ──
    print(f"[2/5] Loading {args.model} via SentenceTransformer...", file=sys.stderr)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"  Device: {device}", file=sys.stderr)
    model = SentenceTransformer(args.model, device=device)

    # ── Apply LoRA ──
    print("[3/5] Applying LoRA to XLM-RoBERTa backbone...", file=sys.stderr)
    base_model = model[0].auto_model  # XLM-RoBERTa backbone
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=args.lora_dropout,
    )
    peft_model = get_peft_model(base_model, lora_config)
    trainable, total = peft_model.get_nb_trainable_parameters()
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)", file=sys.stderr)

    # ── Dataset + Loss ──
    print("[4/5] Setting up training...", file=sys.stderr)
    train_dataset = EntityMatchingDataset(train_samples)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
    )
    # Use MNR loss (CachedMNR requires gradient checkpointing, incompatible with MPS)
    train_loss = MultipleNegativesRankingLoss(model=model)

    # IR evaluator on val set
    queries_val = {str(i): s['query'] for i, s in enumerate(val_samples)}
    corpus_val = {}
    relevant_docs_val = {}
    for i, s in enumerate(val_samples):
        q_id = str(i)
        p_id = f"pos_{i}"
        corpus_val[p_id] = s['positive']
        relevant_docs_val.setdefault(q_id, set()).add(p_id)
        for j, neg in enumerate(s['negatives']):
            n_id = f"neg_{i}_{j}"
            corpus_val[n_id] = neg

    evaluator = InformationRetrievalEvaluator(
        queries=queries_val,
        corpus=corpus_val,
        relevant_docs=relevant_docs_val,
        show_progress_bar=True,
    )

    # ── Train ──
    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"[5/5] Training: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"lr={args.lr}, warmup_steps={warmup_steps}", file=sys.stderr)
    os.makedirs(args.output_dir, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        steps_per_epoch=None,
        optimizer_params={'lr': args.lr},
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        evaluator=evaluator,
        evaluation_steps=len(train_dataloader),  # eval once per epoch
        save_best_model=True,
        show_progress_bar=True,
    )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)", file=sys.stderr)
    print(f"Model saved to {args.output_dir}", file=sys.stderr)

    # Save training config
    config = {
        'model': args.model,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'target_modules': ['query', 'value'],
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'warmup_ratio': args.warmup_ratio,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'trainable_params': trainable,
        'total_params': total,
    }
    config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Config saved to {config_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
