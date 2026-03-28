#!/usr/bin/env python3
"""Compute true grad_norm (no noise, no clipping) for saved checkpoints.

Loads the base model + LoRA adapter, runs forward-backward on a batch of
data, and reports the gradient norm. This gives the actual signal strength
at each checkpoint, comparable across baseline and DP models.

Usage:
    python eval_grad_norm.py \
        --init-model /path/to/base_model \
        --checkpoints baseline/checkpoint-final dp_eps3/checkpoint-final dp_eps8/checkpoint-final \
        --data-path /path/to/clinical_episodes.jsonl \
        --batch-size 64 --max-length 512 --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


def compute_loss_and_grad_norm(model, batch, vocab_size, phys_batch=1):
    """Forward-backward with micro-batching, return (mean_loss, grad_norm)."""
    model.train()
    model.zero_grad()

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    B = input_ids.size(0)
    total_loss = 0.0

    for start in range(0, B, phys_batch):
        end = min(start + phys_batch, B)
        ids = input_ids[start:end]
        mask = attention_mask[start:end]

        outputs = model(input_ids=ids, attention_mask=mask)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        shift_mask = mask[:, 1:].float()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, vocab_size), shift_labels.view(-1),
        ).view(ids.size(0), -1)

        per_example_loss = (per_token_loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
        # Sum aggregation, /B to match Opacus normalization
        mb_loss = per_example_loss.sum() / B
        mb_loss.backward()
        total_loss += per_example_loss.sum().item()

    grad_norm = sum(
        p.grad.float().norm().item() ** 2
        for p in model.parameters() if p.requires_grad and p.grad is not None
    ) ** 0.5

    return total_loss / B, grad_norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-model", required=True, help="Base model (before LoRA)")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint dirs with adapter/")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-batches", type=int, default=5, help="Average over N batches")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.init_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    texts = []
    with open(args.data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])

    # Tokenize a fixed set of batches
    all_batches = []
    idx = 0
    for _ in range(args.num_batches):
        batch_texts = texts[idx:idx + args.batch_size]
        idx += args.batch_size
        if len(batch_texts) < args.batch_size:
            break
        enc = tokenizer(
            batch_texts, max_length=args.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        all_batches.append({k: v.to(device) for k, v in enc.items()})

    print(f"Data: {len(all_batches)} batches x {args.batch_size} examples")
    print()

    # Evaluate each checkpoint
    results = []
    for ckpt_path in args.checkpoints:
        ckpt = Path(ckpt_path)
        adapter_dir = ckpt / "adapter" if (ckpt / "adapter").exists() else ckpt
        name = ckpt.parent.name if ckpt.name == "checkpoint-final" else ckpt.name

        # Load fresh base model + adapter
        base_model = AutoModelForCausalLM.from_pretrained(args.init_model, torch_dtype=torch.float32)
        vocab_size = base_model.config.vocab_size
        model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=True)
        model = model.to(device)

        # Compute grad_norm over multiple batches
        losses, norms = [], []
        for batch in all_batches:
            loss, gn = compute_loss_and_grad_norm(model, batch, vocab_size)
            losses.append(loss)
            norms.append(gn)

        avg_loss = sum(losses) / len(losses)
        avg_norm = sum(norms) / len(norms)
        results.append((name, avg_loss, avg_norm))
        print(f"{name:15s}: loss={avg_loss:.4f}  grad_norm={avg_norm:.4f}")

        del model, base_model
        torch.cuda.empty_cache()

    print()
    print("=" * 50)
    print(f"{'Name':15s}  {'Loss':>10s}  {'grad_norm':>10s}")
    for name, loss, norm in results:
        print(f"{name:15s}  {loss:10.4f}  {norm:10.4f}")


if __name__ == "__main__":
    main()
