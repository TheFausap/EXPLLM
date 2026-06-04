"""Train the QALF proof-of-concept."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from qalf.data import build_tokenizer, encode_examples, make_windows, read_jsonl, relation_counts, trigram_counts
from qalf.joblog import JobLogger
from qalf.model import QALFConfig, QALFModel, cross_entropy_with_l2, device_for_training, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QALF")
    parser.add_argument("--data", default="data/seed_corpus.jsonl")
    parser.add_argument("--out", default="runs/qalf_poc")
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--context-size", type=int, default=24)
    parser.add_argument("--relations", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.018)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--attractor-limit", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--trigram-top-k", type=int, default=32)
    parser.add_argument("--trigram-min-count", type=int, default=2)
    parser.add_argument("--trigram-strength", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    logger = JobLogger(args.log_file)
    examples = read_jsonl(args.data)
    logger.emit({"stage": "read_data", "examples": len(examples), "data": args.data})
    tokenizer = build_tokenizer(examples, vocab_size=args.vocab_size)
    logger.emit({"stage": "build_tokenizer", "vocab": len(tokenizer.vocab)})
    encoded = encode_examples(tokenizer, examples)
    contexts, prev2_tokens, prev_tokens, targets = make_windows(encoded, args.context_size, tokenizer.pad_id, include_prev2=True)
    if args.max_windows is not None and args.max_windows < targets.numel():
        order = torch.randperm(targets.numel())[: args.max_windows]
        contexts = contexts[order]
        prev2_tokens = prev2_tokens[order]
        prev_tokens = prev_tokens[order]
        targets = targets[order]
    window_ram_gb = (contexts.numel() + prev2_tokens.numel() + prev_tokens.numel() + targets.numel()) * contexts.element_size() / (1024 ** 3)
    logger.emit({"stage": "make_windows", "windows": int(targets.numel()), "context_size": args.context_size, "window_tensor_gb": window_ram_gb})
    bigram = relation_counts(encoded, len(tokenizer.vocab))
    trigram = trigram_counts(encoded, len(tokenizer.vocab), top_k=args.trigram_top_k, min_count=args.trigram_min_count)
    bigram_gb = bigram.numel() * bigram.element_size() / (1024 ** 3)
    trigram_gb = sum(t.numel() * t.element_size() for t in trigram.values()) / (1024 ** 3)
    logger.emit({"stage": "build_higher_order_memory", "trigram_contexts": int(trigram["keys"].numel()), "trigram_top_k": args.trigram_top_k, "bigram_gb": bigram_gb, "trigram_gb": trigram_gb})
    config = QALFConfig(
        vocab_size=len(tokenizer.vocab),
        dimension=args.dimension,
        context_size=args.context_size,
        num_relations=args.relations,
        trigram_strength=args.trigram_strength,
        pad_id=tokenizer.pad_id,
    )
    device = device_for_training(args.device)
    logger.emit({"stage": "init_model", "device": str(device), "dimension": args.dimension, "relations": args.relations})
    model = QALFModel(config, bigram_logits=bigram, trigram_prior=trigram).to(device)
    dataset = TensorDataset(contexts, prev2_tokens, prev_tokens, targets)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        seen = 0
        for batch_contexts, batch_prev2, batch_prev, batch_targets in loader:
            batch_contexts = batch_contexts.to(device)
            batch_prev2 = batch_prev2.to(device)
            batch_prev = batch_prev.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_contexts, batch_prev, batch_prev2)
            loss = cross_entropy_with_l2(model, logits, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * batch_targets.numel()
            seen += batch_targets.numel()
        avg_loss = total_loss / max(seen, 1)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            diag_contexts = contexts[: min(16, len(contexts))].to(device)
            diag = model.diagnostics(diag_contexts)
            record = {"epoch": epoch, "loss": avg_loss, **diag}
            history.append(record)
            logger.emit(record)
    out_dir = Path(args.out)
    metadata = {
        "data": args.data,
        "device": str(device),
        "epochs": args.epochs,
        "examples": len(examples),
        "seed_examples": [
            {"prompt": example.prompt, "reply": example.reply}
            for example in examples[: args.attractor_limit]
        ],
        "attractor_limit": args.attractor_limit,
        "windows": int(targets.numel()),
        "history": history,
    }
    save_checkpoint(out_dir / "model.pt", model.cpu(), tokenizer, metadata)
    logger.emit({"stage": "saved", "checkpoint": str(out_dir / "model.pt")})
    logger.close()


if __name__ == "__main__":
    main()
