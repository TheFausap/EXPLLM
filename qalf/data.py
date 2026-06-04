"""Dataset helpers for QALF."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from qalf.tokenizer import EncodedExample, QuantumTokenizer


@dataclass
class DialogueExample:
    prompt: str
    reply: str


def read_jsonl(path: str | Path) -> list[DialogueExample]:
    examples: list[DialogueExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "prompt" not in record or "reply" not in record:
                raise ValueError(f"{path}:{line_no} must contain prompt and reply")
            examples.append(DialogueExample(prompt=record["prompt"], reply=record["reply"]))
    return examples


def build_tokenizer(examples: Iterable[DialogueExample], vocab_size: int) -> QuantumTokenizer:
    texts: list[str] = []
    for example in examples:
        texts.append(example.prompt)
        texts.append(example.reply)
    return QuantumTokenizer.build(texts, vocab_size=vocab_size)


def encode_examples(tokenizer: QuantumTokenizer, examples: Iterable[DialogueExample]) -> list[EncodedExample]:
    return [tokenizer.encode_example(example.prompt, example.reply) for example in examples]


def make_windows(
    encoded: Iterable[EncodedExample],
    context_size: int,
    pad_id: int,
    include_prev2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = list(encoded)
    total = sum(max(len(example.sequence) - 1, 0) for example in examples)
    contexts = torch.full((total, context_size), pad_id, dtype=torch.long)
    prev_tokens = torch.empty(total, dtype=torch.long)
    targets = torch.empty(total, dtype=torch.long)
    prev2_tokens = torch.empty(total, dtype=torch.long) if include_prev2 else None

    row = 0
    for example in examples:
        seq = example.sequence
        for idx in range(1, len(seq)):
            start = max(0, idx - context_size)
            context = seq[start:idx]
            offset = context_size - len(context)
            contexts[row, offset:] = torch.tensor(context, dtype=torch.long)
            prev_tokens[row] = seq[idx - 1]
            targets[row] = seq[idx]
            if prev2_tokens is not None:
                prev2_tokens[row] = seq[idx - 2] if idx >= 2 else pad_id
            row += 1

    base = (contexts, prev_tokens, targets)
    if not include_prev2:
        return base
    return (contexts, prev2_tokens, prev_tokens, targets)


def relation_counts(encoded: Iterable[EncodedExample], vocab_size: int, smoothing: float = 0.05) -> torch.Tensor:
    counts = torch.full((vocab_size, vocab_size), smoothing, dtype=torch.float32)
    for example in encoded:
        seq = example.sequence
        for left, right in zip(seq[:-1], seq[1:]):
            counts[left, right] += 1.0
    probs = counts / counts.sum(dim=-1, keepdim=True)
    return probs.log()


def trigram_counts(
    encoded: Iterable[EncodedExample],
    vocab_size: int,
    top_k: int = 32,
    min_count: int = 2,
) -> dict[str, torch.Tensor]:
    """Build a sparse top-k trigram prior keyed by the previous two tokens."""
    counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    for example in encoded:
        seq = example.sequence
        for left, middle, right in zip(seq[:-2], seq[1:-1], seq[2:]):
            counts[(left, middle)][right] += 1

    rows: list[tuple[int, list[int], list[float]]] = []
    for (left, middle), counter in counts.items():
        total = sum(counter.values())
        if total < min_count:
            continue
        key = left * vocab_size + middle
        most_common = counter.most_common(top_k)
        token_ids = [token for token, _ in most_common]
        logits = [float(count / total) for _, count in most_common]
        rows.append((key, token_ids, logits))

    rows.sort(key=lambda row: row[0])
    keys = torch.tensor([row[0] for row in rows], dtype=torch.long)
    token_ids = torch.full((len(rows), top_k), -1, dtype=torch.long)
    token_logits = torch.zeros((len(rows), top_k), dtype=torch.float32)
    for row_idx, (_, ids, probs) in enumerate(rows):
        n = len(ids)
        token_ids[row_idx, :n] = torch.tensor(ids, dtype=torch.long)
        token_logits[row_idx, :n] = torch.tensor(probs, dtype=torch.float32).clamp_min(1e-8).log()
    return {"keys": keys, "token_ids": token_ids, "token_logits": token_logits}
