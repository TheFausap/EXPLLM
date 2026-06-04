"""Evaluate QALF with fixed prompts and write sample outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qalf.attractor import PromptAttractor
from qalf.data import encode_examples, make_windows, read_jsonl
from qalf.joblog import JobLogger
from qalf.model import device_for_training, load_checkpoint


DEFAULT_PROMPTS = [
    "What is QALF?",
    "Explain the density context in one sentence.",
    "How does the model choose the next word?",
    "What happens when the answer is uncertain?",
    "Give a tiny example of phase in language.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate QALF")
    parser.add_argument("--checkpoint", default="runs/qalf_poc/model.pt")
    parser.add_argument("--data", default="data/seed_corpus.jsonl")
    parser.add_argument("--out", default="runs/qalf_poc/eval.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=18)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--no-attractor", action="store_true")
    parser.add_argument("--attractor-threshold", type=float, default=0.34)
    parser.add_argument("--attractor-data", default=None, help="Optional JSONL prompt/reply data for the attractor")
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = JobLogger(args.log_file)
    device = device_for_training(args.device)
    logger.emit({"stage": "eval_start", "checkpoint": args.checkpoint, "data": args.data, "device": str(device)})
    model, tokenizer, metadata = load_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    model.eval()
    examples = read_jsonl(args.data)
    encoded = encode_examples(tokenizer, examples)
    contexts, prev2_tokens, prev_tokens, targets = make_windows(encoded, model.config.context_size, tokenizer.pad_id, include_prev2=True)
    logger.emit({"stage": "eval_windows", "windows": int(targets.numel()), "eval_batch_size": args.eval_batch_size})
    with torch.no_grad():
        total_loss = 0.0
        total_seen = 0
        for start in range(0, targets.numel(), args.eval_batch_size):
            end = min(start + args.eval_batch_size, targets.numel())
            logits = model(contexts[start:end].to(device), prev_tokens[start:end].to(device), prev2_tokens[start:end].to(device))
            batch_targets = targets[start:end].to(device)
            batch_loss = torch.nn.functional.cross_entropy(logits, batch_targets, reduction="sum")
            total_loss += float(batch_loss.detach().cpu())
            total_seen += int(batch_targets.numel())
        loss = torch.tensor(total_loss / max(total_seen, 1))
        perplexity = float(torch.exp(loss).detach().cpu())
        diagnostics = model.diagnostics(contexts[: min(16, len(contexts))].to(device))
    samples = []
    attractor = None
    attractor_size = 0
    if not args.no_attractor:
        if args.attractor_data:
            attractor_examples = [
                {"prompt": example.prompt, "reply": example.reply}
                for example in read_jsonl(args.attractor_data)
            ]
        else:
            attractor_examples = metadata.get("seed_examples", [])
        attractor_size = len(attractor_examples)
        if attractor_examples:
            attractor = PromptAttractor(tokenizer, attractor_examples, threshold=args.attractor_threshold)
    for idx, prompt in enumerate(DEFAULT_PROMPTS):
        match = attractor.match(prompt) if attractor is not None else None
        if match is None:
            reply = model.generate(
                tokenizer,
                prompt,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=args.seed + idx,
                device=device,
            )
            source = "born_decoder"
            score = None
        else:
            reply = match.reply
            source = "prompt_attractor"
            score = match.score
        samples.append({"prompt": prompt, "reply": reply, "source": source, "attractor_score": score})
    result = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "training_metadata": compact_metadata(metadata),
        "loss": float(loss.detach().cpu()),
        "perplexity": perplexity,
        "diagnostics": diagnostics,
        "samples": samples,
        "generation": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "attractor": not args.no_attractor,
            "attractor_threshold": args.attractor_threshold,
            "attractor_data": args.attractor_data,
            "attractor_size": attractor_size,
            "eval_batch_size": args.eval_batch_size,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    logger.emit({"stage": "eval_complete", "loss": result["loss"], "perplexity": result["perplexity"], "out": str(out_path)})
    logger.close()


def compact_metadata(metadata: dict) -> dict:
    compact = dict(metadata)
    seed_examples = compact.pop("seed_examples", None)
    if seed_examples is not None:
        compact["seed_examples_count"] = len(seed_examples)
    return compact


if __name__ == "__main__":
    main()
