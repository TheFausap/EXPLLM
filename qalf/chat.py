"""Interactive chat for QALF."""

from __future__ import annotations

import argparse

from qalf.attractor import PromptAttractor
from qalf.data import read_jsonl
from qalf.model import device_for_training, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with QALF")
    parser.add_argument("--checkpoint", default="runs/qalf_poc/model.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=18)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--no-attractor", action="store_true")
    parser.add_argument("--attractor-threshold", type=float, default=0.34)
    parser.add_argument("--attractor-data", default=None, help="Optional JSONL prompt/reply data for the attractor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = device_for_training(args.device)
    model, tokenizer, metadata = load_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    model.eval()
    attractor = None
    if not args.no_attractor:
        if args.attractor_data:
            examples = [
                {"prompt": example.prompt, "reply": example.reply}
                for example in read_jsonl(args.attractor_data)
            ]
        else:
            examples = metadata.get("seed_examples", [])
        if examples:
            attractor = PromptAttractor(tokenizer, examples, threshold=args.attractor_threshold)
    if args.prompt is not None:
        if attractor is not None:
            reply = attractor.reply_or_none(args.prompt)
            if reply is not None:
                print(reply)
                return
        print(model.generate(tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_k, seed=args.seed, device=device))
        return
    print(f"QALF ready on {device}. Training device was {metadata.get('device', 'unknown')}. Type /quit to exit.")
    while True:
        try:
            prompt = input("user> ").strip()
        except EOFError:
            break
        if prompt in {"/quit", "/exit"}:
            break
        if not prompt:
            continue
        reply = attractor.reply_or_none(prompt) if attractor is not None else None
        if reply is None:
            reply = model.generate(
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=args.seed,
                device=device,
            )
        print(f"qalf> {reply}")


if __name__ == "__main__":
    main()
