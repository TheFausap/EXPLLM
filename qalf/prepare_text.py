"""Prepare larger plain-text corpora for QALF training."""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

from qalf.data import read_jsonl
from qalf.joblog import JobLogger
from qalf.tokenizer import detokenize, tokenize_text


TINYSTORIES_VALID_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
TINYSTORIES_TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare text data for QALF")
    parser.add_argument("--source", default="tinystories-valid", choices=["tinystories-valid", "tinystories-train", "file"])
    parser.add_argument("--input", default=None, help="Local text path when --source file is used")
    parser.add_argument("--download-to", default=None, help="Optional path for downloaded raw text")
    parser.add_argument("--out", default="data/tinystories_qalf.jsonl")
    parser.add_argument("--mix-seed", default="data/seed_corpus.jsonl")
    parser.add_argument("--max-examples", type=int, default=2000)
    parser.add_argument("--prompt-tokens", type=int, default=16)
    parser.add_argument("--reply-tokens", type=int, default=80)
    parser.add_argument("--min-story-tokens", type=int, default=40)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def read_source(args: argparse.Namespace) -> str:
    if args.source == "file":
        if args.input is None:
            raise ValueError("--input is required with --source file")
        return Path(args.input).read_text(encoding="utf-8", errors="replace")
    url = TINYSTORIES_VALID_URL if args.source == "tinystories-valid" else TINYSTORIES_TRAIN_URL
    raw_path = Path(args.download_to or f"data/{args.source}.txt")
    if not raw_path.exists() or looks_like_lfs_pointer(raw_path):
        print(f"downloading {url} -> {raw_path}")
        download(url, raw_path)
    return raw_path.read_text(encoding="utf-8", errors="replace")


def looks_like_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:128]
    except OSError:
        return False
    return "git-lfs.github.com/spec" in head


def split_stories(text: str) -> list[str]:
    if "<|endoftext|>" in text:
        chunks = text.split("<|endoftext|>")
    else:
        chunks = re.split(r"\n\s*\n", text)
    stories = []
    for chunk in chunks:
        clean = re.sub(r"\s+", " ", chunk).strip()
        if clean:
            stories.append(clean)
    return stories


def story_to_example(story: str, prompt_tokens: int, reply_tokens: int) -> dict[str, str] | None:
    tokens = tokenize_text(story)
    if len(tokens) < prompt_tokens + 8:
        return None
    prompt = detokenize(tokens[:prompt_tokens])
    reply = detokenize(tokens[prompt_tokens : prompt_tokens + reply_tokens])
    if not prompt or not reply:
        return None
    return {"prompt": f"Continue this story: {prompt}", "reply": reply}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    logger = JobLogger(args.log_file)
    raw = read_source(args)
    logger.emit({"stage": "read_source", "source": args.source, "chars": len(raw)})
    stories = split_stories(raw)
    logger.emit({"stage": "split_stories", "stories": len(stories)})
    random.shuffle(stories)
    examples: list[dict[str, str]] = []
    if args.mix_seed:
        for example in read_jsonl(args.mix_seed):
            examples.append({"prompt": example.prompt, "reply": example.reply})
    for story in stories:
        tokens = tokenize_text(story)
        if len(tokens) < args.min_story_tokens:
            continue
        example = story_to_example(story, args.prompt_tokens, args.reply_tokens)
        if example is not None:
            examples.append(example)
        if len(examples) >= args.max_examples:
            break
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=True) + "\n")
    result = {"stage": "prepared", "out": str(out), "examples": len(examples), "stories_seen": len(stories)}
    logger.emit(result)
    logger.close()


if __name__ == "__main__":
    main()
