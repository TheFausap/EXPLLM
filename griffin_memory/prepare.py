"""Prepare document-separated, memory-mappable data and a from-scratch tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .tokenizer import Tokenizer, BOS, EOS


def documents(paths, separator):
    for filename in paths:
        path = Path(filename)
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if "text" in record:
                        text = record["text"]
                    elif "prompt" in record and "reply" in record:
                        text = record["prompt"] + "\n\n" + record["reply"]
                    else:
                        raise ValueError(f"{path}:{line_no}: expected text or prompt/reply")
                    if not isinstance(text, str):
                        raise ValueError(f"{path}:{line_no}: text must be a string")
                    if text.strip():
                        yield text, {"id": str(record.get("id", f"{path.name}:{line_no}")),
                                     "source": str(record.get("source", path)),
                                     "metadata": record.get("metadata", {})}
        else:
            # One file = one document unless an explicit separator is provided.
            text = path.read_text(encoding="utf-8")
            for i, part in enumerate(text.split(separator) if separator else [text]):
                if part.strip():
                    yield part, {"id": f"{path.name}:{i}", "source": str(path), "metadata": {}}


def tokenizer_texts(raw_path, max_characters=0, seed=42):
    """Bound BPE training RAM; sample short spans across the *training* spool.

    The two-pass proportional budget avoids feeding only the first long PG-19
    book to BPE. It does not truncate the documents subsequently tokenized.
    """
    import random
    lengths = []
    if max_characters:
        with Path(raw_path).open(encoding="utf-8") as f:
            lengths = [len(json.loads(line)["text"]) for line in f]
    total = sum(lengths)
    bounded = bool(max_characters and total > max_characters)
    remaining = min(max_characters, total) if max_characters else 0
    rng = random.Random(seed)
    with Path(raw_path).open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            text = json.loads(line)["text"]
            if not bounded:
                yield text
                continue
            # Allocate remaining characters proportionally across all documents.
            length = lengths[index]
            allowance = min(length, remaining * length // total)
            total -= length
            remaining -= allowance
            # Short spans control long-document BPE pretokenization overhead.
            while allowance:
                take = min(4096, allowance)
                start = rng.randrange(max(1, length - take + 1))
                yield text[start:start + take]
                allowance -= take


def prepare(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"Refusing to overwrite existing dataset: {out}")
    if not 0 <= args.val_fraction < 1:
        raise ValueError("val-fraction must be in [0, 1)")
    validation_input = getattr(args, "validation_input", None)
    test_input = getattr(args, "test_input", None)
    tokenizer_file = getattr(args, "tokenizer_file", None)
    tokenizer_max_characters = getattr(args, "tokenizer_max_characters", 0)
    if tokenizer_max_characters < 0:
        raise ValueError("tokenizer-max-characters must be nonnegative")
    if test_input and not validation_input:
        raise ValueError("--test-input requires --validation-input (explicit source splits)")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".prepare-", dir=out.parent))
    splits = ["train", "val"] + (["test"] if test_input else [])
    try:
        counts = {split: 0 for split in splits}

        def write_record(target, split, text, metadata):
            digest = hashlib.sha256(text.encode()).hexdigest()
            target.write(json.dumps({"text": text, **metadata, "sha256": digest}, ensure_ascii=False) + "\n")
            counts[split] += 1

        if validation_input:
            # Explicit official splits are never hash-repartitioned.
            inputs = {"train": args.input, "val": validation_input, "test": test_input}
            for split in splits:
                with (staging / f"{split}.raw").open("w", encoding="utf-8") as target:
                    for text, metadata in documents(inputs[split], args.document_separator):
                        write_record(target, split, text, metadata)
        else:
            # Spool to disk, so BPE and tokenization do not require the corpus in RAM.
            with (staging / "train.raw").open("w", encoding="utf-8") as train, \
                 (staging / "val.raw").open("w", encoding="utf-8") as val:
                for text, metadata in documents(args.input, args.document_separator):
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    split_hash = hashlib.sha256(f"{args.seed}:{digest}".encode()).digest()
                    split = "val" if int.from_bytes(split_hash[:8], "big") / 2**64 < args.val_fraction else "train"
                    write_record(val if split == "val" else train, split, text, metadata)
        require_val = bool(validation_input) or args.val_fraction > 0
        if not counts["train"] or (require_val and not counts["val"]) or (test_input and not counts["test"]):
            raise ValueError("Empty train/validation/test split. Add documents, change seed/fraction, or use --val-fraction 0.")

        tokenizer = (Tokenizer.load(tokenizer_file) if tokenizer_file else
                     Tokenizer.train_bpe(tokenizer_texts(staging / "train.raw", tokenizer_max_characters, args.seed),
                                         args.vocab_size) if args.tokenizer == "bpe" else Tokenizer())
        tokenizer.save(staging / "tokenizer.json")
        stats = {}
        for split in splits:
            offsets = [0]
            with (staging / f"{split}.bin").open("wb") as tokens, \
                 (staging / f"{split}.metadata.jsonl").open("w", encoding="utf-8") as meta, \
                 (staging / f"{split}.raw").open(encoding="utf-8") as raw:
                for line in raw:
                    record = json.loads(line)
                    ids = [BOS] + tokenizer.encode(record.pop("text")) + [EOS]
                    np.asarray(ids, dtype="<u4").tofile(tokens)
                    offsets.append(offsets[-1] + len(ids))
                    meta.write(json.dumps(record, ensure_ascii=False) + "\n")
            np.save(staging / f"{split}.offsets.npy", np.array(offsets, dtype=np.int64))
            stats[split] = {"documents": len(offsets) - 1, "tokens": offsets[-1],
                            "prediction_targets": offsets[-1] - (len(offsets) - 1)}
            (staging / f"{split}.raw").unlink()
        provenance = getattr(args, "provenance", None)
        if provenance:
            # Parse before copying, so malformed provenance fails atomically.
            source = json.loads(Path(provenance).read_text(encoding="utf-8"))
            (staging / "sources.json").write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8")
        hashes = {}
        for path in sorted(staging.iterdir()):
            with path.open("rb") as f:
                hashes[path.name] = hashlib.file_digest(f, "sha256").hexdigest()
        manifest = {"version": 1, "dtype": "<u4", "vocab_size": tokenizer.vocab_size,
                    "seed": args.seed, "split_policy": "explicit" if validation_input else "content_hash",
                    "tokenizer_training": {"reused": bool(tokenizer_file),
                                           "max_characters": tokenizer_max_characters},
                    "stats": stats, "sha256": hashes}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(out)
        print(json.dumps(manifest, indent=2))
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", nargs="+", required=True, help="UTF-8 .txt or document-per-line .jsonl")
    p.add_argument("--validation-input", nargs="+", help="Explicit validation documents; disables hash splitting")
    p.add_argument("--test-input", nargs="+", help="Optional untouched test documents (requires explicit validation)")
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", choices=["byte", "bpe"], default="byte")
    p.add_argument("--tokenizer-file", help="Reuse an existing tokenizer instead of training one")
    p.add_argument("--tokenizer-max-characters", type=int, default=0, help="Cap BPE training text; 0 means unlimited")
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--document-separator", default=None, help="Literal separator, e.g. <|endoftext|>")
    p.add_argument("--provenance", help="Optional source-provenance JSON to include in the manifest")
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
