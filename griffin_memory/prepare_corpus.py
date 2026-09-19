"""Download TinyStories or PG-19 and prepare bounded, split-safe starter corpora.

TinyStories: pinned Hugging Face text files (no remote dataset scripts).
PG-19: official public GCS objects, pinned by generation and MD5 checksum.
Downloads live in a persistent cache; intermediate JSONL is temporary.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import http.client
import json
from pathlib import Path
import random
import re
import shutil
import sqlite3
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from .prepare import prepare

TINY_REPO = "roneneldan/TinyStories"
PG_BUCKET = "deepmind-gutenberg"
PG_REVISION = "b0e1f2925359688af19f8784c23c3deeda52d3ea"
PG_METADATA = f"https://raw.githubusercontent.com/google-deepmind/pg19/{PG_REVISION}/metadata.csv"
PRESETS = {
    "tinystories": {"train_limit": 100_000, "val_limit": 2_000, "license": "CDLA-Sharing-1.0",
                    "source": f"https://huggingface.co/datasets/{TINY_REPO}"},
    "pg19": {"train_limit": 1_000, "val_limit": 50, "license": "Apache-2.0 (dataset distribution)",
             "source": "https://github.com/google-deepmind/pg19"},
}


def sha256_file(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def request(url, timeout):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        "User-Agent": "EXPLLM-griffin-memory/1.0", "Accept-Encoding": "identity"}), timeout=timeout)


def retryable(error):
    return not isinstance(error, urllib.error.HTTPError) or error.code in {408, 429, 500, 502, 503, 504}


def fetch_json(url, timeout=60, retries=4):
    for attempt in range(retries):
        try:
            with request(url, timeout) as response:
                return json.load(response)
        except (OSError, ValueError, http.client.HTTPException) as error:
            if attempt + 1 == retries or not retryable(error):
                raise
            time.sleep(min(2 ** attempt, 10))


def validate_download(path, size=None, md5=None):
    if not path.is_file() or (size is not None and path.stat().st_size != size):
        return False
    if md5:
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "md5").digest()
        if base64.b64encode(digest).decode() != md5:
            return False
    return True


def download_file(url, path, size=None, md5=None, timeout=60, retries=4):
    """Atomic, checksum-checked small-file download; retry only transient errors.

    PG-19 books are separate objects, so interrupted preparation reuses every
    already-verified book. Only the interrupted book is restarted.
    """
    path = Path(path)
    if validate_download(path, size, md5):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        # Unique partial names also avoid two preparations corrupting a cache file.
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".download-", delete=False) as f:
            part = Path(f.name)
        try:
            with request(url, timeout) as response, part.open("wb") as f:
                shutil.copyfileobj(response, f, length=1024 * 1024)
                advertised = response.headers.get("Content-Length")
            expected_size = size if size is not None else int(advertised) if advertised else None
            if not validate_download(part, expected_size, md5):
                raise OSError(f"Checksum/length mismatch downloading {url}")
            part.replace(path)
            return path
        except (OSError, ValueError, http.client.HTTPException) as error:
            if attempt + 1 == retries or not retryable(error):
                raise
            time.sleep(min(2 ** attempt, 10))
        finally:
            part.unlink(missing_ok=True)


def iter_stories(path, block_size=1024 * 1024):
    """Read delimiter-framed stories, including delimiters split across blocks."""
    marker, pending = "<|endoftext|>", ""
    with Path(path).open(encoding="utf-8", newline=None) as f:
        while block := f.read(block_size):
            parts = (pending + block).split(marker)
            pending = parts.pop()
            for part in parts:
                if part.strip():
                    yield part.strip()
        if pending.strip():
            yield pending.strip()


def selected_stories(path, limit, seed):
    # Sample document indices, not text: memory scales with the selected count.
    # A bounded subset takes two sequential disk scans but is not a prefix sample.
    selected = None
    if limit:
        count = sum(1 for _ in iter_stories(path))
        selected = set(random.Random(seed).sample(range(count), min(limit, count)))
    for index, text in enumerate(iter_stories(path)):
        if selected is None or index in selected:
            yield index, text


def tiny_sources(args, provenance):
    if args.include_test:
        raise ValueError("TinyStories has no official test split; --include-test is only for PG-19")
    from huggingface_hub import HfApi, hf_hub_download
    revision = HfApi().dataset_info(TINY_REPO, revision=args.revision).sha
    provenance.update({"repository": TINY_REPO, "requested_revision": args.revision,
                       "resolved_revision": revision, "selection": "seeded uniform document-index sample; source order"})
    cache = str(Path(args.cache_dir) / "huggingface")
    source_files, factories = {}, {}
    for split, filename, limit in (("train", "TinyStories-train.txt", args.train_limit),
                                   ("val", "TinyStories-valid.txt", args.val_limit)):
        path = Path(hf_hub_download(TINY_REPO, filename=filename, repo_type="dataset",
                                   revision=revision, cache_dir=cache))
        source_files[split] = {"file": filename, "sha256": sha256_file(path), "bytes": path.stat().st_size}

        def records(path=path, split=split, limit=limit, filename=filename):
            for index, text in selected_stories(path, limit, f"{args.seed}:{split}"):
                yield {"id": f"tinystories:{split}:{index}", "text": text,
                       "source": f"https://huggingface.co/datasets/{TINY_REPO}/resolve/{revision}/{filename}",
                       "metadata": {"dataset": TINY_REPO, "split": "validation" if split == "val" else split,
                                    "source_index": index, "revision": revision}}
        factories[split] = records
    card = hf_hub_download(TINY_REPO, filename="README.md", repo_type="dataset", revision=revision, cache_dir=cache)
    provenance["source_card"] = Path(card).read_text(encoding="utf-8")
    provenance["source_files"] = source_files
    return factories


def pg_listing(split, cache_dir, refresh=False, timeout=60, retries=4):
    path = Path(cache_dir) / "pg19" / f"listing-{split}.json"
    if path.exists() and not refresh:
        objects = json.loads(path.read_text(encoding="utf-8"))
    else:
        objects, token = [], None
        while True:
            query = {"prefix": f"{split}/", "maxResults": 1000,
                     "fields": "nextPageToken,items(name,generation,size,md5Hash)"}
            if token:
                query["pageToken"] = token
            url = f"https://storage.googleapis.com/storage/v1/b/{PG_BUCKET}/o?{urllib.parse.urlencode(query)}"
            page = fetch_json(url, timeout, retries)
            objects.extend(page.get("items", []))
            token = page.get("nextPageToken")
            if not token:
                break
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as f:
            json.dump(objects, f)
            temporary = Path(f.name)
        temporary.replace(path)
    valid = []
    for obj in objects:
        if not re.fullmatch(rf"{split}/[0-9]+\.txt", obj.get("name", "")):
            continue
        if not str(obj.get("generation", "")).isdigit() or not obj.get("md5Hash"):
            raise ValueError("PG-19 listing lacks generation/checksum; refusing an unpinned download")
        if int(obj["size"]) <= 0:
            raise ValueError(f"Empty source object: {obj['name']}")
        valid.append(obj)
    if not valid:
        raise ValueError(f"No PG-19 books found for {split}")
    return sorted(valid, key=lambda item: item["name"])


def read_pg_metadata(path):
    metadata = {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) != 4 or not row[0].isdigit():
                raise ValueError("Unexpected PG-19 metadata format")
            book_id, title, year, url = row
            metadata[book_id] = {"book_id": book_id, "title": title,
                                 "publication_date": int(year) if year.lstrip("-").isdigit() else year, "url": url}
    return metadata


def pg_sources(args, provenance):
    cache = Path(args.cache_dir) / "pg19"
    metadata_file = download_file(PG_METADATA, cache / PG_REVISION / "metadata.csv",
                                  timeout=args.timeout, retries=args.retries)
    metadata = read_pg_metadata(metadata_file)
    provenance.update({"bucket": f"gs://{PG_BUCKET}", "metadata_url": PG_METADATA,
                       "metadata_sha256": sha256_file(metadata_file),
                       "selection": "seeded uniform book sample; sorted source-name order",
                       "source_objects": {}})
    factories = {}
    splits = [("train", "train", args.train_limit), ("val", "validation", args.val_limit)]
    if args.include_test:
        splits.append(("test", "test", args.test_limit))
    for split, source_split, limit in splits:
        objects = pg_listing(source_split, args.cache_dir, args.refresh_source, args.timeout, args.retries)
        if limit and limit < len(objects):
            objects = sorted(random.Random(f"{args.seed}:{split}").sample(objects, limit), key=lambda item: item["name"])
        provenance["source_objects"][split] = objects

        def records(objects=objects, split=split, source_split=source_split):
            def fetch(obj):
                encoded = urllib.parse.quote(obj["name"], safe="")
                url = (f"https://storage.googleapis.com/download/storage/v1/b/{PG_BUCKET}/o/{encoded}"
                       f"?generation={obj['generation']}&alt=media")
                path = download_file(url, cache / obj["generation"] / obj["name"], size=int(obj["size"]),
                                     md5=obj["md5Hash"], timeout=args.timeout, retries=args.retries)
                return obj, url, path

            # Keep at most `workers` completed books/requests pending. ThreadPool.map
            # over the entire corpus would queue every book on Python <= 3.13.
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for start in range(0, len(objects), args.workers):
                    for obj, url, path in pool.map(fetch, objects[start:start + args.workers]):
                        book_id = Path(obj["name"]).stem
                        if book_id not in metadata:
                            raise ValueError(f"Missing PG-19 metadata for {book_id}")
                        yield {"id": f"pg19:{book_id}", "text": path.read_text(encoding="utf-8"), "source": url,
                               "metadata": {"dataset": "pg19", "split": source_split, **metadata[book_id],
                                            "generation": obj["generation"], "source_md5": obj["md5Hash"]}}
        factories[split] = records
    return factories


def spool_splits(factories, work_dir):
    """Deduplicate on disk, reserving held-out texts before reading train texts."""
    work_dir = Path(work_dir)
    statistics = {}
    with sqlite3.connect(work_dir / "dedup.sqlite") as db:
        db.execute("CREATE TABLE seen (digest BLOB PRIMARY KEY) WITHOUT ROWID")
        for split in ("test", "val", "train"):
            if split not in factories:
                continue
            stats = {"selected": 0, "written": 0, "duplicates_removed": 0, "empty_removed": 0, "characters": 0}
            with (work_dir / f"{split}.jsonl").open("w", encoding="utf-8") as f:
                for record in factories[split]():
                    stats["selected"] += 1
                    text = record["text"]
                    if not isinstance(text, str):
                        raise ValueError("Source text is not a string")
                    if not text.strip():
                        stats["empty_removed"] += 1
                        continue
                    digest = hashlib.sha256(text.encode("utf-8")).digest()
                    inserted = db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (digest,)).rowcount
                    if not inserted:
                        stats["duplicates_removed"] += 1
                        continue
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["written"] += 1
                    stats["characters"] += len(text)
                    if stats["written"] % 1000 == 0:
                        db.commit()
                        print(json.dumps({"event": "spooling", "split": split, **stats}), flush=True)
            db.commit()
            if stats["written"] == 0:
                raise ValueError(f"{split} is empty after filtering; increase the source limit")
            statistics[split] = stats
            print(json.dumps({"event": "split_ready", "split": split, **stats}), flush=True)
    return statistics


def prepare_corpus(args):
    preset = PRESETS[args.dataset]
    args.train_limit = preset["train_limit"] if args.train_limit is None else args.train_limit
    args.val_limit = preset["val_limit"] if args.val_limit is None else args.val_limit
    if min(args.train_limit, args.val_limit, args.test_limit, args.tokenizer_max_characters) < 0:
        raise ValueError("Document limits and tokenizer budget must be nonnegative (0 means unlimited)")
    if args.workers <= 0 or args.retries <= 0 or args.timeout <= 0:
        raise ValueError("workers, retries and timeout must be positive")
    if args.dataset == "tinystories" and args.include_test:
        raise ValueError("TinyStories has no official test split")
    if args.dataset != "tinystories" and args.revision != "main":
        raise ValueError("--revision applies only to TinyStories; PG-19 uses pinned object generations")
    if args.tokenizer_file and not Path(args.tokenizer_file).is_file():
        raise ValueError("--tokenizer-file does not exist")
    if not args.tokenizer_file and args.tokenizer == "bpe" and args.vocab_size < 259:
        raise ValueError("BPE vocab-size must be at least 259")
    # Fail before downloading if the requested optional dependencies are missing.
    if not args.tokenizer_file and args.tokenizer == "bpe":
        import tokenizers  # noqa: F401
    out = Path(args.out)
    if out.exists():
        raise ValueError(f"Refusing to overwrite existing dataset: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    provenance = {"dataset": args.dataset, "source": preset["source"], "declared_license": preset["license"],
                  "seed": args.seed, "requested_limits": {"train": args.train_limit, "val": args.val_limit,
                                                         "test": args.test_limit if args.include_test else None},
                  "deduplication": "exact SHA-256 of UTF-8 text; priority test > val > train; no split reassignment",
                  "test_included": args.include_test}
    print(json.dumps({"event": "preparing", "dataset": args.dataset, "limits": provenance["requested_limits"],
                      "cache": args.cache_dir, "license": preset["license"]}), flush=True)
    print("Check the source license/underlying text rights before use. Downloads are cached; no corpus is loaded at once.", flush=True)
    with tempfile.TemporaryDirectory(prefix=".corpus-", dir=out.parent) as work:
        factories = tiny_sources(args, provenance) if args.dataset == "tinystories" else pg_sources(args, provenance)
        provenance["filter_statistics"] = spool_splits(factories, work)
        source_file = Path(work) / "sources.json"
        source_file.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        return prepare(argparse.Namespace(input=[str(Path(work) / "train.jsonl")],
            validation_input=[str(Path(work) / "val.jsonl")],
            test_input=[str(Path(work) / "test.jsonl")] if args.include_test else None,
            out=args.out, tokenizer=args.tokenizer, vocab_size=args.vocab_size,
            tokenizer_file=args.tokenizer_file, tokenizer_max_characters=args.tokenizer_max_characters,
            val_fraction=0, seed=args.seed, document_separator=None, provenance=str(source_file)))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=sorted(PRESETS), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache-dir", default="griffin_memory/artifacts/downloads")
    p.add_argument("--train-limit", type=int, help="Default: TinyStories 100000, PG-19 1000; 0 = all")
    p.add_argument("--val-limit", type=int, help="Default: TinyStories 2000, PG-19 50; 0 = all")
    p.add_argument("--include-test", action="store_true", help="PG-19 only; never used for training or validation")
    p.add_argument("--test-limit", type=int, default=0, help="0 = all official test books, only with --include-test")
    p.add_argument("--revision", default="main", help="TinyStories revision, resolved to an immutable commit SHA")
    p.add_argument("--refresh-source", action="store_true", help="Refresh cached PG-19 object listings")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer", choices=["bpe", "byte"], default="bpe")
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--tokenizer-file", help="Reuse an existing tokenizer, e.g. for a later evaluation subset")
    p.add_argument("--tokenizer-max-characters", type=int, default=20_000_000,
                   help="BPE-training-only text budget, sampled across train documents; 0 = unlimited")
    p.add_argument("--workers", type=int, default=4, help="Concurrent PG-19 book downloads, not tokenizer workers")
    p.add_argument("--timeout", type=int, default=60, help="Per-request timeout for PG-19 HTTP operations")
    p.add_argument("--retries", type=int, default=4, help="Attempts per PG-19 HTTP operation")
    return p


def main():
    prepare_corpus(parser().parse_args())


if __name__ == "__main__":
    main()
