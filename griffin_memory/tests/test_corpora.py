import argparse
import base64
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import urllib.error

from griffin_memory.data import DocumentDataset, dataset_manifest
from griffin_memory.prepare import prepare, tokenizer_texts
from griffin_memory.prepare_corpus import (download_file, iter_stories, parser, pg_listing, pg_sources,
                                          prepare_corpus, selected_stories, spool_splits, tiny_sources)
from griffin_memory.tokenizer import Tokenizer


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def arguments(self, dataset="tinystories", *extra):
        return parser().parse_args(["--dataset", dataset, "--out", str(self.root / "prepared"),
                                   "--cache-dir", str(self.root / "cache"), "--tokenizer", "byte", *extra])

    def test_streaming_delimiters_unicode_and_tail(self):
        source = self.root / "stories.txt"
        source.write_text("\nŻółw 🌙.\n<|endoftext|>\n\n<|endoftext|>\nNext story.<|endoftext|>Tail", encoding="utf-8")
        for size in (1, 7, 16, 1024):
            self.assertEqual(list(iter_stories(source, size)), ["Żółw 🌙.", "Next story.", "Tail"])

    def test_subset_is_seeded_and_not_just_a_prefix(self):
        source = self.root / "stories.txt"
        source.write_text("<|endoftext|>".join(f"Story {i}" for i in range(100)))
        first = list(selected_stories(source, 10, "42:train"))
        self.assertEqual(first, list(selected_stories(source, 10, "42:train")))
        self.assertNotEqual(first, list(selected_stories(source, 10, "43:train")))
        self.assertEqual(len(first), 10)
        self.assertNotEqual([index for index, _ in first], list(range(10)))
        self.assertEqual(len(list(selected_stories(source, 200, 1))), 100)
        self.assertEqual(len(list(selected_stories(source, 0, 1))), 100)

    def test_heldout_priority_and_exact_deduplication(self):
        def records(split, texts):
            return lambda: iter({"id": f"{split}:{i}", "text": text} for i, text in enumerate(texts))
        factories = {"train": records("train", ["Train", "Train", "Val", "Test", " "]),
                     "val": records("val", ["Val", "Test"]), "test": records("test", ["Test"])}
        with redirect_stdout(io.StringIO()):
            stats = spool_splits(factories, self.root)
        self.assertEqual(stats["train"]["written"], 1)
        self.assertEqual(stats["train"]["duplicates_removed"], 3)
        self.assertEqual(stats["train"]["empty_removed"], 1)
        for split, text in (("train", "Train"), ("val", "Val"), ("test", "Test")):
            records = [json.loads(line) for line in (self.root / f"{split}.jsonl").read_text().splitlines()]
            self.assertEqual([r["text"] for r in records], [text])
            self.assertTrue(records[0]["id"].startswith(split))

    def test_explicit_splits_test_data_and_reused_tokenizer(self):
        texts = {"train": "The entire training book.\nChapter two is still in the same document.",
                 "val": "Validation only. 🌙", "test": "Final test only."}
        for split, text in texts.items():
            (self.root / f"{split}.jsonl").write_text(json.dumps({"id": split, "text": text,
                                                        "source": "https://example.test/book"}))
        tokenizer = self.root / "tokenizer.json"
        Tokenizer().save(tokenizer)
        args = argparse.Namespace(input=[str(self.root / "train.jsonl")],
                                  validation_input=[str(self.root / "val.jsonl")],
                                  test_input=[str(self.root / "test.jsonl")],
                                  out=str(self.root / "out"), tokenizer="bpe", tokenizer_file=str(tokenizer),
                                  vocab_size=8000, val_fraction=0.999, seed=42, document_separator=None)
        with patch.object(Tokenizer, "train_bpe", side_effect=AssertionError("Must reuse tokenizer")), redirect_stdout(io.StringIO()):
            manifest = prepare(args)
        self.assertEqual(manifest["split_policy"], "explicit")
        self.assertEqual(set(manifest["stats"]), {"train", "val", "test"})
        dataset_manifest(args.out)
        for split, text in texts.items():
            dataset = DocumentDataset(args.out, split)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(Tokenizer().decode(dataset[0].tolist()), text)
            self.assertEqual(dataset.metadata[0]["id"], split)
            self.assertEqual(dataset.metadata[0]["source"], "https://example.test/book")

    def test_bpe_budget_is_exact_and_samples_across_train_documents(self):
        raw = self.root / "train.raw"
        raw.write_text("\n".join(json.dumps({"text": character * 10000}) for character in "ABC"))
        first = list(tokenizer_texts(raw, max_characters=10001, seed=42))
        self.assertEqual(sum(map(len, first)), 10001)
        self.assertEqual(set("".join(first)), {"A", "B", "C"})
        self.assertTrue(all(len(span) <= 4096 for span in first))
        self.assertEqual(first, list(tokenizer_texts(raw, 10001, 42)))
        self.assertEqual(sum(map(len, tokenizer_texts(raw, 0))), 30000)
        self.assertEqual(sum(map(len, tokenizer_texts(raw, 50000))), 30000)

    def test_tinystories_pins_revision_uses_original_files_and_preserves_docs(self):
        train, val, card = self.root / "train.txt", self.root / "val.txt", self.root / "README.md"
        train.write_text("First<|endoftext|>Second<|endoftext|>")
        val.write_text("Validation<|endoftext|>")
        card.write_text("license: cdla-sharing-1.0")
        paths = {"TinyStories-train.txt": train, "TinyStories-valid.txt": val, "README.md": card}
        revisions = []

        def download(repo, filename, **kwargs):
            self.assertEqual(repo, "roneneldan/TinyStories")
            revisions.append(kwargs["revision"])
            return str(paths[filename])

        fake_hub = types.SimpleNamespace(HfApi=lambda: types.SimpleNamespace(
            dataset_info=lambda *a, **k: types.SimpleNamespace(sha="pinned-sha")), hf_hub_download=download)
        args = self.arguments("tinystories", "--train-limit", "0", "--val-limit", "0")
        provenance = {}
        with patch.dict("sys.modules", {"huggingface_hub": fake_hub}):
            factories = tiny_sources(args, provenance)
        self.assertEqual(provenance["resolved_revision"], "pinned-sha")
        self.assertEqual(revisions, ["pinned-sha"] * 3)
        self.assertEqual([r["text"] for r in factories["train"]()], ["First", "Second"])
        self.assertEqual([r["text"] for r in factories["val"]()], ["Validation"])
        self.assertEqual(provenance["source_files"]["train"]["bytes"], train.stat().st_size)

    def test_pg19_listing_paginates_and_reuses_cache(self):
        def obj(book):
            return {"name": f"train/{book}.txt", "generation": "123", "size": "42", "md5Hash": "checksum"}
        pages = [{"items": [obj(2)], "nextPageToken": "page2"}, {"items": [obj(1)]}]
        with patch("griffin_memory.prepare_corpus.fetch_json", side_effect=pages) as fetch:
            listing = pg_listing("train", self.root)
            self.assertEqual(fetch.call_count, 2)
            self.assertIn("pageToken=page2", fetch.call_args.args[0])
        self.assertEqual([o["name"] for o in listing], ["train/1.txt", "train/2.txt"])
        with patch("griffin_memory.prepare_corpus.fetch_json", side_effect=AssertionError("Must use cache")):
            self.assertEqual(pg_listing("train", self.root), listing)

    def test_pg19_keeps_whole_books_metadata_and_generations(self):
        args = self.arguments("pg19", "--train-limit", "1", "--val-limit", "1", "--include-test", "--test-limit", "1")
        metadata = self.root / "metadata.csv"
        metadata.write_text('1,"Book, one",1900,http://example.test/1\n2,Book two,1890,http://example.test/2\n3,Book three,1880,http://example.test/3\n')
        bodies = {i: f"Chapter one of book {i}.\n\nChapter two of book {i}." for i in (1, 2, 3)}
        for i, text in bodies.items():
            (self.root / f"{i}.txt").write_text(text)
        split_ids = {"train": 1, "validation": 2, "test": 3}

        def listing(split, *a):
            return [{"name": f"{split}/{split_ids[split]}.txt", "generation": "456", "size": "40", "md5Hash": "md5"}]

        def download(url, path, **kwargs):
            if "metadata.csv" in url:
                return metadata
            self.assertIn("generation=456", url)
            self.assertEqual(kwargs["md5"], "md5")
            return self.root / Path(path).name

        with patch("griffin_memory.prepare_corpus.download_file", side_effect=download), \
             patch("griffin_memory.prepare_corpus.pg_listing", side_effect=listing):
            provenance = {}
            factories = pg_sources(args, provenance)
            records = {split: list(factory()) for split, factory in factories.items()}
        self.assertEqual(records["train"][0]["text"], bodies[1])
        self.assertEqual(records["train"][0]["metadata"]["title"], "Book, one")
        self.assertEqual(records["test"][0]["metadata"]["split"], "test")
        self.assertEqual(provenance["source_objects"]["val"][0]["generation"], "456")

    def test_download_checks_md5_retries_and_reuses_verified_cache(self):
        content = b"complete book"
        md5 = base64.b64encode(hashlib.md5(content).digest()).decode()
        path = self.root / "book.txt"

        def response(body):
            result = io.BytesIO(body)
            result.headers = {"Content-Length": str(len(body))}
            return result

        with patch("griffin_memory.prepare_corpus.request", side_effect=[response(b"wrong"), response(content)]) as fetch, \
             patch("griffin_memory.prepare_corpus.time.sleep"):
            download_file("https://example.test/book", path, size=len(content), md5=md5)
            self.assertEqual(fetch.call_count, 2)
        self.assertEqual(path.read_bytes(), content)
        self.assertEqual(list(self.root.glob(".download-*")), [])
        with patch("griffin_memory.prepare_corpus.request", side_effect=AssertionError("Must use cache")):
            download_file("https://example.test/book", path, size=len(content), md5=md5)
        path.write_bytes(b"corruption")
        with patch("griffin_memory.prepare_corpus.request", return_value=response(content)) as fetch:
            download_file("https://example.test/book", path, size=len(content), md5=md5)
            self.assertEqual(fetch.call_count, 1)

    def test_download_does_not_retry_permanent_404(self):
        error = urllib.error.HTTPError("https://example.test/missing", 404, "Not found", {}, None)
        with patch("griffin_memory.prepare_corpus.request", side_effect=error) as fetch:
            with self.assertRaises(urllib.error.HTTPError):
                download_file("https://example.test/missing", self.root / "missing.txt")
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(list(self.root.glob(".download-*")), [])

    def test_end_to_end_preparation_and_no_validation_in_bpe(self):
        def sources(args, provenance):
            provenance["resolved_revision"] = "test-revision"
            return {split: lambda split=split: iter([{"id": split, "text": split * 20,
                     "source": f"https://example.test/{split}", "metadata": {"split": split}}])
                    for split in ("train", "val")}
        args = self.arguments("tinystories", "--tokenizer", "bpe", "--tokenizer-max-characters", "30")
        consumed = []

        def bpe(texts, vocab):
            consumed.extend(texts)
            return Tokenizer()

        with patch("griffin_memory.prepare_corpus.tiny_sources", side_effect=sources), \
             patch.dict("sys.modules", {"tokenizers": types.SimpleNamespace()}), \
             patch.object(Tokenizer, "train_bpe", side_effect=bpe), redirect_stdout(io.StringIO()):
            prepare_corpus(args)
        self.assertEqual(sum(map(len, consumed)), 30)
        self.assertNotIn("val", "".join(consumed))
        manifest, _ = dataset_manifest(args.out)
        self.assertEqual(manifest["stats"]["train"]["documents"], 1)
        self.assertEqual(manifest["stats"]["val"]["documents"], 1)
        self.assertIn("sources.json", manifest["sha256"])
        self.assertFalse(list(self.root.glob(".corpus-*")))
        self.assertFalse(list(self.root.glob(".prepare-*")))
        self.assertEqual(Tokenizer().decode(DocumentDataset(args.out, "train")[0].tolist()), "train" * 20)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            prepare_corpus(args)

    def test_tinystories_rejects_test_split_before_network(self):
        with self.assertRaisesRegex(ValueError, "no official test"):
            prepare_corpus(self.arguments("tinystories", "--include-test"))


if __name__ == "__main__":
    unittest.main()
