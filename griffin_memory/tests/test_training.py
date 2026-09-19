import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from griffin_memory.data import Cursor, DocumentDataset, DocumentStream, dataset_manifest
from griffin_memory.prepare import prepare
from griffin_memory.tokenizer import BOS, EOS, Tokenizer
from griffin_memory.train import evaluate, load_checkpoint, parser, train
from griffin_memory.config import ModelConfig
from griffin_memory.model import GriffinMemoryLM, StreamState

ROOT = Path(__file__).resolve().parents[1]


def assert_tree_equal(test, a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for key in a:
            assert_tree_equal(test, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_tree_equal(test, x, y)
    else:
        test.assertEqual(a, b)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        with redirect_stdout(io.StringIO()):
            prepare(argparse.Namespace(input=[str(ROOT / "examples/stories.jsonl")], out=str(self.data),
                                       val_fraction=0.2, seed=42, tokenizer="byte", vocab_size=259,
                                       document_separator=None))

    def tearDown(self):
        self.temp.cleanup()

    def test_byte_tokenizer_unicode_roundtrip(self):
        tokenizer = Tokenizer()
        text = "Zażółć gęślą jaźń. 🌙 日本語\n\t"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        self.assertEqual(tokenizer.decode([BOS] + tokenizer.encode(text) + [EOS]), text)
        tokenizer.save(self.root / "tokenizer.json")
        self.assertEqual(Tokenizer.load(self.root / "tokenizer.json").vocab_size, 259)

    @unittest.skipUnless(importlib.util.find_spec("tokenizers"), "optional BPE dependency not installed")
    def test_bpe_from_scratch_roundtrip(self):
        tokenizer = Tokenizer.train_bpe(["Mara carried a blue lantern."] * 10, 280)
        text = "A blue lantern. 🌙"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
        tokenizer.save(self.root / "tokenizer.json")
        restored = Tokenizer.load(self.root / "tokenizer.json")
        self.assertEqual(restored.encode(text), tokenizer.encode(text))

    def test_dataset_integrity_and_disjoint_document_split(self):
        manifest, _ = dataset_manifest(self.data)
        training = DocumentDataset(self.data, "train")
        validation = DocumentDataset(self.data, "val")
        self.assertEqual(manifest["stats"]["train"]["documents"], len(training))
        self.assertFalse({r["sha256"] for r in training.metadata} & {r["sha256"] for r in validation.metadata})
        with (self.data / "train.bin").open("ab") as f:
            f.write(b"x")
        with self.assertRaisesRegex(ValueError, "integrity"):
            dataset_manifest(self.data)

    def test_document_stream_covers_targets_once_and_cursor_resumes(self):
        dataset = DocumentDataset(self.data, "train")
        stream = DocumentStream(dataset, batch_size=2, chunk_size=32, seed=42, shuffle=False)
        count, resets, previous = 0, 0, None
        while (batch := stream.next()) is not None:
            count += int((batch["targets"] != -100).sum())
            resets += batch["reset"]
            if not batch["reset"]:
                self.assertEqual(previous, batch["document_ids"])
            previous = batch["document_ids"]
        self.assertEqual(count, sum(len(dataset[i]) - 1 for i in range(len(dataset))))
        self.assertEqual(resets, (len(dataset) + 1) // 2)
        stream = DocumentStream(dataset, 2, 32, 42)
        stream.next()
        restored = DocumentStream(dataset, 2, 32, 42, cursor=Cursor(**stream.cursor.state_dict()))
        assert_tree_equal(self, stream.next(), restored.next())

    def run_train(self, out, extra=None):
        arguments = ["--data", str(self.data), "--out", str(out), "--device", "cpu", "--threads", "1",
                     "--eval-batches", "2", "--eval-every", "0", "--save-every", "1", "--log-every", "1"]
        arguments += extra or []
        with redirect_stdout(io.StringIO()):
            return train(parser().parse_args(arguments))

    def test_training_resume_is_bitwise_identical_including_memories_and_optimizer(self):
        shared = ["--config", str(ROOT / "configs/smoke.json"), "--max-steps", "4",
                  "--warmup-steps", "1", "--accumulation-steps", "2", "--deterministic"]
        self.run_train(self.root / "full", shared)
        self.run_train(self.root / "resumed", shared + ["--stop-after-steps", "2"])
        middle = load_checkpoint(self.root / "resumed/last.pt")
        self.assertEqual(middle["step"], 2)
        self.assertIsNotNone(middle["stream_state"])
        self.assertGreater(len(middle["stream_state"]["banks"][0]), 0)
        self.run_train(self.root / "resumed", ["--resume", str(self.root / "resumed/last.pt")])
        expected = load_checkpoint(self.root / "full/last.pt")
        actual = load_checkpoint(self.root / "resumed/last.pt")
        for key in ("model", "optimizer", "scaler", "step", "total_tokens", "cursor", "stream_state", "rng_cpu"):
            assert_tree_equal(self, expected[key], actual[key])

    def test_resume_rejects_hyperparameter_changes(self):
        self.run_train(self.root / "run", ["--config", str(ROOT / "configs/smoke.json"), "--max-steps", "1"])
        with self.assertRaisesRegex(ValueError, "Resume mismatch"):
            self.run_train(self.root / "run", ["--resume", str(self.root / "run/last.pt"), "--batch-size", "9"])

    def test_validation_does_not_mutate_training_stream(self):
        config = ModelConfig.load(ROOT / "configs/smoke.json")
        model = GriffinMemoryLM(config).train()
        _, state = model(torch.randint(3, 259, (1, 32)))
        state.detach()
        before = state.state_dict()
        evaluate(model, DocumentDataset(self.data, "val"),
                 {"batch_size": 2, "seed": 42, "precision": "fp32"}, torch.device("cpu"), 3)
        self.assertTrue(model.training)
        assert_tree_equal(self, before, state.state_dict())

    def test_accumulation_flushes_partial_final_update(self):
        # One group, fewer chunks than accumulation_steps; epoch-end must still step.
        stats = self.run_train(self.root / "partial", ["--config", str(ROOT / "configs/smoke.json"),
                               "--max-steps", "5", "--batch-size", "9", "--accumulation-steps", "100"])
        self.assertEqual(stats["step"], 1)
        dataset = DocumentDataset(self.data, "train")
        self.assertEqual(stats["tokens"], sum(len(dataset[i]) - 1 for i in range(len(dataset))))
        checkpoint = load_checkpoint(self.root / "partial/last.pt")
        self.assertIsNone(checkpoint["stream_state"])
        self.assertEqual(checkpoint["cursor"]["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
