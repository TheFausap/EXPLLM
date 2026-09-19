import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import os
import subprocess
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F

from griffin_memory.config import ModelConfig
from griffin_memory.data import DocumentDataset
from griffin_memory.generate_samples import (generate_samples, parser as sample_parser, select_documents,
                                             score_continuation, stream_tokens, window)
from griffin_memory.model import GriffinMemoryLM
from griffin_memory.prepare import prepare
from griffin_memory.run_report import build, estimate, parser as report_parser, plan
from griffin_memory.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT.parent / "griffin_memory/scripts/pg19_full_run.sh"


def tiny_config(**overrides):
    options = dict(vocab_size=259, d_model=16, n_layers=2, n_heads=2, recurrent_dim=16, mlp_dim=32,
                   conv_kernel=3, window_size=8, chunk_size=16, memory_capacity=4, memory_top_k=3,
                   memory_key_dim=8, lsh_tables=3, lsh_bits=3)
    options.update(overrides)
    return ModelConfig(**options)


class PipelineTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        with redirect_stdout(io.StringIO()):
            prepare(argparse.Namespace(input=[str(ROOT / "examples/stories.jsonl")], out=str(self.data),
                                       val_fraction=0.25, seed=42, tokenizer="byte", vocab_size=259,
                                       document_separator=None))
        self.manifest = json.loads((self.data / "manifest.json").read_text())
        self.dataset = DocumentDataset(self.data, "val")
        self.config = tiny_config()
        self.checkpoint = self.write_checkpoint(self.config)

    def tearDown(self):
        self.temp.cleanup()

    def write_checkpoint(self, config, path=None, step=7):
        torch.manual_seed(0)
        model = GriffinMemoryLM(config)
        tokenizer_path = self.root / "tokenizer.json"
        Tokenizer().save(tokenizer_path)
        path = Path(path or (self.root / "model.pt"))
        torch.save({"format_version": 1, "config": config.to_dict(), "options": {"seed": 42},
                    "step": step, "model": model.state_dict(),
                    "tokenizer_json": tokenizer_path.read_text()}, path)
        return path

    def sample_arguments(self, *extra, out=None):
        return sample_parser().parse_args([
            "--checkpoint", str(self.checkpoint), "--data", str(self.data), "--split", "val",
            "--out", str(out or (self.root / "samples")), "--device", "cpu", "--precision", "fp32",
            "--documents", "2", "--prompt-tokens", "32", "--continuation-tokens", "32",
            "--max-new-tokens", "12", "--markdown-characters", "200", *extra])


class WindowTests(PipelineTestBase):
    def test_window_is_contiguous_and_continuation_follows_it(self):
        document = self.dataset[0]
        ids, context_length, text_end = window(document, prompt_tokens=32, prefill_tokens=0, start_fraction=0.0)
        self.assertEqual(ids[0], 1)  # BOS
        self.assertEqual(context_length, len(ids))
        self.assertEqual(ids[1:], [int(token) for token in document[1:33]])
        self.assertEqual(text_end, 32)
        self.assertEqual([int(token) for token in document[1 + text_end:1 + text_end + 4]], [int(token) for token in document[33:37]])

    def test_prefill_and_start_fraction_shift_the_streamed_window(self):
        document = self.dataset[0]
        ids, context_length, text_end = window(document, prompt_tokens=16, prefill_tokens=8, start_fraction=0.0)
        self.assertEqual(len(ids), context_length)
        self.assertEqual(context_length, 25)  # BOS + 8 prefill + 16 prompt
        self.assertEqual(text_end, 24)
        later, _, later_end = window(document, prompt_tokens=16, prefill_tokens=8, start_fraction=0.5)
        self.assertNotEqual(ids, later)
        self.assertGreater(later_end, text_end)
        self.assertEqual(later[1:], [int(token) for token in document[later_end - 24 + 1:later_end + 1]])

    def test_window_rejects_impossible_requests(self):
        document = self.dataset[0]
        with self.assertRaises(ValueError):
            window(document, prompt_tokens=len(document), prefill_tokens=0, start_fraction=0.0)
        with self.assertRaises(ValueError):
            window(document, prompt_tokens=0, prefill_tokens=0, start_fraction=0.0)


class SelectionTests(PipelineTestBase):
    def test_selection_orders_are_deterministic(self):
        first = select_documents(self.dataset, 2, "first", 42)
        self.assertEqual(first, [0, 1])
        self.assertEqual(select_documents(self.dataset, 2, "random", 42),
                         select_documents(self.dataset, 2, "random", 42))
        lengths = [len(self.dataset[index]) for index in range(len(self.dataset))]
        self.assertEqual(len(select_documents(self.dataset, 1, "longest", 1)), 1)
        self.assertEqual(lengths[select_documents(self.dataset, 1, "longest", 1)[0]], max(lengths))
        self.assertEqual(lengths[select_documents(self.dataset, 1, "shortest", 1)[0]], min(lengths))
        self.assertEqual(select_documents(self.dataset, 10, "first", 1, offset=1), list(range(1, len(lengths))))

    def test_explicit_document_ids_and_missing_ids(self):
        identifier = self.dataset.metadata[1]["id"]
        self.assertEqual(select_documents(self.dataset, 5, "first", 1, ids=identifier), [1])
        with self.assertRaises(ValueError):
            select_documents(self.dataset, 1, "first", 1, ids="not-a-document")


class StreamingTests(PipelineTestBase):
    def test_chunked_stream_never_crosses_a_memory_boundary(self):
        model = GriffinMemoryLM(self.config).eval()
        ids = [1] + list(range(3, 3 + 40))
        calls = []
        original = model.forward

        def spy(tokens, *arguments, **keywords):
            calls.append(int(tokens.shape[1]))
            return original(tokens, *arguments, **keywords)

        model.forward = spy
        with torch.inference_mode():
            _, state, _, _ = stream_tokens(model, ids, torch.device("cpu"))
        self.assertTrue(all(count <= self.config.chunk_size for count in calls))
        self.assertEqual(sum(calls), len(ids))
        self.assertEqual(state.position, len(ids))
        # Completed chunks become retrievable memory.
        self.assertEqual(len(state.banks[0].entries), len(ids) // self.config.chunk_size)

    def test_scoring_matches_token_by_token_evaluation(self):
        model = GriffinMemoryLM(self.config).eval()
        document = self.dataset[0]
        ids, context_length, text_end = window(document, prompt_tokens=24, prefill_tokens=8, start_fraction=0.0)
        reference = [int(token) for token in document[1 + text_end:1 + text_end + 16]]
        scored = score_continuation(model, ids + reference, context_length, torch.device("cpu"))
        self.assertEqual(scored["tokens"], len(reference))
        self.assertEqual(len(reference), 16)
        # Reference: feed one token at a time, keep every state, average the relevant CE.
        with torch.inference_mode():
            state, losses = None, []
            full = ids + reference
            for index in range(len(full) - 1):
                logits, state = model(torch.tensor([[full[index]]]), state, document_ids=["x"])
                if index + 1 >= context_length:
                    loss = F.cross_entropy(logits[0, 0].float().unsqueeze(0),
                                                             torch.tensor([full[index + 1]]))
                    losses.append(float(loss))
        self.assertAlmostEqual(scored["loss"], sum(losses) / len(losses), places=6)

    def test_scoring_rejects_an_empty_reference(self):
        model = GriffinMemoryLM(self.config).eval()
        with self.assertRaises(ValueError):
            score_continuation(model, [1, 3, 4, 5], 4, torch.device("cpu"))


class GenerateSamplesTests(PipelineTestBase):
    def test_report_written_with_reference_and_diagnostics(self):
        with redirect_stdout(io.StringIO()):
            report = generate_samples(self.sample_arguments())
        self.assertEqual(report["aggregate"]["documents"], 2)
        self.assertGreater(report["aggregate"]["continuation_loss"], 0)
        self.assertLessEqual(report["aggregate"]["continuation_tokens"], 64)  # two documents x 32 tokens
        self.assertGreaterEqual(report["aggregate"]["continuation_tokens"], 32)
        self.assertIsNotNone(report["aggregate"]["memory_off_loss"])
        markdown = (self.root / "samples" / "samples.md").read_text()
        for heading in ("# Generation report", "### Prompt (real held-out text)", "### Greedy continuation",
                        "### Sampled continuation", "### Real held-out continuation (reference)"):
            self.assertIn(heading, markdown)
        for item in report["samples"]:
            self.assertEqual(item["scores"]["memory_on"]["tokens"], item["reference_tokens"])
            self.assertLessEqual(item["reference_tokens"], 32)
            self.assertGreaterEqual(item["memory"]["memory_entries"][0], 0)
            self.assertIn("prompt_text", item)
        self.assertTrue((self.root / "samples" / "samples.json").is_file())
        self.assertEqual(len(list((self.root / "samples").glob("*.greedy.txt"))), 2)

    def test_greedy_decoding_is_deterministic_and_can_skip_sampling(self):
        with redirect_stdout(io.StringIO()):
            first = generate_samples(self.sample_arguments("--decoding", "greedy", "--no-ablation",
                                                           out=self.root / "a"))
            second = generate_samples(self.sample_arguments("--decoding", "greedy", "--no-ablation",
                                                            out=self.root / "b"))
        self.assertEqual(first["samples"][0]["continuations"]["greedy"]["text"],
                         second["samples"][0]["continuations"]["greedy"]["text"])
        self.assertEqual(first["samples"][0]["scores"]["memory_off"], None)
        self.assertIsNone(first["aggregate"]["memory_off_loss"])
        self.assertTrue(first["samples"][0]["continuations"]["sampled"]["same_as_greedy"])
        self.assertIn("identical to greedy", (self.root / "a" / "samples.md").read_text())

    def test_document_ids_and_split_validation(self):
        identifier = self.dataset.metadata[0]["id"]
        with redirect_stdout(io.StringIO()):
            report = generate_samples(self.sample_arguments("--document-ids", identifier))
        self.assertEqual([item["document_id"] for item in report["samples"]], [identifier])
        with self.assertRaises(ValueError):
            generate_samples(self.sample_arguments("--document-ids", "missing"))
        with self.assertRaises(ValueError):
            generate_samples(self.sample_arguments("--split", "test"))
        with self.assertRaises(ValueError):
            generate_samples(self.sample_arguments("--temperature", "-1"))
        with self.assertRaises(ValueError):
            generate_samples(self.sample_arguments("--checkpoint", str(self.root / "absent.pt")))

    def test_prefill_builds_retrievable_chunks(self):
        with redirect_stdout(io.StringIO()):
            plain = generate_samples(self.sample_arguments("--prefill-tokens", "0", "--prompt-tokens", "16",
                                                           out=self.root / "plain"))
            warmed = generate_samples(self.sample_arguments("--prefill-tokens", "48", "--prompt-tokens", "16",
                                                            out=self.root / "warmed"))
        self.assertEqual(plain["samples"][0]["memory"]["memory_entries"][0], 1)
        self.assertEqual(warmed["samples"][0]["memory"]["memory_entries"][0], 4)
        self.assertEqual(warmed["samples"][0]["context_tokens"], 65)  # BOS + 48 prefill + 16 prompt
        self.assertEqual(warmed["context_tokens"], 65)


class RunReportTests(PipelineTestBase):
    def plan_arguments(self, *extra):
        return report_parser().parse_args(["plan", "--data", str(self.data), "--config",
                                           str(ROOT / "configs/smoke.json"), "--batch-size", "2",
                                           "--accumulation-steps", "1", *extra])

    def test_plan_converts_budgets_into_steps(self):
        targets = self.manifest["stats"]["train"]["prediction_targets"]
        with redirect_stdout(io.StringIO()):
            result = plan(self.plan_arguments("--epochs", "2"))
            budget = plan(self.plan_arguments("--token-budget", "10000"))
            steps = plan(self.plan_arguments("--max-steps", "5"))
        self.assertEqual(result["tokens_per_update"], 64)
        self.assertEqual(result["full_pass_steps"], -(-targets // 64))
        self.assertEqual(result["planned_steps"], 2 * result["full_pass_steps"])
        self.assertEqual(budget["planned_steps"], -(-10000 // 64))
        self.assertEqual(steps["planned_steps"], 5)
        self.assertTrue(any("training split" in note for note in steps["notes"]))
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            plan(self.plan_arguments("--max-steps", "5", "--token-budget", "10"))

    def test_estimate_extrapolates_from_a_log(self):
        log = self.root / "train.jsonl"
        rows = [{"step": step, "train_loss": 6.0 - step / 100, "tokens": step * 64,
                 "tokens_per_second": 1000.0 + step, "memory_entries": 4} for step in range(1, 21)]
        rows.append({"step": 20, "val_loss": 5.5, "val_perplexity": 244.7})
        rows.append({"event": "finished", "step": 20, "tokens": 1280})
        log.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with redirect_stdout(io.StringIO()):
            result = estimate(report_parser().parse_args(["estimate", "--log", str(log),
                                                          "--planned-steps", "100"]))
        self.assertAlmostEqual(result["tokens_per_update"], 64, places=1)
        self.assertGreater(result["median_tokens_per_second"], 1000)
        self.assertAlmostEqual(result["planned_seconds"], 100 * result["seconds_per_update"], places=3)
        self.assertEqual(result["validation"], [[20, 5.5, 244.7]])
        empty = self.root / "empty.jsonl"
        empty.write_text("")
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            estimate(report_parser().parse_args(["estimate", "--log", str(empty)]))

    def build_run(self, samples=True):
        run = self.root / "run"
        (run / "train").mkdir(parents=True)
        (run / "driver").mkdir()
        (run / "eval").mkdir()
        (run / "train/run.json").write_text(json.dumps({"options": {"seed": 42, "batch_size": 1, "epochs": 3,
                                                                   "learning_rate": 0.0003, "precision": "bf16",
                                                                   "accumulation_steps": 16, "warmup_steps": 100,
                                                                   "max_steps": 10, "weight_decay": 0.1, "clip_grad": 1.0,
                                                                   "deterministic": False},
                                                        "device": "cuda", "torch": "2.5.1"}))
        (run / "train/config.json").write_text(json.dumps(self.config.to_dict()))
        (run / "train/train.jsonl").write_text(
            "\n".join(json.dumps(row) for row in
                      [{"step": 1, "train_loss": 6.1, "tokens": 16, "tokens_per_second": 1200.0,
                        "cursor": {"epoch": 0, "group": 0, "chunk": 1}, "memory_entries": 2},
                       {"step": 2, "train_loss": 5.9, "tokens": 32, "tokens_per_second": 1300.0,
                        "cursor": {"epoch": 0, "group": 2, "chunk": 3}, "memory_entries": 4},
                       {"step": 2, "val_loss": 5.8, "val_perplexity": 330.3, "val_tokens": 512},
                       {"event": "finished", "step": 2, "tokens": 32}]) + "\n")
        (run / "driver/commands.log").write_text("$ python -m griffin_memory.train --data corpus\n")
        (run / "driver/plan.json").write_text(json.dumps({"planned_steps": 5, "planned_targets_upper_bound": 320,
                                                          "epochs_equivalent": 0.5, "train_targets": 1234,
                                                          "full_pass_steps": 10, "notes": ["a note"]}))
        (run / "eval/val.json").write_text(json.dumps({"split": "val", "checkpoint_step": 2, "val_loss": 5.7,
                                                       "val_perplexity": 298.9, "val_tokens": 256}))
        if samples:
            (run / "samples").mkdir()
            (run / "samples/samples.json").write_text(json.dumps(
                {"split": "val", "checkpoint_step": 2, "data": str(self.data),
                 "aggregate": {"documents": 1, "continuation_tokens": 32, "continuation_loss": 5.5,
                               "continuation_perplexity": 244.7, "memory_off_loss": 5.7}}))
            (run / "samples/samples.md").write_text("# Generation report — checkpoint step 2\n\n| a |\n|---|\n")
        return run

    def test_build_collects_every_available_artifact(self):
        run = self.build_run()
        with redirect_stdout(io.StringIO()):
            report = build(report_parser().parse_args(["build", "--run-dir", str(run), "--data", str(self.data),
                                                       "--samples", str(run / "samples/samples.json"),
                                                       "--eval", str(run / "eval/val.json"),
                                                       "--out", str(run / "report.md")]))
        for section in ("# Run report", "## Corpus", "## Training", "## Validation", "## Commands",
                        "## Generation samples", "## How to read this"):
            self.assertIn(section, report)
        self.assertIn("median throughput 1,250 targets/s", report)
        self.assertIn("plan: 5 steps", report)
        self.assertIn("note: a note", report)
        self.assertIn("| 2 | 5.8000 | 330.30 | 512 |", report)
        self.assertIn("retrieval tier disabled", report)
        self.assertIn("val.json", report)
        self.assertTrue((run / "report.md").is_file())

    def test_build_requires_a_run_json_and_log(self):
        empty = self.root / "nothing"
        empty.mkdir()
        with self.assertRaises(ValueError):
            build(report_parser().parse_args(["build", "--run-dir", str(empty)]))


@unittest.skipUnless(shutil.which("bash"), "bash is not available")
class DriverScriptTests(PipelineTestBase):
    def run_script(self, *arguments, expect=0):
        environment = {**os.environ, "PYTHON": sys.executable}
        result = subprocess.run(["bash", str(SCRIPT), *arguments], capture_output=True, text=True,
                                timeout=600, env=environment)
        self.assertEqual(result.returncode, expect, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def test_help_and_argument_errors(self):
        self.assertIn("Usage: pg19_full_run.sh", self.run_script("--help"))
        self.run_script(expect=2)  # missing run directory
        self.run_script("/tmp/unused-run-directory", "--nonsense", expect=2)
        self.run_script("/tmp/unused-run-directory", "--dataset", "imagenet", expect=2)

    def test_dry_run_prints_the_pipeline_without_touching_the_corpus(self):
        run = self.root / "dryrun"
        output = self.run_script(str(run), "--data", str(self.data), "--profile", "smoke", "--token-budget", "4096",
                                 "--samples", "2", "--prefill-tokens", "64", "--dry-run")
        self.assertIn("griffin_memory.run_report plan", output)
        self.assertIn("griffin_memory.train", output)
        self.assertIn("griffin_memory.generate_samples", output)
        self.assertIn("--prefill-tokens 64", output)
        self.assertIn("griffin_memory.evaluate", output)
        self.assertFalse(run.exists())
        # Stage selection is honoured in dry-run mode as well.
        only = self.run_script(str(run), "--data", str(self.data), "--profile", "smoke", "--only", "train", "--dry-run")
        self.assertIn("griffin_memory.train", only)
        self.assertNotIn("generate_samples", only)

    def test_smoke_profile_offline_pipeline_end_to_end(self):
        corpus = self.root / "corpus"
        with redirect_stdout(io.StringIO()):
            prepare(argparse.Namespace(input=[str(ROOT / "examples/stories.jsonl")], out=str(corpus),
                                       val_fraction=0.25, seed=42, tokenizer="byte", vocab_size=259,
                                       document_separator=None))
        run = self.root / "smoke-run"
        output = self.run_script(str(run), "--data", str(corpus), "--profile", "smoke", "--max-steps", "4",
                                 "--epochs", "1", "--eval-every", "2", "--save-every", "2", "--eval-batches", "1",
                                 "--samples", "1", "--prompt-tokens", "24", "--max-new-tokens", "4",
                                 "--continuation-tokens", "8", "--threads", "1")
        self.assertIn("Done. Artifacts", output)
        for artifact in ("train/last.pt", "train/best.pt", "train/train.jsonl", "driver/plan.json",
                         "driver/commands.log", "samples/samples.md", "samples/samples.json", "eval/val.json",
                         "run_report.md"):
            self.assertTrue((run / artifact).is_file(), artifact)
        report = (run / "run_report.md").read_text()
        self.assertIn("## Generation samples", report)
        commands = (run / "driver/commands.log").read_text()
        self.assertIn("griffin_memory.train", commands)
        self.assertIn("--max-steps 4", commands)

    def test_missing_corpus_fails_before_training(self):
        run = self.root / "no-corpus"
        output = self.run_script(str(run), "--data", str(self.root / "absent"), "--profile", "smoke", expect=1)
        self.assertIn("has no manifest.json", output)
        self.assertFalse((run / "train").exists())


if __name__ == "__main__":
    unittest.main()
