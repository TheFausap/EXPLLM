import tempfile
import unittest
from pathlib import Path

import torch

from qalf.data import DialogueExample, build_tokenizer, encode_examples, make_windows, relation_counts
from qalf.model import QALFConfig, QALFModel, component_diversity_loss, load_checkpoint, save_checkpoint
from qalf.state import is_hermitian, trace_real


class QALFInvariantTests(unittest.TestCase):
    def setUp(self):
        self.examples = [
            DialogueExample("What is QALF?", "QALF is a small complex language field."),
            DialogueExample("How does it reply?", "It measures candidates with a Born style score."),
        ]
        self.tokenizer = build_tokenizer(self.examples, vocab_size=128)
        self.encoded = encode_examples(self.tokenizer, self.examples)
        bigram = relation_counts(self.encoded, len(self.tokenizer.vocab))
        config = QALFConfig(
            vocab_size=len(self.tokenizer.vocab),
            dimension=16,
            context_size=8,
            num_relations=2,
            pad_id=self.tokenizer.pad_id,
        )
        self.model = QALFModel(config, bigram_logits=bigram)

    def test_token_states_are_normalized(self):
        states = self.model.lexicon()
        norms = torch.linalg.vector_norm(states, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_density_is_hermitian_and_trace_normalized(self):
        contexts, _, _ = make_windows(self.encoded, self.model.config.context_size, self.tokenizer.pad_id)
        rho = self.model.density(contexts[:4])
        self.assertTrue(is_hermitian(rho))
        self.assertTrue(torch.allclose(trace_real(rho), torch.ones(rho.shape[0]), atol=1e-5))

    def test_decoder_distribution_is_valid(self):
        contexts, prev, _ = make_windows(self.encoded, self.model.config.context_size, self.tokenizer.pad_id)
        logits = self.model(contexts[:3], prev[:3])
        probs = torch.softmax(logits, dim=-1)
        self.assertTrue(torch.isfinite(probs).all())
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(3), atol=1e-5))

    def test_component_diagnostics_and_diversity_loss_are_valid(self):
        contexts, _, _ = make_windows(self.encoded, self.model.config.context_size, self.tokenizer.pad_id)
        diagnostics = self.model.diagnostics(contexts[:4])
        self.assertIn("component_overlap_mean", diagnostics)
        self.assertIn("component_overlap_max", diagnostics)
        self.assertIn("density_effective_rank", diagnostics)
        self.assertGreaterEqual(diagnostics["component_overlap_mean"], 0.0)
        self.assertLessEqual(diagnostics["component_overlap_max"], 1.0 + 1e-5)
        loss = component_diversity_loss(self.model, contexts[:4], target_overlap=0.05)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss.detach()), 0.0)


    def test_component_mixture_respects_minimum_weight(self):
        self.model.config.component_min_weight = 0.08
        mix = self.model.component_mixture()
        self.assertTrue(torch.all(mix >= 0.08 - 1e-6))
        self.assertTrue(torch.allclose(mix.sum(), torch.tensor(1.0), atol=1e-6))
        diagnostics = self.model.diagnostics(make_windows(self.encoded, self.model.config.context_size, self.tokenizer.pad_id)[0][:4])
        self.assertGreaterEqual(diagnostics["component_weight_min"], 0.08 - 1e-6)
        self.assertLessEqual(diagnostics["component_weight_max"], 1.0)

    def test_save_load_preserves_generation_with_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            save_checkpoint(path, self.model, self.tokenizer, {"test": True})
            loaded, tokenizer, metadata = load_checkpoint(path)
            self.assertTrue(metadata["test"])
            a = self.model.generate(self.tokenizer, "What is QALF?", seed=3)
            b = loaded.generate(tokenizer, "What is QALF?", seed=3)
            self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
