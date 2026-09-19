import unittest

import torch
from torch.nn import functional as F

from griffin_memory.config import ModelConfig
from griffin_memory.generate import prefill, sample
from griffin_memory.memory import LSHIndex
from griffin_memory.model import GriffinMemoryLM, LocalMQAttention, RecurrentBlock, StreamState


def tiny_config(**overrides):
    options = dict(d_model=16, n_layers=2, n_heads=2, recurrent_dim=16, mlp_dim=32,
                   conv_kernel=3, window_size=4, chunk_size=8, memory_capacity=4,
                   memory_top_k=3, memory_key_dim=8, lsh_tables=3, lsh_bits=3)
    options.update(overrides)
    return ModelConfig(**options)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(123)
        self.config = tiny_config()
        self.model = GriffinMemoryLM(self.config).eval()

    def test_config_rejects_invalid_shapes(self):
        for options in ({"d_model": 15}, {"chunk_size": 0}, {"memory_backend": "bogus"},
                        {"conv_kernel": 0}, {"lsh_bits": 25}):
            with self.assertRaises(ValueError):
                tiny_config(**options)

    @torch.no_grad()
    def test_future_tokens_cannot_change_prefix_logits(self):
        prefix = torch.randint(3, 259, (1, 16))
        _, history = prefill(self.model, prefix)
        saved = history.state_dict()
        first = torch.randint(3, 259, (1, 8))
        second = first.clone()
        second[:, 4:] = torch.randint(3, 259, (1, 4))
        logits_a, _ = self.model(first, StreamState.from_state_dict(self.config, saved, "cpu"))
        logits_b, _ = self.model(second, StreamState.from_state_dict(self.config, saved, "cpu"))
        torch.testing.assert_close(logits_a[:, :4], logits_b[:, :4], atol=1e-6, rtol=1e-6)

    @torch.no_grad()
    def test_chunk_prefill_matches_token_by_token_with_all_memories(self):
        tokens = torch.randint(3, 259, (2, 32))
        chunk_logits, chunk_state = [], None
        for start in range(0, tokens.shape[1], 8):
            logits, chunk_state = self.model(tokens[:, start:start + 8], chunk_state)
            chunk_logits.append(logits)
        step_logits, step_state = [], None
        for token in tokens.split(1, dim=1):
            logits, step_state = self.model(token, step_state)
            step_logits.append(logits)
        torch.testing.assert_close(torch.cat(chunk_logits, 1), torch.cat(step_logits, 1), atol=2e-6, rtol=1e-5)
        self.assertEqual(chunk_state.position, step_state.position)
        for chunk_bank, step_bank in zip(chunk_state.banks, step_state.banks):
            for a, b in zip(chunk_bank.entries, step_bank.entries):
                self.assertEqual(a.token_ids, b.token_ids)
                torch.testing.assert_close(a.summary, b.summary, atol=2e-6, rtol=1e-5)
        for recurrent, attention in step_state.layers:
            self.assertEqual(recurrent[0].dtype, torch.float32)
            self.assertEqual(recurrent[1].shape[1], self.config.conv_kernel - 1)
            self.assertLessEqual(attention[0].shape[2], self.config.window_size - 1)
            self.assertEqual(attention[0].shape[1], 1)

    @torch.no_grad()
    def test_recurrent_convolution_stream_equivalence(self):
        for kernel in (1, 4):
            block = RecurrentBlock(tiny_config(conv_kernel=kernel))
            x = torch.randn(2, 12, 16)
            expected, _ = block(x, None)
            results, state = [], None
            for fragment in x.split(3, dim=1):
                value, state = block(fragment, state)
                results.append(value)
            torch.testing.assert_close(expected, torch.cat(results, 1), atol=1e-6, rtol=1e-5)

    @torch.no_grad()
    def test_attention_window_and_absolute_rope(self):
        for window in (1, 4):
            attention = LocalMQAttention(tiny_config(window_size=window))
            x = torch.randn(1, 12, 16)
            first, _ = attention(x, None, 1000)
            modified = x.clone()
            modified[:, :5] += 10
            second, _ = attention(modified, None, 1000)
            torch.testing.assert_close(first[:, 8:], second[:, 8:], atol=1e-6, rtol=1e-5)
            results, state = [], None
            for i, token in enumerate(x.split(1, dim=1)):
                result, state = attention(token, state, 1000 + i)
                results.append(result)
            torch.testing.assert_close(first, torch.cat(results, 1), atol=1e-6, rtol=1e-5)

    @torch.no_grad()
    def test_only_completed_chunks_are_written_and_capacity_is_bounded(self):
        tokens = torch.randint(3, 259, (1, 48))
        _, state = self.model(tokens[:, :7])
        self.assertEqual(len(state.banks[0].entries), 0)
        self.assertEqual(state.pending_tokens[0], tokens[0, :7].tolist())
        _, state = self.model(tokens[:, 7:8], state, document_ids=["story"])
        self.assertEqual(state.banks[0].entries[0].token_ids, tokens[0, :8].tolist())
        self.assertEqual(state.banks[0].entries[0].metadata["document_id"], "story")
        for start in range(8, 48, 8):
            _, state = self.model(tokens[:, start:start + 8], state)
        self.assertEqual(len(state.banks[0].entries), 4)
        self.assertEqual(state.banks[0].entries[0].metadata["start"], 16)
        with self.assertRaises(ValueError):
            self.model(tokens[:, :9])

    @torch.no_grad()
    def test_padded_lanes_do_not_write_pad_tokens(self):
        tokens = torch.randint(3, 259, (2, 8))
        mask = torch.ones_like(tokens, dtype=torch.bool)
        mask[0, 3:] = False
        mask[1] = False
        _, state = self.model(tokens, token_mask=mask)
        self.assertEqual(state.banks[0].entries[0].token_ids, tokens[0, :3].tolist())
        self.assertEqual(state.banks[0].entries[0].metadata["end"], 3)
        self.assertEqual(len(state.banks[1].entries), 0)

    def test_all_trainable_components_receive_finite_gradients(self):
        self.model.train()
        tokens = torch.randint(3, 259, (1, 32))
        state = None
        for start in range(0, 24, 8):
            _, state = self.model(tokens[:, start:start + 8], state)
            state.detach()
        logits, state = self.model(tokens[:, 24:], state)
        loss = F.cross_entropy(logits.reshape(-1, 259), torch.randint(3, 259, (8,)))
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertFalse(state.banks[0].entries[0].summary.requires_grad)
        state.detach()
        self.assertIsNone(state.layers[0][0][0].grad_fn)

    @torch.no_grad()
    def test_retrieval_changes_logits_and_reset_clears_it(self):
        tokens = torch.randint(3, 259, (1, 24))
        _, state = prefill(self.model, tokens[:, :16])
        saved = state.state_dict()
        with_memory, _ = self.model(tokens[:, 16:], state)
        no_memory = StreamState.from_state_dict(self.config, saved, "cpu")
        for bank in no_memory.banks:
            bank.entries.clear()
        without_memory, _ = self.model(tokens[:, 16:], no_memory)
        self.assertGreater((with_memory - without_memory).abs().max().item(), 1e-7)
        fresh = self.model.initial_state(1, "cpu")
        self.assertEqual(fresh.position, 0)
        self.assertEqual(len(fresh.banks[0].entries), 0)

    @torch.no_grad()
    def test_memory_checkpoint_roundtrip_mid_chunk(self):
        tokens = torch.randint(3, 259, (1, 23))
        _, state = prefill(self.model, tokens[:, :19])
        saved = state.state_dict()
        expected, _ = self.model(tokens[:, 19:], state)
        restored = StreamState.from_state_dict(self.config, saved, "cpu")
        actual, _ = self.model(tokens[:, 19:], restored)
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)

    def test_lsh_and_exact_search_and_empty_store(self):
        index = LSHIndex(8, bits=4)
        keys = F.normalize(torch.randn(100, 8), dim=-1)
        self.assertEqual(index.search(keys[:1], 3).shape, (1, 0))
        index.rebuild(keys)
        found = index.search(keys[:10], 1)
        torch.testing.assert_close(found[:, 0], torch.arange(10))
        query = F.normalize(torch.randn(4, 8), dim=-1)
        expected = torch.argsort(query @ keys.T, descending=True, stable=True)[:, :3]
        torch.testing.assert_close(index.search(query, 3, exact=True), expected)

    def test_cpu_bf16_backward_keeps_recurrence_float32(self):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits, state = self.model(torch.randint(3, 259, (1, 8)))
            loss = logits.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(state.layers[0][0][0].dtype, torch.float32)

    def test_sampling_masks_special_tokens(self):
        logits = torch.zeros(1, 259)
        logits[0, 0] = 100
        logits[0, 1] = 99
        logits[0, 50] = 98
        self.assertEqual(sample(logits, 0, 0, 1).item(), 50)
        self.assertEqual(sample(logits, 1, 1, 0.9).item(), 50)


if __name__ == "__main__":
    unittest.main()
