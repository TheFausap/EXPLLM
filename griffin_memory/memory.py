"""Bounded document-local chunk store with multi-table random-hyperplane ANN.

Index selection is non-differentiable; selected keys/values are recomputed by
trainable projections. Original token chunks and metadata remain available for
inspection/export rather than being falsely advertised as lossless neural recall.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


class LSHIndex:
    def __init__(self, dim, tables=4, bits=8, seed=1729):
        generator = torch.Generator().manual_seed(seed)
        self.planes = torch.randn(tables, dim, bits, generator=generator)
        self.bits = bits
        self.buckets = []
        self.keys = torch.empty(0, dim)

    def codes(self, vectors):
        signs = torch.einsum("nd,tdh->nth", vectors.float().cpu(), self.planes) >= 0
        return (signs.long() * (2 ** torch.arange(self.bits))).sum(-1)

    def rebuild(self, keys):
        self.keys = F.normalize(keys.detach().float().cpu(), dim=-1)
        codes = self.codes(self.keys).tolist()
        self.buckets = [{} for _ in range(self.planes.shape[0])]
        for index, row in enumerate(codes):
            for table, code in enumerate(row):
                self.buckets[table].setdefault(code, []).append(index)

    def search(self, queries, k, exact=False):
        queries = F.normalize(queries.detach().float().cpu(), dim=-1)
        codes = self.codes(queries).tolist()
        k = min(k, len(self.keys))
        if not k:
            return torch.empty(len(queries), 0, dtype=torch.long)
        results = []
        for q, row in zip(queries, codes):
            candidates = set()
            if not exact:
                for table, code in enumerate(row):
                    # Multi-probe: same bucket and Hamming-distance-one neighbors.
                    for probe in [code] + [code ^ (1 << bit) for bit in range(self.bits)]:
                        candidates.update(self.buckets[table].get(probe, []))
            if exact or len(candidates) < k:
                candidates = set(range(len(self.keys)))
            ids = torch.tensor(sorted(candidates), dtype=torch.long)
            scores = self.keys[ids] @ q
            # Stable tie ordering aids deterministic checkpoint continuation.
            order = torch.argsort(scores, descending=True, stable=True)[:k]
            results.append(ids[order])
        return torch.stack(results)


@dataclass
class MemoryEntry:
    summary: torch.Tensor
    token_ids: list
    metadata: dict
    key: torch.Tensor
    value: torch.Tensor


class MemoryBank:
    def __init__(self, config):
        self.config = config
        self.entries = []
        self.index = LSHIndex(config.memory_key_dim, config.lsh_tables,
                              config.lsh_bits, config.lsh_seed)

    def write(self, summary, token_ids, metadata, key, value):
        self.entries.append(MemoryEntry(summary.detach().float().cpu(), list(token_ids),
                                        dict(metadata), key.detach().float().cpu(),
                                        value.detach().float().cpu()))
        if len(self.entries) > self.config.memory_capacity:
            del self.entries[0]

    def state_dict(self):
        return [{"summary": e.summary, "token_ids": e.token_ids, "metadata": e.metadata,
                 "key": e.key, "value": e.value} for e in self.entries]

    @classmethod
    def from_state_dict(cls, config, state):
        bank = cls(config)
        bank.entries = [MemoryEntry(**entry) for entry in state]
        return bank


class MemoryAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query = nn.Linear(config.d_model, config.memory_key_dim, bias=False)
        self.key = nn.Linear(config.d_model, config.memory_key_dim, bias=False)
        self.value = nn.Linear(config.d_model, config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        self.gate = nn.Linear(config.d_model, 1)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, x, banks):
        results = []
        queries = F.normalize(self.query(x).float(), dim=-1)
        for b, bank in enumerate(banks):
            if not bank.entries:
                results.append(torch.zeros_like(x[b]))
                continue
            summaries = torch.stack([entry.summary for entry in bank.entries]).to(x.device)
            keys = F.normalize(self.key(summaries).float(), dim=-1)
            values = self.value(summaries)
            # Refresh index embeddings with current weights (no stale-key drift
            # after optimizer updates). Selection itself does not enter autograd.
            bank.index.rebuild(keys)
            for entry, key, value in zip(bank.entries, keys.detach().cpu(), values.detach().float().cpu()):
                entry.key, entry.value = key, value
            ids = bank.index.search(queries[b], self.config.memory_top_k,
                                    exact=self.config.memory_backend == "exact").to(x.device)
            selected_keys, selected_values = keys[ids], values[ids]
            scores = (queries[b, :, None, :] * selected_keys).sum(-1)
            scores = scores * self.log_temperature.exp().clamp(max=100)
            weights = scores.softmax(-1).to(selected_values.dtype)
            retrieved = (weights[..., None] * selected_values).sum(-2)
            results.append(self.output(retrieved) * torch.sigmoid(self.gate(x[b])))
        return torch.stack(results)

    @torch.no_grad()
    def write(self, bank, summary, token_ids, metadata):
        key = F.normalize(self.key(summary).float(), dim=-1)
        value = self.value(summary)
        bank.write(summary, token_ids, metadata, key, value)
