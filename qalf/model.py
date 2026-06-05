"""QALF model: complex associative field with Born-style decoding."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from qalf.state import density_from_state, normalize_complex, purity, trace_real
from qalf.tokenizer import QuantumTokenizer


@dataclass
class QALFConfig:
    vocab_size: int
    dimension: int = 64
    context_size: int = 24
    num_relations: int = 4
    num_components: int = 4
    bigram_strength: float = 0.35
    trigram_strength: float = 0.75
    pad_id: int = 0


class QuantumLexicon(nn.Module):
    """Trainable complex token states in a finite Hilbert space."""

    def __init__(self, vocab_size: int, dimension: int):
        super().__init__()
        scale = 1.0 / math.sqrt(dimension)
        self.real = nn.Parameter(torch.randn(vocab_size, dimension) * scale)
        self.imag = nn.Parameter(torch.randn(vocab_size, dimension) * scale)

    def forward(self) -> torch.Tensor:
        states = torch.complex(self.real, self.imag)
        return normalize_complex(states)


class EntangledMemory(nn.Module):
    """A small bank of complex relation operators."""

    def __init__(self, dimension: int, num_relations: int):
        super().__init__()
        scale = 1.0 / math.sqrt(dimension)
        eye = torch.eye(dimension)
        self.real = nn.Parameter(torch.randn(num_relations, dimension, dimension) * scale)
        self.imag = nn.Parameter(torch.randn(num_relations, dimension, dimension) * scale)
        with torch.no_grad():
            self.real[0].copy_(eye)
            self.imag[0].zero_()
        self.mixing_logits = nn.Parameter(torch.zeros(num_relations))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        operators = torch.complex(self.real, self.imag)
        mixed = torch.softmax(self.mixing_logits, dim=0).to(operators.dtype)
        transformed = torch.einsum("bd,kde->bke", state, operators)
        return torch.einsum("k,bkd->bd", mixed, transformed)


class QALFModel(nn.Module):
    """Generative language model without attention, recurrence, or pretrained weights."""

    def __init__(
        self,
        config: QALFConfig,
        bigram_logits: torch.Tensor | None = None,
        trigram_prior: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.config = config
        self.lexicon = QuantumLexicon(config.vocab_size, config.dimension)
        self.memory = EntangledMemory(config.dimension, config.num_relations)
        self.phase_frequencies = nn.Parameter(torch.linspace(0.05, 1.25, config.dimension))
        self.component_logits = nn.Parameter(torch.zeros(config.num_components))
        self.component_phase_offsets = nn.Parameter(torch.linspace(0.0, 1.0, config.num_components))
        self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))
        if bigram_logits is None:
            bigram_logits = torch.zeros(config.vocab_size, config.vocab_size, dtype=torch.float32)
        self.register_buffer("bigram_logits", bigram_logits.float())
        if trigram_prior is None:
            trigram_prior = {
                "keys": torch.empty(0, dtype=torch.long),
                "token_ids": torch.empty(0, 0, dtype=torch.long),
                "token_logits": torch.empty(0, 0, dtype=torch.float32),
            }
        self.register_buffer("trigram_keys", trigram_prior["keys"].long())
        self.register_buffer("trigram_token_ids", trigram_prior["token_ids"].long())
        self.register_buffer("trigram_token_logits", trigram_prior["token_logits"].float())

    def component_weights(self, context_ids: torch.Tensor) -> torch.Tensor:
        batch, length = context_ids.shape
        device = context_ids.device
        positions = torch.linspace(0.0, 1.0, length, device=device)
        components: list[torch.Tensor] = []
        components.append(torch.linspace(0.35, 1.0, length, device=device))
        components.append(torch.exp(-5.0 * (1.0 - positions)))
        components.append(torch.exp(-2.2 * positions))
        components.append(0.55 + 0.45 * torch.sin(torch.pi * positions).clamp_min(0.0))
        for idx in range(4, self.config.num_components):
            center = (idx - 3) / max(self.config.num_components - 3, 1)
            components.append(torch.exp(-16.0 * (positions - center).square()))
        weights = torch.stack(components[: self.config.num_components], dim=0)
        mask = context_ids.ne(self.config.pad_id).float()
        weights = weights[None, :, :] * mask[:, None, :]
        return weights

    def context_components(self, context_ids: torch.Tensor) -> torch.Tensor:
        states = self.lexicon()
        embedded = states[context_ids]
        weights = self.component_weights(context_ids)
        positions = torch.arange(context_ids.shape[1], device=context_ids.device, dtype=torch.float32)
        phase_base = positions[:, None] * self.phase_frequencies[None, :]
        phases = []
        for component in range(self.config.num_components):
            offset = self.component_phase_offsets[component]
            phases.append(torch.exp(1j * (phase_base + offset * positions[:, None])))
        phase = torch.stack(phases, dim=0)
        weighted = embedded[:, None, :, :] * phase[None, :, :, :] * weights[:, :, :, None].to(embedded.dtype)
        components = weighted.sum(dim=2)
        empty = weights.sum(dim=-1).eq(0)
        if empty.any():
            pad_state = states[self.config.pad_id]
            components[empty] = pad_state
        return normalize_complex(components)

    def context_state(self, context_ids: torch.Tensor) -> torch.Tensor:
        mix = torch.softmax(self.component_logits, dim=0).to(self.context_components(context_ids).dtype)
        return normalize_complex(torch.einsum("c,bcd->bd", mix, self.context_components(context_ids)))

    def density(self, context_ids: torch.Tensor) -> torch.Tensor:
        components = self.context_components(context_ids)
        mix = torch.softmax(self.component_logits, dim=0).to(components.dtype)
        projectors = components.unsqueeze(-1) * components.conj().unsqueeze(-2)
        rho = torch.einsum("c,bcde->bde", mix, projectors)
        trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1).real.clamp_min(1e-8)
        return rho / trace[:, None, None].to(rho.dtype)

    def trigram_logits_for(self, prev2_ids: torch.Tensor, prev_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(prev_ids.shape[0], self.config.vocab_size, device=prev_ids.device)
        if self.trigram_keys.numel() == 0:
            return logits
        keys = prev2_ids.long() * self.config.vocab_size + prev_ids.long()
        table_keys = self.trigram_keys.to(prev_ids.device)
        positions = torch.searchsorted(table_keys, keys)
        in_range = positions.lt(table_keys.numel())
        safe_positions = positions.clamp_max(max(table_keys.numel() - 1, 0))
        matched = in_range & table_keys[safe_positions].eq(keys)
        if not matched.any():
            return logits
        rows = safe_positions[matched]
        ids = self.trigram_token_ids.to(prev_ids.device)[rows]
        vals = self.trigram_token_logits.to(prev_ids.device)[rows]
        valid = ids.ge(0)
        target = logits[matched]
        target.scatter_add_(1, ids.clamp_min(0), vals * valid.float())
        logits[matched] = target
        return logits

    def forward(
        self,
        context_ids: torch.Tensor,
        prev_ids: torch.Tensor | None = None,
        prev2_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_states = self.lexicon()
        components = self.context_components(context_ids)
        batch, num_components, dim = components.shape
        related = normalize_complex(self.memory(components.reshape(batch * num_components, dim)))
        related = related.reshape(batch, num_components, dim)
        amplitudes = torch.einsum("bcd,vd->bcv", related, token_states.conj())
        born_logits = torch.log(amplitudes.abs().square().clamp_min(1e-8))
        mix_log = torch.log_softmax(self.component_logits, dim=0).to(born_logits.dtype)
        logits = torch.logsumexp(born_logits + mix_log[None, :, None], dim=1) + self.output_bias[None, :]
        if prev_ids is not None:
            bg = self.bigram_logits[prev_ids.to(self.bigram_logits.device)].to(logits.device)
            logits = logits + self.config.bigram_strength * bg
            if prev2_ids is not None:
                logits = logits + self.config.trigram_strength * self.trigram_logits_for(prev2_ids, prev_ids)
        return logits

    @torch.no_grad()
    def diagnostics(self, context_ids: torch.Tensor) -> dict[str, float]:
        rho = self.density(context_ids)
        mix = torch.softmax(self.component_logits, dim=0)
        entropy = -(mix * mix.clamp_min(1e-8).log()).sum()
        return {
            "trace_mean": float(trace_real(rho).mean().cpu()),
            "purity_mean": float(purity(rho).mean().cpu()),
            "context_norm_mean": float(torch.linalg.vector_norm(self.context_state(context_ids), dim=-1).mean().cpu()),
            "component_entropy": float(entropy.detach().cpu()),
            "components": float(self.config.num_components),
        }

    @torch.no_grad()
    def generate(
        self,
        tokenizer: QuantumTokenizer,
        prompt: str,
        max_new_tokens: int = 48,
        temperature: float = 0.85,
        top_k: int = 24,
        repetition_penalty: float = 1.12,
        seed: int | None = None,
        device: str | torch.device | None = None,
    ) -> str:
        if device is None:
            device = next(self.parameters()).device
        generator = torch.Generator(device=str(device))
        if seed is not None:
            generator.manual_seed(seed)
        ids = [tokenizer.bos_id, tokenizer.user_id, *tokenizer.encode(prompt), tokenizer.assistant_id]
        generated: list[int] = []
        for _ in range(max_new_tokens):
            context = ids[-self.config.context_size :]
            padded = [tokenizer.pad_id] * (self.config.context_size - len(context)) + context
            context_tensor = torch.tensor([padded], dtype=torch.long, device=device)
            prev_tensor = torch.tensor([ids[-1]], dtype=torch.long, device=device)
            prev2_tensor = torch.tensor([ids[-2] if len(ids) >= 2 else tokenizer.pad_id], dtype=torch.long, device=device)
            logits = self(context_tensor, prev_tensor, prev2_tensor)[0]
            logits[tokenizer.pad_id] = -float("inf")
            logits[tokenizer.bos_id] = -float("inf")
            logits[tokenizer.user_id] = -float("inf")
            if generated:
                recent_counts = Counter(generated[-10:])
                for token_id, count in recent_counts.items():
                    if count >= 2:
                        logits[token_id] = -float("inf")
                logits[generated[-1]] = -float("inf")
                recent = set(generated[-8:])
                for token_id in recent:
                    logits[token_id] = logits[token_id] / repetition_penalty
            if temperature <= 0:
                next_id = int(torch.argmax(logits).item())
            else:
                logits = logits / max(temperature, 1e-4)
                if top_k > 0 and top_k < logits.numel():
                    values, indices = torch.topk(logits, top_k)
                    probs = torch.softmax(values, dim=-1)
                    sample = torch.multinomial(probs, 1, generator=generator)
                    next_id = int(indices[sample].item())
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_id = int(torch.multinomial(probs, 1, generator=generator).item())
            if next_id == tokenizer.eos_id:
                break
            ids.append(next_id)
            generated.append(next_id)
        return tokenizer.decode(generated)


def save_checkpoint(
    path: str | Path,
    model: QALFModel,
    tokenizer: QuantumTokenizer,
    metadata: dict[str, Any] | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(model.config),
        "model_state": model.state_dict(),
        "vocab": tokenizer.vocab,
        "metadata": metadata or {},
        "training_state": training_state or {},
    }
    torch.save(payload, path)
    with (path.parent / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload["metadata"], handle, indent=2)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[QALFModel, QuantumTokenizer, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    tokenizer = QuantumTokenizer(payload["vocab"])
    config = QALFConfig(**payload["config"])
    state = payload["model_state"]
    trigram_prior = None
    if "trigram_keys" in state and "trigram_token_ids" in state and "trigram_token_logits" in state:
        trigram_prior = {
            "keys": state["trigram_keys"],
            "token_ids": state["trigram_token_ids"],
            "token_logits": state["trigram_token_logits"],
        }
    model = QALFModel(config, trigram_prior=trigram_prior)
    model.load_state_dict(state, strict=False)
    return model, tokenizer, payload.get("metadata", {})


def device_for_training(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def cross_entropy_with_l2(model: QALFModel, logits: torch.Tensor, targets: torch.Tensor, l2_weight: float = 1e-5) -> torch.Tensor:
    loss = F.cross_entropy(logits, targets)
    relation_energy = model.memory.real.square().mean() + model.memory.imag.square().mean()
    return loss + l2_weight * relation_energy
