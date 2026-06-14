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
    component_temperature: float = 1.0
    component_min_weight: float = 0.0
    attention_mode: str = "component"
    memory_mode: str = "linear"
    attention_layers: int = 2
    attention_phase_rank: int = 4
    attention_rotation_scale: float = 0.1


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


def _edge_groups(size: int) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Local-plus-log disjoint edge groups for unitary Givens sweeps."""
    groups: list[list[tuple[int, int]]] = []
    stride = 1
    while stride < size:
        starts = (0, 1) if stride == 1 else (0, max(1, stride // 2))
        for start in dict.fromkeys(starts):
            edges = [(left, left + stride) for left in range(start, size - stride, 2 * stride)]
            if edges:
                groups.append(edges)
        stride *= 2
    flat = [edge for group in groups for edge in group]
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for group in groups:
        next_cursor = cursor + len(group)
        ranges.append((cursor, next_cursor))
        cursor = next_cursor
    if not flat:
        return torch.empty(0, 2, dtype=torch.long), ranges
    return torch.tensor(flat, dtype=torch.long), ranges


class QuantumWindowAttention(nn.Module):
    """Unitary context-window mixer followed by position-projector readout."""

    def __init__(
        self,
        context_size: int,
        dimension: int,
        num_components: int,
        layers: int,
        phase_rank: int,
        rotation_scale: float,
        pad_id: int,
    ):
        super().__init__()
        self.context_size = context_size
        self.dimension = dimension
        self.num_components = num_components
        self.layers = max(0, layers)
        self.phase_rank = max(1, phase_rank)
        self.rotation_scale = float(rotation_scale)
        self.pad_id = pad_id

        position_edges, self._position_group_ranges = _edge_groups(context_size)
        feature_edges, self._feature_group_ranges = _edge_groups(dimension)
        self.register_buffer("position_edges", position_edges)
        self.register_buffer("feature_edges", feature_edges)

        pos_edges = position_edges.shape[0]
        feat_edges = feature_edges.shape[0]
        init = 0.02
        self.position_theta = nn.Parameter(torch.randn(self.layers, pos_edges) * init)
        self.position_phi = nn.Parameter(torch.randn(self.layers, pos_edges) * init)
        self.feature_theta = nn.Parameter(torch.randn(self.layers, feat_edges) * init)
        self.feature_phi = nn.Parameter(torch.randn(self.layers, feat_edges) * init)
        self.phase_position = nn.Parameter(torch.randn(self.layers, self.phase_rank, context_size) * init)
        self.phase_feature = nn.Parameter(torch.randn(self.layers, self.phase_rank, dimension) * init)
        self.readout_real = nn.Parameter(torch.randn(num_components, context_size) / math.sqrt(max(context_size, 1)))
        self.readout_imag = nn.Parameter(torch.randn(num_components, context_size) / math.sqrt(max(context_size, 1)))

    def initial_register(self, context_ids: torch.Tensor, token_states: torch.Tensor) -> torch.Tensor:
        embedded = token_states[context_ids]
        mask = context_ids.ne(self.pad_id).to(embedded.real.dtype)
        psi = embedded * mask[..., None].to(embedded.dtype)
        norms = torch.linalg.vector_norm(psi.reshape(psi.shape[0], -1), dim=-1)
        empty = norms.le(1e-8)
        if empty.any():
            psi = psi.clone()
            psi[empty, -1, :] = token_states[self.pad_id]
        norms = torch.linalg.vector_norm(psi.reshape(psi.shape[0], -1), dim=-1, keepdim=True).clamp_min(1e-8)
        return psi / norms[:, :, None].to(psi.dtype)

    def _apply_position_group(self, psi: torch.Tensor, layer: int, start: int, end: int) -> torch.Tensor:
        if start == end:
            return psi
        edges = self.position_edges[start:end]
        left = edges[:, 0]
        right = edges[:, 1]
        theta = self.rotation_scale * torch.tanh(self.position_theta[layer, start:end])
        phi = self.position_phi[layer, start:end]
        c = theta.cos()[None, :, None].to(psi.dtype)
        s = theta.sin()[None, :, None].to(psi.dtype)
        phase = torch.exp(1j * phi)[None, :, None].to(psi.dtype)
        x = psi.index_select(1, left)
        y = psi.index_select(1, right)
        rotated = psi.clone()
        rotated[:, left, :] = c * x - phase * s * y
        rotated[:, right, :] = phase.conj() * s * x + c * y
        return rotated

    def _apply_feature_group(self, psi: torch.Tensor, layer: int, start: int, end: int) -> torch.Tensor:
        if start == end:
            return psi
        edges = self.feature_edges[start:end]
        left = edges[:, 0]
        right = edges[:, 1]
        theta = self.rotation_scale * torch.tanh(self.feature_theta[layer, start:end])
        phi = self.feature_phi[layer, start:end]
        c = theta.cos()[None, None, :].to(psi.dtype)
        s = theta.sin()[None, None, :].to(psi.dtype)
        phase = torch.exp(1j * phi)[None, None, :].to(psi.dtype)
        x = psi.index_select(2, left)
        y = psi.index_select(2, right)
        rotated = psi.clone()
        rotated[:, :, left] = c * x - phase * s * y
        rotated[:, :, right] = phase.conj() * s * x + c * y
        return rotated

    def evolve(self, psi: torch.Tensor) -> torch.Tensor:
        for layer in range(self.layers):
            for start, end in self._position_group_ranges:
                psi = self._apply_position_group(psi, layer, start, end)
            pos = torch.tanh(self.phase_position[layer])
            feat = torch.tanh(self.phase_feature[layer])
            angle = self.rotation_scale * torch.einsum("rl,rd->ld", pos, feat)
            psi = psi * torch.exp(1j * angle).to(psi.dtype)[None, :, :]
            for start, end in self._feature_group_ranges:
                psi = self._apply_feature_group(psi, layer, start, end)
        return psi

    def readout_projectors(self) -> torch.Tensor:
        return normalize_complex(torch.complex(self.readout_real, self.readout_imag))

    def components_from_register(self, psi: torch.Tensor, fallback_state: torch.Tensor) -> torch.Tensor:
        projectors = self.readout_projectors()
        components = torch.einsum("cl,bld->bcd", projectors.conj(), psi)
        norms = torch.linalg.vector_norm(components, dim=-1)
        empty = norms.le(1e-8)
        if empty.any():
            components = components.clone()
            components[empty] = fallback_state
        return normalize_complex(components)

    def forward(self, context_ids: torch.Tensor, token_states: torch.Tensor) -> torch.Tensor:
        psi = self.initial_register(context_ids, token_states)
        evolved = self.evolve(psi)
        return self.components_from_register(evolved, token_states[self.pad_id])

    @torch.no_grad()
    def norm_drift(self, context_ids: torch.Tensor, token_states: torch.Tensor) -> tuple[float, float]:
        psi = self.initial_register(context_ids, token_states)
        before = torch.linalg.vector_norm(psi.reshape(psi.shape[0], -1), dim=-1)
        evolved = self.evolve(psi)
        after = torch.linalg.vector_norm(evolved.reshape(evolved.shape[0], -1), dim=-1)
        drift = (after - before).abs()
        return float(drift.mean().cpu()), float(drift.max().cpu())


class UnitaryFeatureMemory(nn.Module):
    """Norm-preserving feature-register memory used by the unitary experiment."""

    def __init__(self, dimension: int, layers: int, rotation_scale: float):
        super().__init__()
        self.dimension = dimension
        self.layers = max(1, layers)
        self.rotation_scale = float(rotation_scale)
        feature_edges, self._feature_group_ranges = _edge_groups(dimension)
        self.register_buffer("feature_edges", feature_edges)
        edge_count = feature_edges.shape[0]
        init = 0.02
        self.theta = nn.Parameter(torch.randn(self.layers, edge_count) * init)
        self.phi = nn.Parameter(torch.randn(self.layers, edge_count) * init)
        self.phase = nn.Parameter(torch.randn(self.layers, dimension) * init)

    def _apply_group(self, state: torch.Tensor, layer: int, start: int, end: int) -> torch.Tensor:
        if start == end:
            return state
        edges = self.feature_edges[start:end]
        left = edges[:, 0]
        right = edges[:, 1]
        theta = self.rotation_scale * torch.tanh(self.theta[layer, start:end])
        phi = self.phi[layer, start:end]
        c = theta.cos()[None, :].to(state.dtype)
        s = theta.sin()[None, :].to(state.dtype)
        phase = torch.exp(1j * phi)[None, :].to(state.dtype)
        x = state.index_select(1, left)
        y = state.index_select(1, right)
        rotated = state.clone()
        rotated[:, left] = c * x - phase * s * y
        rotated[:, right] = phase.conj() * s * x + c * y
        return rotated

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = normalize_complex(state)
        for layer in range(self.layers):
            phase = torch.exp(1j * (self.rotation_scale * torch.tanh(self.phase[layer]))).to(state.dtype)
            state = state * phase[None, :]
            for start, end in self._feature_group_ranges:
                state = self._apply_group(state, layer, start, end)
        return state

    def energy(self) -> torch.Tensor:
        return self.theta.square().mean() + self.phi.square().mean() + self.phase.square().mean()


class QALFModel(nn.Module):
    """Generative language model without attention, recurrence, or pretrained weights."""

    def __init__(
        self,
        config: QALFConfig,
        bigram_logits: torch.Tensor | None = None,
        trigram_prior: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        if config.attention_mode not in {"component", "entangling"}:
            raise ValueError("attention_mode must be 'component' or 'entangling'")
        if config.memory_mode not in {"linear", "unitary"}:
            raise ValueError("memory_mode must be 'linear' or 'unitary'")
        self.config = config
        self.lexicon = QuantumLexicon(config.vocab_size, config.dimension)
        self.memory = EntangledMemory(config.dimension, config.num_relations)
        self.unitary_memory = UnitaryFeatureMemory(
            config.dimension,
            config.num_relations,
            config.attention_rotation_scale,
        )
        self.window_attention = QuantumWindowAttention(
            config.context_size,
            config.dimension,
            config.num_components,
            config.attention_layers,
            config.attention_phase_rank,
            config.attention_rotation_scale,
            config.pad_id,
        )
        self.phase_frequencies = nn.Parameter(torch.linspace(0.05, 1.25, config.dimension))
        self.component_logits = nn.Parameter(torch.zeros(config.num_components))
        self.component_phase_offsets = nn.Parameter(torch.linspace(0.0, 1.0, config.num_components))
        # Learnable positional focus: each component learns where in the context to look
        self.component_centers = nn.Parameter(torch.linspace(0.0, 1.0, config.num_components))
        self.component_log_widths = nn.Parameter(torch.full((config.num_components,), math.log(0.5)))
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

    def component_mixture(self) -> torch.Tensor:
        temperature = max(float(self.config.component_temperature), 1e-4)
        mix = torch.softmax(self.component_logits / temperature, dim=0)
        floor = max(float(self.config.component_min_weight), 0.0)
        max_floor = 1.0 / max(self.config.num_components, 1)
        floor = min(floor, max_floor)
        if floor > 0.0:
            mix = mix * (1.0 - floor * self.config.num_components) + floor
            mix = mix / mix.sum().clamp_min(1e-8)
        return mix

    def component_weights(self, context_ids: torch.Tensor) -> torch.Tensor:
        batch, length = context_ids.shape
        positions = torch.linspace(0.0, 1.0, length, device=context_ids.device)
        centers = torch.sigmoid(self.component_centers)           # (num_components,) in [0, 1]
        widths = self.component_log_widths.exp().clamp_min(0.05)  # (num_components,) positive
        # Gaussian focus window per component: (num_components, length)
        weights = torch.exp(-0.5 * ((positions[None, :] - centers[:, None]) / widths[:, None]).square())
        mask = context_ids.ne(self.config.pad_id).float()
        return weights[None, :, :] * mask[:, None, :]

    def context_components(self, context_ids: torch.Tensor) -> torch.Tensor:
        states = self.lexicon()
        if self.config.attention_mode == "entangling":
            return self.window_attention(context_ids, states)
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
        components = self.context_components(context_ids)
        mix = self.component_mixture().to(components.dtype)
        return normalize_complex(torch.einsum("c,bcd->bd", mix, components))

    def density(self, context_ids: torch.Tensor) -> torch.Tensor:
        components = self.context_components(context_ids)
        mix = self.component_mixture().to(components.dtype)
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
        flat_components = components.reshape(batch * num_components, dim)
        if self.config.memory_mode == "unitary":
            related = self.unitary_memory(flat_components)
        else:
            related = normalize_complex(self.memory(flat_components))
        related = related.reshape(batch, num_components, dim)
        amplitudes = torch.einsum("bcd,vd->bcv", related, token_states.conj())
        born_logits = torch.log(amplitudes.abs().square().clamp_min(1e-8))
        mix_log = self.component_mixture().clamp_min(1e-8).log().to(born_logits.dtype)
        logits = torch.logsumexp(born_logits + mix_log[None, :, None], dim=1) + self.output_bias[None, :]
        if prev_ids is not None:
            bg = self.bigram_logits[prev_ids.to(self.bigram_logits.device)].to(logits.device)
            logits = logits + self.config.bigram_strength * bg
            if prev2_ids is not None:
                logits = logits + self.config.trigram_strength * self.trigram_logits_for(prev2_ids, prev_ids)
        return logits

    @torch.no_grad()
    def diagnostics(self, context_ids: torch.Tensor) -> dict[str, float | str]:
        rho = self.density(context_ids)
        purity_values = purity(rho)
        components = self.context_components(context_ids)
        overlap = component_overlap_matrix(components)
        offdiag = offdiagonal_values(overlap)
        mix = self.component_mixture()
        entropy = -(mix * mix.clamp_min(1e-8).log()).sum()
        if self.config.attention_mode == "entangling":
            drift_mean, drift_max = self.window_attention.norm_drift(context_ids, self.lexicon())
        else:
            drift_mean, drift_max = 0.0, 0.0
        return {
            "trace_mean": float(trace_real(rho).mean().cpu()),
            "purity_mean": float(purity_values.mean().cpu()),
            "density_effective_rank": float((1.0 / purity_values.clamp_min(1e-8)).mean().cpu()),
            "context_norm_mean": float(torch.linalg.vector_norm(self.context_state(context_ids), dim=-1).mean().cpu()),
            "component_entropy": float(entropy.detach().cpu()),
            "component_effective_count": float(torch.exp(entropy).detach().cpu()),
            "component_weight_min": float(mix.min().detach().cpu()),
            "component_weight_max": float(mix.max().detach().cpu()),
            "component_overlap_mean": float(offdiag.mean().cpu()) if offdiag.numel() else 0.0,
            "component_overlap_max": float(offdiag.max().cpu()) if offdiag.numel() else 0.0,
            "components": float(self.config.num_components),
            "bigram_strength": self.config.bigram_strength,
            "trigram_strength": self.config.trigram_strength,
            "window_norm_drift_mean": drift_mean,
            "window_norm_drift_max": drift_max,
            "attention_layers": float(self.config.attention_layers),
            "attention_phase_rank": float(self.config.attention_phase_rank),
            "attention_mode": self.config.attention_mode,
            "memory_mode": self.config.memory_mode,
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


def component_overlap_matrix(components: torch.Tensor) -> torch.Tensor:
    """Squared pairwise overlaps between normalized complex context components."""
    gram = torch.einsum("bcd,bed->bce", components.conj(), components)
    return gram.abs().square()


def offdiagonal_values(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-1] <= 1:
        return matrix.new_empty(0)
    eye = torch.eye(matrix.shape[-1], dtype=torch.bool, device=matrix.device)
    return matrix[..., ~eye]


def component_diversity_loss(
    model: QALFModel,
    context_ids: torch.Tensor,
    target_overlap: float = 0.05,
) -> torch.Tensor:
    """Penalize component states that collapse into the same Hilbert direction."""
    components = model.context_components(context_ids)
    overlap = component_overlap_matrix(components)
    offdiag = offdiagonal_values(overlap)
    if offdiag.numel() == 0:
        return context_ids.new_tensor(0.0, dtype=torch.float32)
    return F.relu(offdiag - target_overlap).square().mean()


def cross_entropy_with_l2(
    model: QALFModel,
    logits: torch.Tensor,
    targets: torch.Tensor,
    l2_weight: float = 1e-5,
    entropy_weight: float = 0.0,
    component_context_ids: torch.Tensor | None = None,
    component_diversity_weight: float = 0.0,
    component_diversity_target: float = 0.05,
) -> torch.Tensor:
    loss = F.cross_entropy(logits, targets)
    if model.config.memory_mode == "unitary":
        relation_energy = model.unitary_memory.energy()
    else:
        relation_energy = model.memory.real.square().mean() + model.memory.imag.square().mean()
    loss = loss + l2_weight * relation_energy
    if entropy_weight > 0.0:
        comp_mix = model.component_mixture()
        comp_entropy = -(comp_mix * comp_mix.clamp_min(1e-8).log()).sum()
        rel_mix = torch.softmax(model.memory.mixing_logits, dim=0)
        rel_entropy = -(rel_mix * rel_mix.clamp_min(1e-8).log()).sum()
        # subtract entropy to maximise it (resist collapse of both mixture distributions)
        loss = loss - entropy_weight * (comp_entropy + rel_entropy)
    if component_diversity_weight > 0.0 and component_context_ids is not None:
        loss = loss + component_diversity_weight * component_diversity_loss(
            model,
            component_context_ids,
            target_overlap=component_diversity_target,
        )
    return loss
