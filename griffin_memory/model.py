"""Causal Griffin-inspired hybrid decoder; no pretrained components."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .memory import MemoryAdapter, MemoryBank


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class RecurrentBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, r = config.d_model, config.recurrent_dim
        self.kernel = config.conv_kernel
        self.up = nn.Linear(d, r)
        self.branch = nn.Linear(d, r)
        self.conv = nn.Conv1d(r, r, self.kernel, groups=r)
        self.input_gate = nn.Linear(r, r)
        self.decay_gate = nn.Linear(r, r)
        # At decay_gate=0, exp(-8 * softplus(raw_decay) * sigmoid(0))
        # spans 0.9..0.999; gates then learn input-dependent timescales.
        initial_decay = torch.linspace(0.9, 0.999, r)
        self.raw_decay = nn.Parameter(torch.log(torch.expm1(-initial_decay.log() / 4)))
        self.down = nn.Linear(r, d, bias=False)

    def forward(self, x, state):
        u = self.up(x)
        b, _, r = u.shape
        h, history = state if state is not None else (
            torch.zeros(b, r, device=x.device, dtype=torch.float32),
            u.new_zeros(b, self.kernel - 1, r))
        conv_input = torch.cat([history.to(u.dtype), u], dim=1)
        convolved = self.conv(conv_input.transpose(1, 2)).transpose(1, 2)
        log_a = -8 * F.softplus(self.raw_decay.float()) * torch.sigmoid(self.decay_gate(convolved).float())
        a = log_a.exp()
        # -expm1(2 log(a)) is stable when a is close to one.
        scale = torch.sqrt((-torch.expm1(2 * log_a)).clamp_min(1e-8))
        inputs = torch.sigmoid(self.input_gate(convolved).float()) * convolved.float()
        outputs = []
        # Reference scan; intentionally explicit, replace with a fused scan to scale.
        for t in range(x.shape[1]):
            h = a[:, t] * h.float() + scale[:, t] * inputs[:, t]
            outputs.append(h)
        recurrent = torch.stack(outputs, dim=1).to(u.dtype)
        y = self.down(recurrent * F.gelu(self.branch(x)))
        history = conv_input[:, -(self.kernel - 1):] if self.kernel > 1 else conv_input[:, :0]
        return y, (h, history)


class LocalMQAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.window = config.window_size
        self.q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.kv = nn.Linear(config.d_model, 2 * self.head_dim, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        inv_freq = config.rope_theta ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rope(self, x, positions):
        # Compute angles in fp32 even under autocast, using absolute stream positions.
        angles = positions.float()[:, None] * self.inv_freq.float()[None, :]
        cos, sin = angles.cos()[None, None], angles.sin()[None, None]
        even, odd = x.float()[..., 0::2], x.float()[..., 1::2]
        return torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2).to(x.dtype)

    def forward(self, x, state, position):
        b, t, _ = x.shape
        positions = torch.arange(position, position + t, device=x.device)
        q = self.rope(self.q(x).view(b, t, self.heads, self.head_dim).transpose(1, 2), positions)
        k, v = self.kv(x).chunk(2, dim=-1)
        k = self.rope(k[:, None], positions)
        v = v[:, None]
        if state is not None:
            k = torch.cat([state[0].to(k.dtype), k], dim=2)
            v = torch.cat([state[1].to(v.dtype), v], dim=2)
        key_positions = torch.arange(position + t - k.shape[2], position + t, device=x.device)
        distance = positions[:, None] - key_positions[None, :]
        allowed = (distance >= 0) & (distance < self.window)
        # Single shared KV head (multi-query), broadcast over query heads.
        y = F.scaled_dot_product_attention(q, k.expand(-1, self.heads, -1, -1),
                                          v.expand(-1, self.heads, -1, -1), attn_mask=allowed)
        keep = self.window - 1
        cache = (k[:, :, -keep:], v[:, :, -keep:]) if keep else (k[:, :, :0], v[:, :, :0])
        return self.out(y.transpose(1, 2).reshape(b, t, -1)), cache


class GatedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up = nn.Linear(config.d_model, 2 * config.mlp_dim, bias=False)
        self.down = nn.Linear(config.mlp_dim, config.d_model, bias=False)

    def forward(self, x):
        value, gate = self.up(x).chunk(2, dim=-1)
        return self.down(value * F.gelu(gate))


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.recurrent_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.recurrent = RecurrentBlock(config)
        self.attention = LocalMQAttention(config)
        self.mlp = GatedMLP(config)

    def temporal(self, x, state, position):
        recurrent_state, attention_state = state if state is not None else (None, None)
        y, recurrent_state = self.recurrent(self.recurrent_norm(x), recurrent_state)
        x = x + y
        y, attention_state = self.attention(self.attention_norm(x), attention_state, position)
        return x + y, (recurrent_state, attention_state)


def tensor_tree(tree, device=None):
    if isinstance(tree, torch.Tensor):
        return tree.detach().to(device=device or tree.device)
    if isinstance(tree, (tuple, list)):
        return tuple(tensor_tree(v, device) for v in tree)
    return tree


@dataclass
class StreamState:
    layers: list
    banks: list
    position: int
    pending_sum: torch.Tensor
    pending_count: torch.Tensor
    pending_tokens: list

    def detach(self):
        self.layers = list(tensor_tree(self.layers))
        self.pending_sum = self.pending_sum.detach()
        return self

    def state_dict(self):
        return {"layers": tensor_tree(self.layers, "cpu"),
                "banks": [bank.state_dict() for bank in self.banks], "position": self.position,
                "pending_sum": self.pending_sum.detach().cpu(),
                "pending_count": self.pending_count.cpu(), "pending_tokens": self.pending_tokens}

    @classmethod
    def from_state_dict(cls, config, state, device):
        return cls(list(tensor_tree(state["layers"], device)),
                   [MemoryBank.from_state_dict(config, bank) for bank in state["banks"]],
                   state["position"], state["pending_sum"].to(device),
                   state["pending_count"].to(device), state["pending_tokens"])


class GriffinMemoryLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(config.n_layers))
        self.memory_norm = RMSNorm(config.d_model, config.norm_eps)
        self.memory = MemoryAdapter(config)
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.apply(self._initialize)
        nn.init.constant_(self.memory.gate.bias, -2.0)
        # Residual scale initialization, not a runtime modification of the recurrence.
        for name, parameter in self.named_parameters():
            if name.endswith(("down.weight", "out.weight", "memory.output.weight")):
                nn.init.normal_(parameter, std=0.02 / math.sqrt(3 * config.n_layers + 1))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            nn.init.normal_(module.weight, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def initial_state(self, batch_size, device):
        return StreamState([None] * self.config.n_layers,
                           [MemoryBank(self.config) for _ in range(batch_size)], 0,
                           torch.zeros(batch_size, self.config.d_model, device=device),
                           torch.zeros(batch_size, dtype=torch.long, device=device),
                           [[] for _ in range(batch_size)])

    def forward(self, tokens, state=None, token_mask=None, document_ids=None, metadata=None):
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError("tokens must be a nonempty [batch, time] tensor")
        b, t = tokens.shape
        if b == 0:
            raise ValueError("Batch must not be empty")
        if document_ids is not None and len(document_ids) != b:
            raise ValueError("One document ID is required per batch lane")
        if metadata is not None and len(metadata) != b:
            raise ValueError("One metadata record is required per batch lane")
        if token_mask is not None:
            if token_mask.shape != tokens.shape or token_mask.dtype != torch.bool:
                raise ValueError("token_mask must be boolean and match tokens")
            if (token_mask[:, 1:] & ~token_mask[:, :-1]).any():
                raise ValueError("Only right-padded document lanes are supported")
        state = self.initial_state(b, tokens.device) if state is None else state
        if len(state.banks) != b:
            raise ValueError("Batch size cannot change within a document stream")
        remaining = self.config.chunk_size - state.position % self.config.chunk_size
        if t > remaining:
            raise ValueError(f"Forward crosses a memory chunk boundary; pass at most {remaining} tokens")
        token_mask = torch.ones_like(tokens, dtype=torch.bool) if token_mask is None else token_mask
        x = self.embedding(tokens)
        layer_states = []
        for i, layer in enumerate(self.layers):
            x, layer_state = layer.temporal(x, state.layers[i], state.position)
            if i == len(self.layers) - 1:
                # One shared read adapter before the final layer's gated MLP.
                x = x + self.memory(self.memory_norm(x), state.banks)
            x = x + layer.mlp(layer.mlp_norm(x))
            layer_states.append(layer_state)
        hidden = self.final_norm(x)
        logits = F.linear(hidden, self.embedding.weight)  # tied output head
        # Writes occur AFTER logits and only at fixed, completed chunk boundaries.
        # A query can never retrieve its own chunk or future labels.
        with torch.no_grad():
            pending_sum = state.pending_sum + (hidden.float() * token_mask[..., None]).sum(1)
            pending_count = state.pending_count + token_mask.sum(1)
            pending_tokens = [old + row[mask].tolist() for old, row, mask in
                              zip(state.pending_tokens, tokens.detach().cpu(), token_mask.cpu())]
            position = state.position + t
            if position % self.config.chunk_size == 0:
                for i, bank in enumerate(state.banks):
                    if pending_count[i] > 0:
                        info = {"document_id": str(document_ids[i]) if document_ids else str(i),
                                "start": position - self.config.chunk_size,
                                "end": position - self.config.chunk_size + int(pending_count[i]),
                                "metadata": metadata[i] if metadata else {}}
                        self.memory.write(bank, pending_sum[i] / pending_count[i], pending_tokens[i], info)
                pending_sum = torch.zeros_like(pending_sum)
                pending_count = torch.zeros_like(pending_count)
                pending_tokens = [[] for _ in range(b)]
        new_state = StreamState(layer_states, state.banks, position, pending_sum, pending_count, pending_tokens)
        return logits, new_state
