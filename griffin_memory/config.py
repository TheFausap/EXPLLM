from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 259
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    recurrent_dim: int = 256
    mlp_dim: int = 768
    conv_kernel: int = 4
    window_size: int = 256
    chunk_size: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    memory_capacity: int = 256
    memory_top_k: int = 4
    memory_key_dim: int = 64
    lsh_tables: int = 4
    lsh_bits: int = 8
    lsh_seed: int = 1729
    memory_backend: str = "lsh"

    def __post_init__(self):
        for name in ("vocab_size", "d_model", "n_layers", "n_heads", "recurrent_dim",
                     "mlp_dim", "conv_kernel", "window_size", "chunk_size",
                     "memory_capacity", "memory_top_k", "memory_key_dim", "lsh_tables", "lsh_bits"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.vocab_size < 4:
            raise ValueError("vocab_size must include specials and text")
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("head dimension must be an even integer")
        if self.lsh_bits > 24:
            raise ValueError("lsh_bits must be <= 24")
        if self.rope_theta <= 0 or self.norm_eps <= 0:
            raise ValueError("rope_theta and norm_eps must be positive")
        if self.memory_backend not in {"lsh", "exact"}:
            raise ValueError("memory_backend must be lsh or exact")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)
