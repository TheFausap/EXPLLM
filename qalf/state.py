"""Quantum-inspired state utilities."""

from __future__ import annotations

import torch


def normalize_complex(states: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.vector_norm(states, dim=-1, keepdim=True).clamp_min(eps)
    return states / norm


def density_from_state(state: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Build trace-normalized density matrices from state vectors.

    Shape:
        state: ``(..., d)`` complex tensor.
        return: ``(..., d, d)`` complex tensor.
    """
    state = normalize_complex(state, eps=eps)
    rho = state.unsqueeze(-1) * state.conj().unsqueeze(-2)
    trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1).real.clamp_min(eps)
    return rho / trace[..., None, None].to(rho.dtype)


def is_hermitian(matrix: torch.Tensor, atol: float = 1e-5) -> bool:
    return torch.allclose(matrix, matrix.conj().transpose(-1, -2), atol=atol, rtol=0.0)


def trace_real(matrix: torch.Tensor) -> torch.Tensor:
    return torch.diagonal(matrix, dim1=-2, dim2=-1).sum(dim=-1).real


def purity(matrix: torch.Tensor) -> torch.Tensor:
    product = matrix @ matrix
    return torch.diagonal(product, dim1=-2, dim2=-1).sum(dim=-1).real
