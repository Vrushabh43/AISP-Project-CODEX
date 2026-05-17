"""Singular-component attenuation utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _torch() -> Any:
    import torch

    return torch


def _as_float32_cpu(tensor: Any) -> Any:
    torch = _torch()
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    return tensor.detach().to(device="cpu", dtype=torch.float32)


def _validate_finite(name: str, tensor: Any) -> None:
    torch = _torch()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf values")


def select_top_spectral_components(S: Any, k: int) -> list[int]:
    """Select indices of the k largest singular values."""
    torch = _torch()
    S = _as_float32_cpu(S).flatten()
    _validate_finite("S", S)
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if S.numel() == 0 or k == 0:
        return []
    count = min(int(k), int(S.numel()))
    _, indices = torch.topk(S, k=count, largest=True, sorted=True)
    return [int(index) for index in indices.tolist()]


def attenuate_singular_values(S: Any, selected_indices: Iterable[int], gamma: float) -> Any:
    """Return singular values where selected components are multiplied by gamma."""
    S = _as_float32_cpu(S).flatten().clone()
    _validate_finite("S", S)
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    for index in selected_indices:
        index = int(index)
        if index < 0 or index >= int(S.numel()):
            raise IndexError(f"selected singular index out of range: {index}")
        S[index] = S[index] * float(gamma)
    _validate_finite("attenuated S", S)
    return S


def refactor_svd_to_lora_A_B(U: Any, S_new: Any, Vh: Any) -> tuple[Any, Any]:
    """Refactor U diag(S_new) Vh into LoRA A/B factors.

    Returns A_new [r, in_dim] and B_new [out_dim, r] so that
    B_new @ A_new == U @ diag(S_new) @ Vh.
    """
    torch = _torch()
    U = _as_float32_cpu(U)
    S_new = _as_float32_cpu(S_new).flatten()
    Vh = _as_float32_cpu(Vh)
    _validate_finite("U", U)
    _validate_finite("S_new", S_new)
    _validate_finite("Vh", Vh)

    if U.ndim != 2 or Vh.ndim != 2:
        raise ValueError(f"U and Vh must be 2D, got {tuple(U.shape)} and {tuple(Vh.shape)}")
    rank = int(S_new.numel())
    if U.shape[1] != rank or Vh.shape[0] != rank:
        raise ValueError(
            f"SVD shape mismatch: U {tuple(U.shape)}, S {tuple(S_new.shape)}, Vh {tuple(Vh.shape)}"
        )
    if torch.any(S_new < 0):
        raise ValueError("S_new must be non-negative")

    sqrt_s = torch.sqrt(S_new)
    B_new = U * sqrt_s.reshape(1, rank)
    A_new = sqrt_s.reshape(rank, 1) * Vh
    _validate_finite("A_new", A_new)
    _validate_finite("B_new", B_new)
    return A_new.contiguous(), B_new.contiguous()


def reconstruction_error(A_new: Any, B_new: Any, delta_w_target: Any, eps: float = 1e-12) -> dict[str, float]:
    """Compute dense reconstruction error for small optional sanity checks only."""
    torch = _torch()
    A_new = _as_float32_cpu(A_new)
    B_new = _as_float32_cpu(B_new)
    delta_w_target = _as_float32_cpu(delta_w_target)
    validate_lora_factor_shapes(A_new, B_new)
    if delta_w_target.ndim != 2 or delta_w_target.shape != (B_new.shape[0], A_new.shape[1]):
        raise ValueError(
            "delta_w_target shape must be "
            f"{(int(B_new.shape[0]), int(A_new.shape[1]))}, got {tuple(delta_w_target.shape)}"
        )
    reconstructed = B_new @ A_new
    error = torch.linalg.vector_norm(reconstructed - delta_w_target)
    target_norm = torch.linalg.vector_norm(delta_w_target)
    relative = error / target_norm.clamp_min(eps)
    return {
        "absolute_error": float(error.item()),
        "relative_error": float(relative.item()),
        "target_norm": float(target_norm.item()),
    }


def validate_lora_factor_shapes(A: Any, B: Any) -> int:
    """Validate LoRA factor shapes and return rank."""
    A = _as_float32_cpu(A)
    B = _as_float32_cpu(B)
    _validate_finite("A", A)
    _validate_finite("B", B)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be 2D tensors, got {tuple(A.shape)} and {tuple(B.shape)}")
    if A.shape[0] != B.shape[1]:
        raise ValueError(f"LoRA rank mismatch: A {tuple(A.shape)}, B {tuple(B.shape)}")
    return int(A.shape[0])
