"""Compact spectral utilities for LoRA update matrices."""

from __future__ import annotations

import math
from typing import Any


def _torch() -> Any:
    import torch

    return torch


def _as_float32_cpu(tensor: Any) -> Any:
    torch = _torch()
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    return tensor.detach().to(device="cpu", dtype=torch.float32)


def compact_svd_singular_values(A: Any, B: Any, eps: float = 1e-12) -> Any:
    """Return singular values of Delta W = B @ A via a compact r x r SVD.

    A is expected to have shape [r, in_dim], B shape [out_dim, r].
    The dense out_dim x in_dim Delta W matrix is never formed.
    """
    torch = _torch()
    A = _as_float32_cpu(A)
    B = _as_float32_cpu(B)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be 2D tensors, got {tuple(A.shape)} and {tuple(B.shape)}")
    if A.shape[0] != B.shape[1]:
        raise ValueError(f"Rank mismatch: A shape {tuple(A.shape)}, B shape {tuple(B.shape)}")

    rank = int(A.shape[0])
    if rank == 0:
        return torch.empty(0, dtype=torch.float32)

    # B = Q_B R_B, A.T = Q_A R_A. Singular values of B @ A are the
    # singular values of the small core R_B @ R_A.T.
    _, r_b = torch.linalg.qr(B, mode="reduced")
    _, r_a = torch.linalg.qr(A.T.contiguous(), mode="reduced")
    core = r_b @ r_a.T
    singular_values = torch.linalg.svdvals(core)
    return torch.where(singular_values.abs() <= eps, torch.zeros_like(singular_values), singular_values)


def compact_svd_full_for_pair(A: Any, B: Any, eps: float = 1e-12) -> tuple[Any, Any, Any]:
    """Return compact SVD factors for Delta W = B @ A without forming Delta W.

    A is expected to have shape [r, in_dim], B shape [out_dim, r]. The returned
    factors have shapes U [out_dim, r], S [r], Vh [r, in_dim] such that
    B @ A == U @ diag(S) @ Vh up to numerical precision.
    """
    torch = _torch()
    A = _as_float32_cpu(A)
    B = _as_float32_cpu(B)

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be 2D tensors, got {tuple(A.shape)} and {tuple(B.shape)}")
    if A.shape[0] != B.shape[1]:
        raise ValueError(f"Rank mismatch: A shape {tuple(A.shape)}, B shape {tuple(B.shape)}")

    rank = int(A.shape[0])
    if rank == 0:
        return (
            torch.empty((int(B.shape[0]), 0), dtype=torch.float32),
            torch.empty(0, dtype=torch.float32),
            torch.empty((0, int(A.shape[1])), dtype=torch.float32),
        )

    q_b, r_b = torch.linalg.qr(B, mode="reduced")
    q_a, r_a = torch.linalg.qr(A.T.contiguous(), mode="reduced")
    core = r_b @ r_a.T
    u_core, singular_values, vh_core = torch.linalg.svd(core, full_matrices=False)
    singular_values = torch.where(
        singular_values.abs() <= eps,
        torch.zeros_like(singular_values),
        singular_values,
    )
    U = q_b @ u_core
    Vh = vh_core @ q_a.T
    return U.contiguous(), singular_values.contiguous(), Vh.contiguous()


def compute_energy_shares(S: Any, eps: float = 1e-12) -> Any:
    torch = _torch()
    S = _as_float32_cpu(S)
    if S.numel() == 0:
        return S
    energy = S.square()
    total = energy.sum()
    if float(total) <= eps:
        return torch.zeros_like(S)
    return energy / total


def compute_topk_energy(S: Any, k: int, eps: float = 1e-12) -> float:
    shares = compute_energy_shares(S, eps=eps)
    if shares.numel() == 0 or k <= 0:
        return 0.0
    return float(shares[: min(k, int(shares.numel()))].sum().item())


def compute_spectral_entropy(S: Any, eps: float = 1e-12, normalize: bool = True) -> float:
    torch = _torch()
    shares = compute_energy_shares(S, eps=eps)
    positive = shares[shares > eps]
    if positive.numel() == 0:
        return 0.0
    entropy = -float((positive * torch.log(positive)).sum().item())
    if normalize and shares.numel() > 1:
        return entropy / math.log(float(shares.numel()))
    return entropy


def compute_effective_rank(S: Any, eps: float = 1e-12) -> float:
    entropy = compute_spectral_entropy(S, eps=eps, normalize=False)
    return float(math.exp(entropy)) if entropy > 0.0 else 0.0


def frobenius_norm_from_singular_values(S: Any) -> float:
    torch = _torch()
    S = _as_float32_cpu(S)
    if S.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(S).item())
