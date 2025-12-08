from __future__ import annotations

import torch

from quantum_twin.core.BaseComponent import BaseComponent


class MathUtils(BaseComponent):
    """Static math helpers for quantum operations."""

    def __init__(self) -> None:
        super().__init__()
        self.logger.info("MathUtils initialized")

    @staticmethod
    def enforce_hermitian(matrix: torch.Tensor) -> torch.Tensor:
        return 0.5 * (matrix + matrix.conj().transpose(-1, -2))

    @staticmethod
    def enforce_trace_one(matrix: torch.Tensor) -> torch.Tensor:
        trace = torch.real(torch.diagonal(matrix, dim1=-2, dim2=-1).sum(dim=-1))
        trace = trace.unsqueeze(-1).unsqueeze(-1) + 1e-8
        return matrix / trace

    @staticmethod
    def enforce_psd(matrix: torch.Tensor) -> torch.Tensor:
        evals, evecs = torch.linalg.eigh(matrix)
        clipped = torch.clamp(evals.real, min=1e-8).to(evecs.dtype)
        psd = evecs @ torch.diag_embed(clipped) @ evecs.conj().transpose(-1, -2)
        return MathUtils.enforce_trace_one(psd)
