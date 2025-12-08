from __future__ import annotations

import torch

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.utils.MathUtils import MathUtils


class DensityMatrixUtils(BaseComponent):
    """Physical constraint utilities for density matrices."""

    def __init__(self) -> None:
        super().__init__()
        self.logger.info("DensityMatrixUtils ready")

    @staticmethod
    def clamp_physical(rho: torch.Tensor) -> torch.Tensor:
        hermitian = MathUtils.enforce_hermitian(rho)
        return MathUtils.enforce_psd(hermitian)

    @staticmethod
    def fidelity(rho: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        evals, evecs = torch.linalg.eigh(rho)
        sqrt_diag = torch.diag_embed(torch.sqrt(torch.clamp(evals, min=0.0)))
        sqrt_rho = evecs @ sqrt_diag @ evecs.conj().transpose(-1, -2)
        inner = sqrt_rho @ sigma @ sqrt_rho
        evals_inner = torch.linalg.eigvals(inner)
        return torch.real(torch.sum(torch.sqrt(torch.clamp(evals_inner, min=0.0)), dim=-1))

