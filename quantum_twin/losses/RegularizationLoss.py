from __future__ import annotations

from typing import Dict

import torch

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.utils.DensityMatrixUtils import DensityMatrixUtils


class RegularizationLoss(BaseComponent):
    """Penalty terms enforcing physical constraints."""

    def __init__(self, weights: Dict[str, float]) -> None:
        super().__init__()
        self._weights = weights
        self.logger.info("RegularizationLoss initialized weights=%s", weights)

    def compute(self, rho: torch.Tensor) -> torch.Tensor:
        trace_penalty = torch.mean(
            torch.abs(torch.real(torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1)) - 1.0)
        )
        hermitian_penalty = torch.mean(torch.abs(rho - rho.conj().transpose(-1, -2)))
        rho_psd = DensityMatrixUtils.clamp_physical(rho)
        positivity_penalty = torch.mean(torch.relu(-torch.real(torch.linalg.eigvals(rho_psd))))
        reg = (
            self._weights.get("trace", 0.0) * trace_penalty
            + self._weights.get("hermitian", 0.0) * hermitian_penalty
            + self._weights.get("positivity", 0.0) * positivity_penalty
        )
        return reg

