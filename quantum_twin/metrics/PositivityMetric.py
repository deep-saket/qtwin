from __future__ import annotations

from typing import Dict

import torch

from quantum_twin.metrics.MetricBase import MetricBase
from quantum_twin.utils.DensityMatrixUtils import DensityMatrixUtils


class PositivityMetric(MetricBase):
    """Measures minimal eigenvalue to assess positivity."""

    def __init__(self) -> None:
        super().__init__()
        self.logger.info("PositivityMetric ready")

    def __call__(self, rho_pred: torch.Tensor, rho_true: torch.Tensor) -> Dict[str, float]:
        rho_psd = DensityMatrixUtils.clamp_physical(rho_pred)
        eigvals = torch.real(torch.linalg.eigvals(rho_psd))
        min_eval = torch.min(eigvals)
        value = float(min_eval.item())
        self.logger.debug("Positivity min eigenvalue=%.6f", value)
        return {"min_eigenvalue": value}

