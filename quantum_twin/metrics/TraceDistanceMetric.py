from __future__ import annotations

from typing import Dict

import torch

from quantum_twin.metrics.MetricBase import MetricBase
from quantum_twin.utils.DensityMatrixUtils import DensityMatrixUtils


class TraceDistanceMetric(MetricBase):
    """Computes trace distance between density matrices."""

    def __init__(self) -> None:
        super().__init__()
        self.logger.info("TraceDistanceMetric ready")

    def __call__(self, rho_pred: torch.Tensor, rho_true: torch.Tensor) -> Dict[str, float]:
        diff = rho_pred - rho_true
        evals = torch.linalg.eigvals(diff)
        distance = 0.5 * torch.sum(torch.abs(evals), dim=-1)
        value = float(torch.mean(torch.real(distance)).item())
        self.logger.debug("TraceDistance=%.6f", value)
        return {"trace_distance": value}

