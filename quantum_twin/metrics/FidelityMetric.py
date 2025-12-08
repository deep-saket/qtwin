from __future__ import annotations

from typing import Dict

import torch

from quantum_twin.metrics.MetricBase import MetricBase
from quantum_twin.utils.DensityMatrixUtils import DensityMatrixUtils


class FidelityMetric(MetricBase):
    """Computes state fidelity."""

    def __init__(self) -> None:
        super().__init__()
        self.logger.info("FidelityMetric ready")

    def __call__(self, rho_pred: torch.Tensor, rho_true: torch.Tensor) -> Dict[str, float]:
        fidelity = DensityMatrixUtils.fidelity(rho_pred, rho_true)
        value = float(torch.mean(fidelity).item())
        self.logger.debug("Fidelity=%.6f", value)
        return {"fidelity": value}

