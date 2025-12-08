from __future__ import annotations

from typing import Any, Dict

import torch

from quantum_twin.optimisation.OptimizerBase import OptimizerBase


class ParameterEstimator(OptimizerBase):
    """Estimates Hamiltonian parameters from data."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(params)
        self.logger.info("ParameterEstimator ready")

    def run(self) -> Dict[str, float]:
        # Placeholder heuristic
        estimated = {"drift": float(self._params.get("drift", 0.0))}
        self.logger.info("Estimated parameters %s", estimated)
        return estimated

