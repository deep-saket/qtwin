from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from quantum_twin.optimisation.ControlOptimizerBase import ControlOptimizerBase


class GaussianControlOptimizer(ControlOptimizerBase):
    """Samples controls from a Gaussian distribution."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(params)
        self._mean = float(self._params.get("mean", 0.0))
        self._std = float(self._params.get("std", 0.2))
        self._seed = int(self._params.get("seed", 0))
        self.logger.info("GaussianControlOptimizer mean=%.3f std=%.3f seed=%d", self._mean, self._std, self._seed)

    def run(self) -> List[float]:
        rng = np.random.default_rng(self._seed)
        controls = rng.normal(loc=self._mean, scale=self._std, size=3)
        optimized = controls.tolist()
        self.logger.info("Gaussian control solution %s", optimized)
        return optimized
