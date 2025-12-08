from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from quantum_twin.optimisation.ControlOptimizerBase import ControlOptimizerBase


class UniformControlOptimizer(ControlOptimizerBase):
    """Samples controls from a uniform distribution."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(params)
        self._scale = float(self._params.get("scale", 0.5))
        self._seed = int(self._params.get("seed", 0))
        self.logger.info("UniformControlOptimizer scale=%.3f seed=%d", self._scale, self._seed)

    def run(self) -> List[float]:
        rng = np.random.default_rng(self._seed)
        controls = rng.uniform(-self._scale, self._scale, size=3)
        optimized = controls.tolist()
        self.logger.info("Uniform control solution %s", optimized)
        return optimized
