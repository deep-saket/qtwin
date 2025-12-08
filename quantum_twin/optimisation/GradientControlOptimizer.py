from __future__ import annotations

from typing import Any, Dict, List

import torch

from quantum_twin.optimisation.ControlOptimizerBase import ControlOptimizerBase


class GradientControlOptimizer(ControlOptimizerBase):
    """Optimizes controls via simple gradient descent on a quadratic surrogate."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(params)
        self._steps = int(self._params.get("steps", 100))
        self._lr = float(self._params.get("learning_rate", 1e-2))
        self._init_scale = float(self._params.get("scale", 0.5))
        self.logger.info("GradientControlOptimizer steps=%d lr=%.4f scale=%.3f", self._steps, self._lr, self._init_scale)

    def run(self) -> List[float]:
        device = torch.device(self._device)
        controls = torch.randn(3, device=device) * self._init_scale
        controls = torch.nn.Parameter(controls)
        optimizer = torch.optim.SGD([controls], lr=self._lr)
        for _ in range(self._steps):
            optimizer.zero_grad()
            loss = torch.sum(controls ** 2)
            loss.backward()
            optimizer.step()
        optimized = controls.detach().cpu().tolist()
        self.logger.info("Gradient control solution %s", optimized)
        return optimized
