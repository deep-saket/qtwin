from __future__ import annotations

from typing import Any, Dict

from quantum_twin.optimisation.OptimizerBase import OptimizerBase


class ControlOptimizerBase(OptimizerBase):
    """Base class for control optimizers."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__(params)
        self.logger.info("ControlOptimizerBase initialized with device=%s", self._device)
