from __future__ import annotations

from typing import Any, Dict

from quantum_twin.optimisation.OptimizerBase import OptimizerBase


class SurrogateFitter(OptimizerBase):
    """Fits surrogate models to expensive simulators."""

    def __init__(self, **params: Dict[str, Any]) -> None:
        super().__init__(**params)
        self.logger.info("SurrogateFitter ready")

    def run(self) -> Dict[str, Any]:
        result = {"status": "completed", "epochs": self._params.get("epochs", 10)}
        self.logger.info("Surrogate fitting complete: %s", result)
        return result
