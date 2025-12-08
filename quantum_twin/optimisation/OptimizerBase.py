from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from quantum_twin.core.BaseComponent import BaseComponent


class OptimizerBase(BaseComponent, ABC):
    """Abstract base class for optimisation routines."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self._params = params
        self._device = params.get("device", "cpu")
        self.logger.info("OptimizerBase params=%s device=%s", params, self._device)

    @abstractmethod
    def run(self) -> Any:
        """Execute the optimization."""
        raise NotImplementedError
