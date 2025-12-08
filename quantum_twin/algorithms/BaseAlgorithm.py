from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from quantum_twin.core.BaseComponent import BaseComponent


class BaseAlgorithm(BaseComponent, ABC):
    """Base class for algorithms runnable via AlgorithmAPI."""

    def __init__(self) -> None:
        super().__init__()
        self.logger.info("Initialized algorithm %s", self.__class__.__name__)

    @abstractmethod
    def run(self, twin: Any, **kwargs: Any) -> Any:
        """Execute algorithm using twin APIs."""
