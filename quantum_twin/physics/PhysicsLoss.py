from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch

from quantum_twin.core.BaseComponent import BaseComponent


class PhysicsLoss(BaseComponent, ABC):
    """Abstract base for physics-informed loss terms."""

    def __init__(self, weights: Dict[str, float]) -> None:
        super().__init__()
        self._weights = weights
        self.logger.info("PhysicsLoss initialized with weights: %s", weights)

    @abstractmethod
    def compute(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the physics-informed loss for the given batch."""
        raise NotImplementedError

