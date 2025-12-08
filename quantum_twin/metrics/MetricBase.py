from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch

from quantum_twin.core.BaseComponent import BaseComponent


class MetricBase(BaseComponent, ABC):
    """Base class for evaluation metrics."""

    def __init__(self) -> None:
        super().__init__()
        self.logger.info("MetricBase initialized")

    @abstractmethod
    def __call__(self, rho_pred: torch.Tensor, rho_true: torch.Tensor) -> Dict[str, float]:
        """Compute the metric."""
        raise NotImplementedError

