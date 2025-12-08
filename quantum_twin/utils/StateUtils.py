from __future__ import annotations

import torch

from quantum_twin.core.BaseComponent import BaseComponent


class StateUtils(BaseComponent):
    """Utility helpers for state vector operations."""

    def __init__(self) -> None:
        super().__init__()
        self.logger.info("StateUtils ready")

    @staticmethod
    def vectorize_density(rho: torch.Tensor) -> torch.Tensor:
        return rho.view(rho.shape[0], -1)

    @staticmethod
    def devectorize_density(vec: torch.Tensor) -> torch.Tensor:
        side = int(vec.shape[-1] ** 0.5)
        return vec.view(vec.shape[0], side, side)

