from __future__ import annotations

import numpy as np
from typing import Tuple

from quantum_twin.core.BaseComponent import BaseComponent


class BlochSphereUtils(BaseComponent):
    """Utility functions for Bloch sphere conversions."""

    def __init__(self) -> None:
        super().__init__()
        self.logger.info("BlochSphereUtils ready")

    @staticmethod
    def density_to_bloch(rho: np.ndarray) -> Tuple[float, float, float]:
        """Convert 2x2 density matrix to Bloch vector."""
        sx = 2 * np.real(rho[0, 1])
        sy = -2 * np.imag(rho[0, 1])
        sz = np.real(rho[0, 0] - rho[1, 1])
        return float(sx), float(sy), float(sz)
