from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from quantum_twin.core.BaseComponent import BaseComponent


class DemoDataAPI(BaseComponent):
    """Provides pre-built demo datasets for quick starts."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self._params = params
        self.logger.info("DemoDataAPI ready")

    def simple_pulses(self) -> np.ndarray:
        return np.array([[0.1, 0.0, 0.0], [0.2, -0.1, 0.0]])

    def noise_profiles(self) -> Dict[str, float]:
        return {"t1": 30.0, "t2": 20.0, "tphi": 0.01}

    def sample_trajectories(self) -> Dict[str, Any]:
        t = np.linspace(0, 1.0, 10)
        rho = np.zeros((10, 2, 2), dtype=np.complex128)
        return {"t": t, "rho": rho}

    def optimized_pulses(self) -> np.ndarray:
        return np.array([[0.05, -0.02, 0.0]])

    def surrogate_predictions(self) -> Dict[str, Any]:
        return {"rho_pred": np.zeros((10, 2, 2), dtype=np.complex128)}
