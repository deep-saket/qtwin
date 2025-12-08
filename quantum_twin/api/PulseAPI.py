from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from quantum_twin.core.BaseComponent import BaseComponent


class PulseAPI(BaseComponent):
    """Pulse helper utilities."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self._params = params
        self.logger.info("PulseAPI initialized")

    def gaussian_pulse(self, t: np.ndarray, amp: float, sigma: float) -> np.ndarray:
        return amp * np.exp(-0.5 * (t / sigma) ** 2)

    def drag_pulse(self, t: np.ndarray, amp: float, sigma: float, alpha: float) -> np.ndarray:
        gauss = self.gaussian_pulse(t, amp, sigma)
        deriv = -(t / (sigma**2)) * gauss
        return np.stack([gauss, alpha * deriv, np.zeros_like(gauss)], axis=-1)

    def square_pulse(self, t: np.ndarray, amp: float) -> np.ndarray:
        return amp * np.ones_like(t)

    def custom_pulse_from_user(self, pulse: List[float]) -> np.ndarray:
        return np.array(pulse)

    def normalize(self, pulse: np.ndarray, scale: float = 1.0) -> np.ndarray:
        max_val = np.max(np.abs(pulse)) + 1e-8
        return scale * pulse / max_val
