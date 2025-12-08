from __future__ import annotations

import numpy as np
from typing import Callable

from quantum_twin.core.BaseComponent import BaseComponent


class PulseGenerator(BaseComponent):
    """Generates randomized control pulses."""

    def __init__(self, scale: float = 0.5, seed: int | None = None) -> None:
        super().__init__()
        self._scale = scale
        self._rng = np.random.default_rng(seed)
        self.logger.info("PulseGenerator initialized scale=%.3f seed=%s", scale, seed)

    def sample_pulse(self) -> np.ndarray:
        pulse = self._rng.uniform(-self._scale, self._scale, size=(3,))
        self.logger.debug("Sampled pulse %s", pulse)
        return pulse

    def sinusoidal(self, omega: float = 2.0 * np.pi) -> Callable[[np.ndarray], np.ndarray]:
        def fn(t: np.ndarray) -> np.ndarray:
            return self._scale * np.stack([np.sin(omega * t), np.cos(omega * t), np.zeros_like(t)], axis=-1)

        return fn

