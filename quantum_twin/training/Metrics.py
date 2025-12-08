from __future__ import annotations

from typing import Dict

from quantum_twin.core.BaseComponent import BaseComponent


class Metrics(BaseComponent):
    """Lightweight metrics container."""

    def __init__(self) -> None:
        super().__init__()
        self.values: Dict[str, float] = {}
        self.logger.info("Metrics initialized")

    def log(self, name: str, value: float) -> None:
        self.values[name] = value
        self.logger.debug("Metric %s=%.6f", name, value)

    def snapshot(self) -> Dict[str, float]:
        return dict(self.values)

