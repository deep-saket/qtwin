from __future__ import annotations

from typing import Dict

from quantum_twin.core.BaseComponent import BaseComponent


class Styles(BaseComponent):
    """Provides shared style dictionaries."""

    def __init__(self) -> None:
        super().__init__()
        self.logger.info("Styles ready")

    @staticmethod
    def line_style() -> Dict[str, str | float]:
        return {"linewidth": 2.0}

    @staticmethod
    def scatter_style() -> Dict[str, str | float]:
        return {"s": 20, "alpha": 0.8}
