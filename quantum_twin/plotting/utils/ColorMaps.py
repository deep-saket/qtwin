from __future__ import annotations

from typing import Dict

from quantum_twin.core.BaseComponent import BaseComponent


class ColorMaps(BaseComponent):
    """Defines named colormaps for reuse."""

    def __init__(self) -> None:
        super().__init__()
        self.logger.info("ColorMaps ready")

    @staticmethod
    def defaults() -> Dict[str, str]:
        return {"density": "viridis", "residual": "magma", "heatmap": "plasma"}
