from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt

from quantum_twin.core.BaseComponent import BaseComponent


class BasePlot(BaseComponent, ABC):
    """Abstract base class for all plotters in Quantum Twin."""

    def __init__(self, save_path: str | Path | None = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self._save_path = Path(save_path) if save_path else None
        self._figure: plt.Figure | None = None
        self._kwargs = kwargs
        self.logger.info("Initialized %s with save_path=%s", self.__class__.__name__, self._save_path)

    @abstractmethod
    def render(self, **data: Any) -> plt.Figure:
        """Render the plot using provided data."""

    def save(self, path: str | Path | None = None) -> Path | None:
        """Save the rendered plot to disk."""
        if self._figure is None:
            self.logger.error("Attempted to save before rendering: %s", self.__class__.__name__)
            return None
        target = Path(path) if path else self._save_path
        if target is None:
            self.logger.warning("No save path provided for %s; skipping save.", self.__class__.__name__)
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        self._figure.savefig(target)
        self.logger.info("Saved plot %s to %s", self.__class__.__name__, target)
        return target
