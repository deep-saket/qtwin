from __future__ import annotations

from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.core.BaseComponent import BaseComponent


class PlotUtils(BaseComponent):
    """Helper utilities for plotting."""

    def __init__(self) -> None:
        super().__init__()
        self.logger.info("PlotUtils ready")

    @staticmethod
    def setup_axes(title: str, xlabel: str, ylabel: str) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return fig, ax

    @staticmethod
    def to_numpy(arr: Any) -> np.ndarray:
        if hasattr(arr, "detach"):
            return arr.detach().cpu().numpy()
        if hasattr(arr, "numpy"):
            return arr.numpy()
        return np.array(arr)
