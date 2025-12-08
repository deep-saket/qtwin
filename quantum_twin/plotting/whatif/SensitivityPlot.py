from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot


class SensitivityPlot(BasePlot):
    """Plots sensitivity heatmaps or curves."""

    def render(self, **data: Any) -> plt.Figure:
        sensitivity = np.array(data.get("sensitivity", np.zeros((2, 2))), dtype=float)
        fig, ax = plt.subplots()
        cax = ax.imshow(sensitivity, cmap=self._kwargs.get("cmap", "viridis"))
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Sensitivity Heatmap")
        self._figure = fig
        self.logger.info("Rendered SensitivityPlot shape=%s", sensitivity.shape)
        return fig
