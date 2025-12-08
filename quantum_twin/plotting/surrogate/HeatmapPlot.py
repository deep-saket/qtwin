from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot
from quantum_twin.plotting.utils.PlotUtils import PlotUtils


class HeatmapPlot(BasePlot):
    """Plots heatmaps for density matrices or differences."""

    def render(self, **data: Any) -> plt.Figure:
        matrix = PlotUtils.to_numpy(data.get("matrix", np.zeros((2, 2))))
        fig, ax = plt.subplots()
        cax = ax.imshow(matrix, cmap=self._kwargs.get("cmap", "plasma"))
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(self._kwargs.get("title", "Heatmap"))
        self._figure = fig
        self.logger.info("Rendered HeatmapPlot with shape %s", matrix.shape)
        return fig
