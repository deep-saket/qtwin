from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot
from quantum_twin.plotting.utils.PlotUtils import PlotUtils


class ResidualPlot(BasePlot):
    """Plots physics residual heatmaps."""

    def render(self, **data: Any) -> plt.Figure:
        residuals = PlotUtils.to_numpy(data.get("residuals", np.zeros((1, 2, 2)))))
        fig, ax = plt.subplots()
        cax = ax.imshow(residuals.squeeze(), cmap=self._kwargs.get("cmap", "magma"))
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Physics Residual Heatmap")
        self._figure = fig
        self.logger.info("Rendered ResidualPlot with shape %s", residuals.shape)
        return fig
