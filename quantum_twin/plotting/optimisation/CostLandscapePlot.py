from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot


class CostLandscapePlot(BasePlot):
    """Visualizes cost landscape in 2D."""

    def render(self, **data: Any) -> plt.Figure:
        x = np.array(data.get("x", np.linspace(-1, 1, 20)))
        y = np.array(data.get("y", np.linspace(-1, 1, 20)))
        z = np.array(data.get("z", np.zeros((len(x), len(y)))))
        fig, ax = plt.subplots()
        cax = ax.contourf(x, y, z, levels=20, cmap=self._kwargs.get("cmap", "viridis"))
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("Param 1")
        ax.set_ylabel("Param 2")
        ax.set_title("Cost Landscape")
        self._figure = fig
        self.logger.info("Rendered CostLandscapePlot with grids %d x %d", len(x), len(y))
        return fig
