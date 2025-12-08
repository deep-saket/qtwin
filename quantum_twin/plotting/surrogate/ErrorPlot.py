from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot
from quantum_twin.plotting.utils.PlotUtils import PlotUtils


class ErrorPlot(BasePlot):
    """Plots error metrics over time."""

    def render(self, **data: Any) -> plt.Figure:
        errors = PlotUtils.to_numpy(data.get("errors", np.zeros(1)))
        t = PlotUtils.to_numpy(data.get("t", np.arange(errors.shape[0])))
        fig, ax = plt.subplots()
        ax.plot(t, errors, label="Error")
        ax.set_xlabel("Time")
        ax.set_ylabel("Error")
        ax.set_title("Error Trajectory")
        ax.legend()
        self._figure = fig
        self.logger.info("Rendered ErrorPlot with %d points", errors.shape[0])
        return fig
