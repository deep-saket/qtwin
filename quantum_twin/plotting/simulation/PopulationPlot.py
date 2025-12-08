from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot
from quantum_twin.plotting.utils.PlotUtils import PlotUtils


class PopulationPlot(BasePlot):
    """Plots ground/excited state populations over time."""

    def render(self, **data: Any) -> plt.Figure:
        rho = PlotUtils.to_numpy(data.get("rho", np.zeros((1, 2, 2), dtype=np.complex128)))
        t = PlotUtils.to_numpy(data.get("t", np.arange(rho.shape[0])))
        p0 = np.real(rho[:, 0, 0])
        p1 = np.real(rho[:, 1, 1])
        fig, ax = plt.subplots()
        ax.plot(t, p0, label="P(|0>)")
        ax.plot(t, p1, label="P(|1>)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Population")
        ax.set_ylim([0, 1])
        ax.legend()
        ax.set_title("Population Trajectories")
        self._figure = fig
        self.logger.info("Rendered PopulationPlot with %d points", rho.shape[0])
        return fig
