from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot
from quantum_twin.plotting.utils.PlotUtils import PlotUtils


class PulsePlot(BasePlot):
    """Plots control pulses Ωx, Ωy over time."""

    def render(self, **data: Any) -> plt.Figure:
        controls = PlotUtils.to_numpy(data.get("controls", np.zeros((1, 3))))
        t = PlotUtils.to_numpy(data.get("t", np.arange(controls.shape[0])))
        fig, ax = plt.subplots()
        ax.plot(t, controls[:, 0], label="Ωx")
        if controls.shape[1] > 1:
            ax.plot(t, controls[:, 1], label="Ωy")
        if controls.shape[1] > 2:
            ax.plot(t, controls[:, 2], label="Ωz")
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.set_title("Control Pulses")
        ax.legend()
        self._figure = fig
        self.logger.info("Rendered PulsePlot with %d points", controls.shape[0])
        return fig
