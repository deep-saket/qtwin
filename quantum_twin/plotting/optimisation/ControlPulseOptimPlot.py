from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot


class ControlPulseOptimPlot(BasePlot):
    """Plots control pulse updates across optimization iterations."""

    def render(self, **data: Any) -> plt.Figure:
        history = np.array(data.get("controls_history", [[0, 0, 0]]), dtype=float)
        steps = np.arange(history.shape[0])
        fig, ax = plt.subplots()
        ax.plot(steps, history[:, 0], label="Ωx")
        if history.shape[1] > 1:
            ax.plot(steps, history[:, 1], label="Ωy")
        if history.shape[1] > 2:
            ax.plot(steps, history[:, 2], label="Ωz")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Amplitude")
        ax.set_title("Control Pulse Optimization")
        ax.legend()
        self._figure = fig
        self.logger.info("Rendered ControlPulseOptimPlot with %d steps", history.shape[0])
        return fig
