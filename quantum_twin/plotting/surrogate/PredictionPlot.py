from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot
from quantum_twin.plotting.utils.PlotUtils import PlotUtils


class PredictionPlot(BasePlot):
    """Plots ground truth vs predicted density matrix entries."""

    def render(self, **data: Any) -> plt.Figure:
        rho_true = PlotUtils.to_numpy(data.get("rho_true", np.zeros((1, 2, 2), dtype=np.complex128)))
        rho_pred = PlotUtils.to_numpy(data.get("rho_pred", np.zeros((1, 2, 2), dtype=np.complex128)))
        t = PlotUtils.to_numpy(data.get("t", np.arange(rho_true.shape[0])))
        fig, ax = plt.subplots()
        ax.plot(t, np.real(rho_true[:, 0, 0]), label="Real ρ00 true")
        ax.plot(t, np.real(rho_pred[:, 0, 0]), "--", label="Real ρ00 pred")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.set_title("Prediction vs Ground Truth")
        ax.legend()
        self._figure = fig
        self.logger.info("Rendered PredictionPlot with %d points", rho_true.shape[0])
        return fig
