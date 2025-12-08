from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot
from quantum_twin.plotting.utils.PlotUtils import PlotUtils


class EigenvalueTrajectoryPlot(BasePlot):
    """Plots eigenvalues of predicted density matrices."""

    def render(self, **data: Any) -> plt.Figure:
        rho_pred = PlotUtils.to_numpy(data.get("rho_pred", np.zeros((1, 2, 2), dtype=np.complex128)))
        eigvals = np.linalg.eigvals(rho_pred)
        t = PlotUtils.to_numpy(data.get("t", np.arange(eigvals.shape[0])))
        fig, ax = plt.subplots()
        ax.plot(t, np.real(eigvals[:, 0]), label="λ1")
        ax.plot(t, np.real(eigvals[:, 1]), label="λ2")
        ax.set_xlabel("Time")
        ax.set_ylabel("Eigenvalue (real)")
        ax.set_title("Eigenvalue Trajectories")
        ax.legend()
        self._figure = fig
        self.logger.info("Rendered EigenvalueTrajectoryPlot with %d points", eigvals.shape[0])
        return fig
