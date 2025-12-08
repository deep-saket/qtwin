from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot
from quantum_twin.plotting.utils.PlotUtils import PlotUtils


class DensityMatrixPlot(BasePlot):
    """Plots real and imaginary parts of density matrices over time."""

    def render(self, **data: Any) -> plt.Figure:
        rho = PlotUtils.to_numpy(data.get("rho", np.zeros((1, 2, 2), dtype=np.complex128)))
        t = PlotUtils.to_numpy(data.get("t", np.arange(rho.shape[0])))
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(np.real(rho[-1]), cmap=self._kwargs.get("cmap", "viridis"))
        axes[0].set_title("Real(ρ) at final t")
        axes[1].imshow(np.imag(rho[-1]), cmap=self._kwargs.get("cmap", "magma"))
        axes[1].set_title("Imag(ρ) at final t")
        fig.suptitle("Density Matrix Snapshot")
        self._figure = fig
        self.logger.info("Rendered DensityMatrixPlot with %d time steps", rho.shape[0])
        return fig
