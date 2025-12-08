from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot


class SurrogateFitComparisonPlot(BasePlot):
    """Compares surrogate predictions vs simulator or experimental data."""

    def render(self, **data: Any) -> plt.Figure:
        simulator = np.array(data.get("simulator", np.zeros(1)))
        surrogate = np.array(data.get("surrogate", np.zeros_like(simulator)))
        fig, ax = plt.subplots()
        ax.plot(simulator, label="Simulator")
        ax.plot(surrogate, "--", label="Surrogate")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.set_title("Surrogate Fit Comparison")
        ax.legend()
        self._figure = fig
        self.logger.info("Rendered SurrogateFitComparisonPlot length=%d", len(simulator))
        return fig
