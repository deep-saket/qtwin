from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot


class ParameterEstimationPlot(BasePlot):
    """Plots convergence of parameter estimation."""

    def render(self, **data: Any) -> plt.Figure:
        estimates = np.array(data.get("estimates", []), dtype=float)
        true_params = np.array(data.get("true_params", []), dtype=float)
        steps = np.arange(len(estimates))
        fig, ax = plt.subplots()
        ax.plot(steps, estimates, label="Estimate")
        if true_params.size > 0:
            ax.hlines(true_params, xmin=0, xmax=len(estimates), colors="red", linestyles="--", label="True")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Parameter")
        ax.set_title("Parameter Estimation")
        ax.legend()
        self._figure = fig
        self.logger.info("Rendered ParameterEstimationPlot with %d steps", len(estimates))
        return fig
