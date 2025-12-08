from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot


class DriftOverServingPlot(BasePlot):
    """Plots error drift over repeated ONNX inference calls."""

    def render(self, **data: Any) -> plt.Figure:
        drift = np.array(data.get("drift", []), dtype=float)
        calls = np.arange(len(drift))
        fig, ax = plt.subplots()
        ax.plot(calls, drift, label="Error Drift")
        ax.set_xlabel("Call")
        ax.set_ylabel("Drift")
        ax.set_title("Drift Over Serving")
        ax.legend()
        self._figure = fig
        self.logger.info("Rendered DriftOverServingPlot length=%d", len(drift))
        return fig
