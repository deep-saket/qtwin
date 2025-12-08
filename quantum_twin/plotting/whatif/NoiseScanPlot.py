from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot


class NoiseScanPlot(BasePlot):
    """Plots fidelity/error over T1/T2 noise sweeps."""

    def render(self, **data: Any) -> plt.Figure:
        t1 = np.array(data.get("t1", []), dtype=float)
        metric = np.array(data.get("metric", []), dtype=float)
        fig, ax = plt.subplots()
        ax.plot(t1, metric, label="Metric")
        ax.set_xlabel("T1")
        ax.set_ylabel("Metric")
        ax.set_title("Noise Scan")
        ax.legend()
        self._figure = fig
        self.logger.info("Rendered NoiseScanPlot length=%d", len(t1))
        return fig
