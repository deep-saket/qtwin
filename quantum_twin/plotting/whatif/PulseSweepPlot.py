from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot


class PulseSweepPlot(BasePlot):
    """Plots impact of pulse sweeps on dynamics."""

    def render(self, **data: Any) -> plt.Figure:
        amplitude = np.array(data.get("amplitude", []), dtype=float)
        metric = np.array(data.get("metric", []), dtype=float)
        fig, ax = plt.subplots()
        ax.plot(amplitude, metric, label="Outcome")
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Metric")
        ax.set_title("Pulse Sweep")
        ax.legend()
        self._figure = fig
        self.logger.info("Rendered PulseSweepPlot length=%d", len(amplitude))
        return fig
