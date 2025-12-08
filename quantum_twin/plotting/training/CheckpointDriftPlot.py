from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot


class CheckpointDriftPlot(BasePlot):
    """Compares model outputs across checkpoints."""

    def render(self, **data: Any) -> plt.Figure:
        drift = np.array(data.get("drift", []), dtype=float)
        checkpoints = np.arange(len(drift))
        fig, ax = plt.subplots()
        ax.bar(checkpoints, drift)
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Drift")
        ax.set_title("Checkpoint Drift")
        self._figure = fig
        self.logger.info("Rendered CheckpointDriftPlot length=%d", len(drift))
        return fig
