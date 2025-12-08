from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot
from quantum_twin.plotting.utils.PlotUtils import PlotUtils


class LossCurvePlot(BasePlot):
    """Plots training and validation loss curves."""

    def render(self, **data: Any) -> plt.Figure:
        train_loss = PlotUtils.to_numpy(data.get("train_loss", np.zeros(1)))
        val_loss = PlotUtils.to_numpy(data.get("val_loss", np.zeros_like(train_loss)))
        steps = np.arange(len(train_loss))
        fig, ax = plt.subplots()
        ax.plot(steps, train_loss, label="Train Loss")
        ax.plot(steps, val_loss, label="Val Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curves")
        ax.legend()
        self._figure = fig
        self.logger.info("Rendered LossCurvePlot with %d points", len(train_loss))
        return fig
