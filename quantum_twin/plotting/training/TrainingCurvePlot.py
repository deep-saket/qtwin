from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot


class TrainingCurvePlot(BasePlot):
    """Plots training diagnostics (loss, gradient norm)."""

    def render(self, **data: Any) -> plt.Figure:
        loss = np.array(data.get("loss", []), dtype=float)
        grad = np.array(data.get("grad_norm", []), dtype=float)
        steps = np.arange(len(loss))
        fig, ax = plt.subplots()
        ax.plot(steps, loss, label="Loss")
        if grad.size:
            ax.plot(steps[: len(grad)], grad, label="Grad Norm")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_title("Training Curves")
        ax.legend()
        self._figure = fig
        self.logger.info("Rendered TrainingCurvePlot length=%d", len(loss))
        return fig
