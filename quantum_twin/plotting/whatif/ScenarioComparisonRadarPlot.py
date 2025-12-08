from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot


class ScenarioComparisonRadarPlot(BasePlot):
    """Compares scenarios on a radar chart."""

    def render(self, **data: Any) -> plt.Figure:
        labels: List[str] = data.get("labels", [])
        scenarios = np.array(data.get("scenarios", []), dtype=float)
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(subplot_kw={"polar": True})
        for scenario in scenarios:
            values = scenario.tolist()
            values += values[:1]
            ax.plot(angles, values)
            ax.fill(angles, values, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title("Scenario Comparison")
        self._figure = fig
        self.logger.info("Rendered ScenarioComparisonRadarPlot scenarios=%d", scenarios.shape[0] if scenarios.size else 0)
        return fig
