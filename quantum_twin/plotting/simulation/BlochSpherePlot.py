from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot
from quantum_twin.plotting.utils.BlochSphereUtils import BlochSphereUtils


class BlochSpherePlot(BasePlot):
    """Plots Bloch vector trajectories."""

    def render(self, **data: Any) -> plt.Figure:
        rho = np.array(data.get("rho", np.zeros((1, 2, 2), dtype=np.complex128)))
        vectors = np.array([BlochSphereUtils.density_to_bloch(r) for r in rho])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(vectors[:, 0], vectors[:, 1], vectors[:, 2], label="Bloch trajectory")
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_title("Bloch Sphere Trajectory")
        ax.legend()
        self._figure = fig
        self.logger.info("Rendered BlochSpherePlot with %d points", vectors.shape[0])
        return fig
