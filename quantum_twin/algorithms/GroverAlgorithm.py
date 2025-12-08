from __future__ import annotations

from typing import Any, Dict

import numpy as np

from quantum_twin.algorithms.BaseAlgorithm import BaseAlgorithm


class GroverAlgorithm(BaseAlgorithm):
    """Approximate single-qubit Grover-style phase kick and amplification."""

    def run(self, twin: Any, iterations: int = 1, **kwargs: Any) -> Dict[str, Any]:
        phase_kick = np.exp(1j * np.pi)
        results = []
        for _ in range(iterations):
            data = twin.simulator.simulate_schrodinger(trajectories=1)
            results.append({"rho": data["rho"], "phase": phase_kick})
        self.logger.info("GroverAlgorithm completed iterations=%d", iterations)
        return {"iterations": iterations, "runs": results}
