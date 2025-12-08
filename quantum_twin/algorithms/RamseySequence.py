from __future__ import annotations

from typing import Any, Dict

import numpy as np

from quantum_twin.algorithms.BaseAlgorithm import BaseAlgorithm


class RamseySequence(BaseAlgorithm):
    """Ramsey sequence: π/2 - free evolution - π/2."""

    def run(self, twin: Any, **kwargs: Any) -> Dict[str, Any]:
        detunings = np.linspace(-0.1, 0.1, 5)
        results = []
        for det in detunings:
            data = twin.simulator.simulate_schrodinger(trajectories=1)
            results.append({"detuning": det, "rho": data["rho"]})
        self.logger.info("RamseySequence completed with %d detunings", len(detunings))
        return {"sweep": results}
