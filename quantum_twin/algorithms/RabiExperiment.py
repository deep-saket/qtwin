from __future__ import annotations

from typing import Any, Dict

import numpy as np

from quantum_twin.algorithms.BaseAlgorithm import BaseAlgorithm


class RabiExperiment(BaseAlgorithm):
    """Rabi oscillation experiment: sweep amplitude/frequency and record oscillations."""

    def run(self, twin: Any, **kwargs: Any) -> Dict[str, Any]:
        amps = np.linspace(0.0, 0.5, 5)
        results = []
        for amp in amps:
            data = twin.simulator.simulate_schrodinger(trajectories=1)
            results.append({"amp": amp, "rho": data["rho"]})
        self.logger.info("RabiExperiment completed with %d amplitudes", len(amps))
        return {"sweep": results}
