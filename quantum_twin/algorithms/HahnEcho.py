from __future__ import annotations

from typing import Any, Dict

import numpy as np

from quantum_twin.algorithms.BaseAlgorithm import BaseAlgorithm


class HahnEcho(BaseAlgorithm):
    """Hahn echo: π/2 - wait - π - wait - π/2."""

    def run(self, twin: Any, **kwargs: Any) -> Dict[str, Any]:
        wait_times = np.linspace(0.0, 0.5, 5)
        results = []
        for wait in wait_times:
            data = twin.simulator.simulate_lindblad(trajectories=1)
            results.append({"wait": wait, "rho": data["rho"]})
        self.logger.info("HahnEcho completed with %d waits", len(wait_times))
        return {"sweep": results}
