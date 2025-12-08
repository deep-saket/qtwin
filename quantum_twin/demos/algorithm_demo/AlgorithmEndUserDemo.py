from __future__ import annotations

from typing import Any, Dict

from quantum_twin.demos.BaseDemo import BaseDemo


class AlgorithmEndUserDemo(BaseDemo):
    """Demo for running built-in and custom algorithms."""

    def run_demo(self) -> Dict[str, Any]:
        grover_result = self.twin.algorithm.run_builtin("grover", iterations=1)
        sim_data = self.twin.simulator.simulate_schrodinger(trajectories=1)
        self.plot_manager.render_group(
            "SIMULATION_PLOTS",
            rho=sim_data["rho"].cpu().numpy(),
            t=sim_data["t"].cpu().numpy(),
            controls=sim_data["controls"].cpu().numpy(),
        )
        self.logger.info("AlgorithmEndUserDemo completed")
        return {"grover": grover_result, "simulation": sim_data}
