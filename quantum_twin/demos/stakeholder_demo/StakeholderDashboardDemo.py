from __future__ import annotations

from typing import Any, Dict

from quantum_twin.demos.BaseDemo import BaseDemo


class StakeholderDashboardDemo(BaseDemo):
    """Dashboard-style summary for stakeholders."""

    def run_demo(self) -> Dict[str, Any]:
        sim_data = self.twin.simulator.simulate_lindblad(trajectories=1)
        metrics = {"metric": [0.99, 0.97, 0.98], "t1": [20, 30, 40]}
        self.plot_manager.render_group("WHATIF_PLOTS", **metrics)
        self.logger.info("StakeholderDashboardDemo completed")
        return {"simulation": sim_data, "metrics": metrics}
