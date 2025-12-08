from __future__ import annotations

from typing import Any, Dict

from quantum_twin.demos.BaseDemo import BaseDemo


class ControlCalibrationDemo(BaseDemo):
    """Demo for control/calibration engineers."""

    def run_demo(self) -> Dict[str, Any]:
        optimized = self.twin.optimizer.optimize_control(strategy="gradient")
        sim_data = self.twin.simulator.simulate_lindblad(trajectories=1)
        self.plot_manager.render_group("OPTIMISATION_PLOTS", controls_history=[optimized])
        self.plot_manager.render_group(
            "SURROGATE_PLOTS",
            rho_true=sim_data["rho"].cpu().numpy(),
            rho_pred=sim_data["rho"].cpu().numpy(),
            t=sim_data["t"].cpu().numpy(),
        )
        self.logger.info("ControlCalibrationDemo completed")
        return {"optimized_controls": optimized, "simulation": sim_data}


if __name__ == "__main__":
    demo = ControlCalibrationDemo()
    demo.run_demo()
