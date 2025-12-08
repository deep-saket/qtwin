from __future__ import annotations

from typing import Any, Dict

from quantum_twin.demos.BaseDemo import BaseDemo


class PhysicistDemo(BaseDemo):
    """Demo for physicists: simulation, decoherence, parameter sweeps, residuals."""

    def run_demo(self) -> Dict[str, Any]:
        data = self.twin.simulator.simulate_lindblad(trajectories=2)
        self.plot_manager.render_group(
            "SIMULATION_PLOTS",
            rho=data["rho"].cpu().numpy(),
            t=data["t"].cpu().numpy(),
            controls=data["controls"].cpu().numpy(),
        )
        residuals = {"residuals": data["rho"].real}
        self.plot_manager.render_group("SURROGATE_PLOTS", **residuals)
        self.logger.info("PhysicistDemo completed")
        return {"simulation": data}


if __name__ == "__main__":
    demo = PhysicistDemo()
    demo.run_demo()
