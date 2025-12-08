from __future__ import annotations

import argparse

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.core.ConfigLoader import ConfigLoader, ConfigStep
from quantum_twin.core.StepInstantiator import StepInstantiator
from quantum_twin.metrics.MetricsFactory import MetricsFactory


class SimulationApplication(BaseComponent):
    """Runs simulation-only pipeline."""

    def __init__(self, config_path: str) -> None:
        super().__init__()
        self._config_path = config_path
        loader = ConfigLoader(config_path)
        raw = loader.load_yaml()
        steps = loader.parse_steps(raw)
        self._steps = {step.name: step for step in steps}
        self._instantiator = StepInstantiator()
        self.logger.info("SimulationApplication loaded %s", config_path)

    def run(self) -> None:
        simulator = self._instantiator.instantiate(self._steps["simulator"])
        export_params = self._steps["export"].params
        trajectories = int(export_params.get("trajectories", 4))
        use_lindblad = bool(export_params.get("use_lindblad", True))
        data = simulator.generate_dataset(trajectories=trajectories, use_lindblad=use_lindblad)

        metrics_factory = self._instantiator.instantiate(self._steps["metrics"])
        metrics = metrics_factory.build()
        for metric in metrics:
            values = metric(data["rho"], data["rho"])
            self.logger.info("Simulation metric %s", values)
        self.logger.info("Simulation completed with %d samples", len(data["t"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Twin Simulation")
    parser.add_argument("--config", type=str, default="quantum_twin/configs/simulation.yml", help="Simulation config path")
    args = parser.parse_args()
    SimulationApplication(args.config).run()
