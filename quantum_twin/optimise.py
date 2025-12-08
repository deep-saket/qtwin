from __future__ import annotations

import argparse

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.core.ConfigLoader import ConfigLoader
from quantum_twin.core.StepInstantiator import StepInstantiator


class OptimisationApplication(BaseComponent):
    """Runs optimisation pipeline (parameters, controls, surrogate)."""

    def __init__(self, config_path: str) -> None:
        super().__init__()
        loader = ConfigLoader(config_path)
        raw = loader.load_yaml()
        steps = loader.parse_steps(raw)
        self._steps = {step.name: step for step in steps}
        self._instantiator = StepInstantiator()
        self.logger.info("OptimisationApplication loaded %s", config_path)

    def run(self) -> None:
        param_estimator = self._instantiator.instantiate(self._steps["parameter_estimator"])
        control_opt = self._instantiator.instantiate(self._steps["control_optimizer"])
        surrogate = self._instantiator.instantiate(self._steps["surrogate_fitter"])

        params = param_estimator.run()
        controls = control_opt.run()
        surrogate_result = surrogate.run()

        self.logger.info("Optimisation results params=%s controls=%s surrogate=%s", params, controls, surrogate_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Twin Optimisation")
    parser.add_argument("--config", type=str, default="quantum_twin/configs/optimiser.yml", help="Optimiser config path")
    args = parser.parse_args()
    OptimisationApplication(args.config).run()
