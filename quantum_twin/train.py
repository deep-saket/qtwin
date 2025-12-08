from __future__ import annotations

import argparse
from typing import Dict

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.core.ConfigLoader import ConfigLoader, ConfigStep
from quantum_twin.core.StepInstantiator import StepInstantiator
from quantum_twin.training.Trainer import Trainer


def run_training(config_path: str) -> None:
    """Entry function to launch training using step-based config."""
    app = TrainingApplication(config_path)
    app.run()


class TrainingApplication(BaseComponent):
    """Loads configuration steps and orchestrates training."""

    def __init__(self, config_path: str) -> None:
        super().__init__()
        self._config_path = config_path
        self.logger.info("TrainingApplication initialized with %s", config_path)
        loader = ConfigLoader(config_path)
        raw = loader.load_yaml()
        steps = loader.parse_steps(raw)
        self._steps: Dict[str, ConfigStep] = {step.name: step for step in steps}
        self._instantiator = StepInstantiator()

    def _build_components(self) -> Dict[str, object]:
        dataloader = self._instantiator.instantiate(self._steps["dataloader"])
        model_factory = self._instantiator.instantiate(self._steps["model"])
        loss_factory = self._instantiator.instantiate(self._steps["loss"])
        metrics_factory = self._instantiator.instantiate(self._steps["metrics"])
        training_params_step = self._steps["training_params"].params

        model = model_factory.build()
        loss_fn = loss_factory.build()
        metrics = metrics_factory.build()

        return {
            "model": model,
            "loss_fn": loss_fn,
            "loader": dataloader.loader(),
            "metrics": metrics,
            "training_params": training_params_step,
        }

    def run(self) -> None:
        components = self._build_components()
        trainer = Trainer(
            model=components["model"],
            loss_fn=components["loss_fn"],
            dataloader=components["loader"],
            metrics=components["metrics"],
            training_params=components["training_params"],
        )
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Twin Training")
    parser.add_argument("--config", type=str, default="quantum_twin/configs/training.yml", help="Training config path")
    args = parser.parse_args()
    run_training(args.config)
