from __future__ import annotations

import importlib
from typing import Any, Dict

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.core.ConfigLoader import ConfigStep


class StepInstantiator(BaseComponent):
    """Instantiates classes from ConfigStep metadata without storing class objects in configs."""

    def __init__(self) -> None:
        super().__init__()
        self.logger.info("StepInstantiator ready")

    def instantiate(self, step: ConfigStep) -> Any:
        module_name, class_name = step.class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        instance = cls(**step.params)
        self.logger.info("Instantiated %s for step %s", step.class_path, step.name)
        return instance

    def instantiate_all(self, steps: Dict[str, ConfigStep]) -> Dict[str, Any]:
        return {name: self.instantiate(step) for name, step in steps.items()}
