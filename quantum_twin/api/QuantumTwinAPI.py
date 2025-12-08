from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from quantum_twin.api.AlgorithmAPI import AlgorithmAPI
from quantum_twin.api.OptimizerAPI import OptimizerAPI
from quantum_twin.api.PulseAPI import PulseAPI
from quantum_twin.api.SimulatorAPI import SimulatorAPI
from quantum_twin.api.SurrogateAPI import SurrogateAPI
from quantum_twin.api.DemoDataAPI import DemoDataAPI
from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.core.ConfigLoader import ConfigLoader
from quantum_twin.core.StepInstantiator import StepInstantiator


class QuantumTwinAPI(BaseComponent):
    """Central orchestrator exposing simulator, surrogate, optimizer, algorithm, and demo data APIs."""

    def __init__(self, config: str = "quantum_twin/configs/api.yml") -> None:
        super().__init__()
        self._config_path = Path(config)
        loader = ConfigLoader(self._config_path)
        self._config = loader.load_yaml()
        self._instantiator = StepInstantiator()
        self.logger.info("QuantumTwinAPI initialized with %s", self._config_path)
        self._init_modules()

    def _init_modules(self) -> None:
        self.simulator = SimulatorAPI(self._config.get("simulator", {}))
        self.surrogate = SurrogateAPI(self._config.get("surrogate", {}))
        self.optimizer = OptimizerAPI(self._config.get("optimizer", {}))
        self.pulse = PulseAPI(self._config.get("pulse", {}))
        self.algorithm = AlgorithmAPI(self._config.get("algorithms", {}), self)
        self.demo_data = DemoDataAPI(self._config.get("demo_data", {}))

    def get_config(self) -> Dict[str, Any]:
        return self._config
