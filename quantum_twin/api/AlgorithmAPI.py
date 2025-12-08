from __future__ import annotations

from typing import Any, Dict, List

from quantum_twin.algorithms.BaseAlgorithm import BaseAlgorithm
from quantum_twin.algorithms.RabiExperiment import RabiExperiment
from quantum_twin.algorithms.RamseySequence import RamseySequence
from quantum_twin.algorithms.HahnEcho import HahnEcho
from quantum_twin.algorithms.GroverAlgorithm import GroverAlgorithm
from quantum_twin.algorithms.CustomAlgorithm import CustomAlgorithm
from quantum_twin.core.BaseComponent import BaseComponent


class AlgorithmAPI(BaseComponent):
    """Runs built-in and custom quantum algorithms using simulator/surrogate/pulse APIs."""

    def __init__(self, params: Dict[str, Any], twin: Any) -> None:
        super().__init__()
        self._params = params
        self._twin = twin
        self._builtins: Dict[str, BaseAlgorithm] = {
            "rabi": RabiExperiment(),
            "ramsey": RamseySequence(),
            "hahn": HahnEcho(),
            "grover": GroverAlgorithm(),
        }
        self.logger.info("AlgorithmAPI initialized with built-ins: %s", list(self._builtins))

    def list_algorithms(self) -> List[str]:
        return list(self._builtins.keys())

    def run_builtin(self, name: str, **kwargs: Any) -> Any:
        algo = self._builtins.get(name.lower())
        if algo is None:
            raise ValueError(f"Algorithm {name} not registered")
        return algo.run(self._twin, **kwargs)

    def run_custom(self, custom: CustomAlgorithm, **kwargs: Any) -> Any:
        return custom.run(self._twin, **kwargs)
