from __future__ import annotations

from typing import Any, Dict, List

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.optimisation.ParameterEstimator import ParameterEstimator
from quantum_twin.optimisation.UniformControlOptimizer import UniformControlOptimizer
from quantum_twin.optimisation.GaussianControlOptimizer import GaussianControlOptimizer
from quantum_twin.optimisation.GradientControlOptimizer import GradientControlOptimizer
from quantum_twin.optimisation.SurrogateFitter import SurrogateFitter


class OptimizerAPI(BaseComponent):
    """API for parameter estimation, control optimization, and surrogate fitting."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self._params = params
        self.logger.info("OptimizerAPI initialized %s", params)

    def estimate_parameters(self, data: Dict[str, Any]) -> Dict[str, float]:
        estimator = ParameterEstimator(self._params.get("parameter_estimator", {}))
        return estimator.run()

    def optimize_control(self, target_state: str | None = None, strategy: str = "gradient") -> List[float]:
        strategy = strategy.lower()
        if strategy == "uniform":
            optimizer = UniformControlOptimizer(self._params.get("control_optimizer", {}))
        elif strategy == "gaussian":
            optimizer = GaussianControlOptimizer(self._params.get("control_optimizer", {}))
        else:
            optimizer = GradientControlOptimizer(self._params.get("control_optimizer", {}))
        return optimizer.run()

    def fit_surrogate(self, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        fitter = SurrogateFitter(params or self._params.get("surrogate_fitter", {}))
        return fitter.run()
