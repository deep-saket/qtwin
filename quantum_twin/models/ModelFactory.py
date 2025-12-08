from __future__ import annotations

from typing import Any, Dict, List

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.models.PINNModel import PINNModel


class ModelFactory(BaseComponent):
    """Factory for constructing models from inline params."""

    def __init__(self, **params: Dict[str, Any]) -> None:
        super().__init__()
        self._params = params
        self.logger.info("ModelFactory initialized with params %s", params)

    def build(self) -> PINNModel:
        layers: List[int] = self._params.get("layers", [4, 128, 128, 4])
        activation = self._params.get("activation", "tanh")
        dropout = float(self._params.get("dropout", 0.0))
        fourier = int(self._params.get("fourier_features", 0))
        input_dim = int(self._params.get("input_dim", 4))
        model = PINNModel(
            input_dim=input_dim,
            layers=layers,
            activation=activation,
            dropout=dropout,
            fourier_features=fourier,
        )
        self.logger.info("Model constructed via ModelFactory")
        return model
