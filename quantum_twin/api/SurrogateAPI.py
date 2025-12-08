from __future__ import annotations

from typing import Any, Dict

import torch

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.deployment.ModelWrapper import ModelWrapper
from quantum_twin.model.PINNModel import PINNModel
from quantum_twin.models.ModelFactory import ModelFactory


class SurrogateAPI(BaseComponent):
    """API wrapper for surrogate model inference (PyTorch + ONNX)."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self._params = params
        self._onnx_path = params.get("onnx_path", "artifacts/pinn.onnx")
        self._model_wrapper = ModelWrapper(self._onnx_path)
        self.logger.info("SurrogateAPI ready onnx=%s", self._onnx_path)

    def predict(self, t: float, controls: list[float]) -> Any:
        return self._model_wrapper.run(t=t, controls=controls)

    def predict_trajectory(self, t: list[float], controls: list[list[float]]) -> Any:
        return self._model_wrapper._server.predict({"t": t, "controls": controls})

    def compare_with_simulator(self, simulator_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        t = simulator_data["t"]
        controls = simulator_data["controls"]
        pred = self._model_wrapper._server.predict({"t": t.numpy(), "controls": controls.numpy()})
        return {"simulator": simulator_data["rho"], "surrogate": torch.tensor(pred)}
