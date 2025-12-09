from __future__ import annotations

from typing import Any, Dict

import torch

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.deployment.ModelWrapper import ModelWrapper
from quantum_twin.models.ModelFactory import ModelFactory


class SurrogateAPI(BaseComponent):
    """API wrapper for surrogate model inference (PyTorch + ONNX)."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self._params = params
        self._onnx_path = params.get("onnx_path", "artifacts/pinn.onnx")
        self._model_wrapper: ModelWrapper | None = None
        try:
            self._model_wrapper = ModelWrapper(self._onnx_path)
            self.logger.info("SurrogateAPI ready onnx=%s", self._onnx_path)
        except FileNotFoundError:
            self.logger.warning("ONNX model not found at %s; surrogate calls will be skipped.", self._onnx_path)
        except Exception as exc:  # pragma: no cover
            self.logger.warning("Failed to load ONNX model %s (%s); surrogate calls disabled.", self._onnx_path, exc)

    def predict(self, t: float, controls: list[float]) -> Any:
        if self._model_wrapper is None:
            self.logger.error("Surrogate not available; run export.py to generate ONNX.")
            return None
        return self._model_wrapper.run(t=t, controls=controls)

    def predict_trajectory(self, t: list[float], controls: list[list[float]]) -> Any:
        if self._model_wrapper is None:
            self.logger.error("Surrogate not available; run export.py to generate ONNX.")
            return None
        return self._model_wrapper._server.predict({"t": t, "controls": controls})

    def compare_with_simulator(self, simulator_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if self._model_wrapper is None:
            self.logger.error("Surrogate not available; run export.py to generate ONNX.")
            return {"simulator": simulator_data["rho"], "surrogate": torch.zeros_like(simulator_data["rho"])}
        t = simulator_data["t"]
        controls = simulator_data["controls"]
        pred = self._model_wrapper._server.predict({"t": t.numpy(), "controls": controls.numpy()})
        return {"simulator": simulator_data["rho"], "surrogate": torch.tensor(pred)}
