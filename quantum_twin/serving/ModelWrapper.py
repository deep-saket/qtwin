from __future__ import annotations

import numpy as np
from typing import List

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.serving.ONNXRuntimeServer import ONNXRuntimeServer


class ModelWrapper(BaseComponent):
    """High-level interface for ONNXRuntime inference."""

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self._server = ONNXRuntimeServer(model_path)
        self.logger.info("ModelWrapper ready for %s", model_path)

    def run(self, t: float, controls: List[float]) -> np.ndarray:
        t_arr = np.array([[t]], dtype=np.float64)
        ctrl_arr = np.array([controls], dtype=np.float64)
        result = self._server.predict({"t": t_arr, "controls": ctrl_arr})
        self.logger.debug("Inference result shape=%s", result.shape)
        return result

