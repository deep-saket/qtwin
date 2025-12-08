from __future__ import annotations

from typing import List

import numpy as np

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.deployment.ONNXRuntimeServer import ONNXRuntimeServer


class ModelWrapper(BaseComponent):
    """Thin wrapper over ONNXRuntimeServer for convenient inference."""

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self._server = ONNXRuntimeServer(model_path)
        self.logger.info("ModelWrapper ready for %s", model_path)

    def run(self, t: float, controls: List[float]) -> np.ndarray:
        t_arr = np.array([[t]], dtype=np.float64)
        ctrl_arr = np.array([controls], dtype=np.float64)
        result = self._server.predict({"t": t_arr, "controls": ctrl_arr})
        self.logger.debug("ModelWrapper result shape=%s", result.shape)
        return result

