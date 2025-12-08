from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnxruntime as ort

from quantum_twin.core.BaseComponent import BaseComponent


class ONNXRuntimeServer(BaseComponent):
    """Runs ONNX models for fast inference."""

    def __init__(self, model_path: str) -> None:
        super().__init__()
        self._model_path = Path(model_path)
        self._session = ort.InferenceSession(self._model_path.as_posix(), providers=["CPUExecutionProvider"])
        self.logger.info("ONNXRuntimeServer loaded model %s", self._model_path)

    def predict(self, inputs: Dict[str, Any]) -> np.ndarray:
        t = np.array(inputs["t"], dtype=np.float64)
        controls = np.array(inputs["controls"], dtype=np.float64)
        self.logger.debug("Predict called t shape=%s controls shape=%s", t.shape, controls.shape)
        outputs = self._session.run(None, {"t": t, "controls": controls})
        return outputs[0]

