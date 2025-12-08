from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch

from quantum_twin.core.BaseComponent import BaseComponent


class PINNExporter(BaseComponent):
    """Exports PINN models to ONNX."""

    def __init__(self, export_path: str, opset: int = 17) -> None:
        super().__init__()
        self._export_path = Path(export_path)
        self._opset = opset
        self.logger.info("PINNExporter configured path=%s opset=%d", self._export_path, opset)

    def export(self, model: torch.nn.Module, sample_input: Dict[str, torch.Tensor]) -> Path:
        self._export_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info("Exporting ONNX model to %s", self._export_path)
        model.eval()
        with torch.no_grad():
            torch.onnx.export(
                model,
                (sample_input["t"], sample_input["controls"]),
                self._export_path.as_posix(),
                input_names=["t", "controls"],
                output_names=["rho"],
                opset_version=self._opset,
                dynamic_axes={"t": {0: "batch"}, "controls": {0: "batch"}, "rho": {0: "batch"}},
            )
        self.logger.info("ONNX export finished: %s", self._export_path)
        return self._export_path

