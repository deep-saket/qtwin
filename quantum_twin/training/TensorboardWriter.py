from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter

from quantum_twin.core.BaseComponent import BaseComponent


class TensorboardWriter(BaseComponent):
    """TensorBoard logger for training diagnostics."""

    def __init__(self, log_dir: str) -> None:
        super().__init__()
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(self._log_dir.as_posix())
        self.logger.info("TensorBoard logging to %s", self._log_dir)

    def log_scalars(self, step: int, scalars: Dict[str, float]) -> None:
        for name, value in scalars.items():
            self._writer.add_scalar(name, value, step)
            self.logger.debug("TensorBoard scalar %s=%.6f at step %d", name, value, step)

    def log_histogram(self, step: int, name: str, values: torch.Tensor) -> None:
        self._writer.add_histogram(name, values, step)
        self.logger.debug("TensorBoard histogram %s at step %d", name, step)

    def log_image(self, step: int, name: str, image: torch.Tensor) -> None:
        self._writer.add_image(name, image, step)
        self.logger.debug("TensorBoard image %s at step %d", name, step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()
        self.logger.info("TensorBoard writer closed")

