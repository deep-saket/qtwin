from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.core.Constants import DEFAULT_CHECKPOINT_DIR


class CheckpointManager(BaseComponent):
    """Handles saving and loading model checkpoints."""

    def __init__(self, directory: str = DEFAULT_CHECKPOINT_DIR, keep_last: int = 3) -> None:
        super().__init__()
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._keep_last = keep_last
        self.logger.info("CheckpointManager using %s keep_last=%d", self._dir, keep_last)

    def save(self, step: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> Path:
        path = self._dir / f"step_{step}.pt"
        torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
        self.logger.info("Saved checkpoint %s", path)
        self._prune()
        return path

    def load(self, path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None) -> int:
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if optimizer and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))
        self.logger.info("Loaded checkpoint %s at step %d", path, step)
        return step

    def _prune(self) -> None:
        checkpoints = sorted(self._dir.glob("step_*.pt"))
        if len(checkpoints) > self._keep_last:
            for old in checkpoints[: -self._keep_last]:
                try:
                    old.unlink()
                    self.logger.info("Pruned checkpoint %s", old)
                except OSError as exc:
                    self.logger.error("Failed to prune %s: %s", old, exc)

