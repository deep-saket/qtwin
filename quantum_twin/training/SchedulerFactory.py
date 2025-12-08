from __future__ import annotations

from typing import Optional

import torch

from quantum_twin.core.BaseComponent import BaseComponent


class SchedulerFactory(BaseComponent):
    """Factory for creating learning rate schedulers."""

    def __init__(self) -> None:
        super().__init__()
        self.logger.info("SchedulerFactory initialized")

    def build(self, optimizer: torch.optim.Optimizer, name: str | None, **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        if not name:
            self.logger.info("No scheduler requested")
            return None
        name = name.lower()
        if name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(kwargs.get("t_max", 50)), eta_min=float(kwargs.get("eta_min", 1e-6))
            )
        elif name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=int(kwargs.get("step_size", 100)), gamma=float(kwargs.get("gamma", 0.5))
            )
        else:
            self.logger.warning("Unknown scheduler %s; skipping", name)
            return None
        self.logger.info("SchedulerFactory built %s", name)
        return scheduler

