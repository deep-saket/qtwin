from __future__ import annotations

from typing import Any, Dict

from quantum_twin.core.BaseComponent import BaseComponent


class TrainingParams(BaseComponent):
    """Container for training hyperparameters."""

    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params: Dict[str, Any] = params
        self.logger.info("TrainingParams initialized %s", params)

