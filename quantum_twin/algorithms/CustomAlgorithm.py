from __future__ import annotations

from typing import Any, Dict

from quantum_twin.algorithms.BaseAlgorithm import BaseAlgorithm


class CustomAlgorithm(BaseAlgorithm):
    """Base class for user-defined algorithms implemented via sequence()."""

    def sequence(self, twin: Any) -> Dict[str, Any]:
        """User-implemented pulse/operation sequence."""
        raise NotImplementedError

    def run(self, twin: Any, **kwargs: Any) -> Dict[str, Any]:
        result = self.sequence(twin)
        self.logger.info("CustomAlgorithm executed")
        return result
