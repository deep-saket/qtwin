from __future__ import annotations

from typing import Dict, List

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.metrics.FidelityMetric import FidelityMetric
from quantum_twin.metrics.PositivityMetric import PositivityMetric
from quantum_twin.metrics.TraceDistanceMetric import TraceDistanceMetric
from quantum_twin.metrics.MetricBase import MetricBase


class MetricsFactory(BaseComponent):
    """Factory for building configured metrics."""

    def __init__(self, **params: Dict[str, object]) -> None:
        super().__init__()
        self._params = params
        self.logger.info("MetricsFactory initialized with params %s", params)

    def build(self) -> List[MetricBase]:
        metrics: List[MetricBase] = []
        enabled = self._params.get("enabled", ["fidelity", "trace_distance", "positivity"])
        if "fidelity" in enabled:
            metrics.append(FidelityMetric())
        if "trace_distance" in enabled:
            metrics.append(TraceDistanceMetric())
        if "positivity" in enabled:
            metrics.append(PositivityMetric())
        self.logger.info("MetricsFactory built %d metrics", len(metrics))
        return metrics
