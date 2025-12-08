from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from quantum_twin.plotting.BasePlot import BasePlot


class ONNXBenchmarksPlot(BasePlot):
    """Plots ONNX latency and throughput benchmarks."""

    def render(self, **data: Any) -> plt.Figure:
        latency = np.array(data.get("latency_ms", []), dtype=float)
        throughput = np.array(data.get("throughput", []), dtype=float)
        fig, ax1 = plt.subplots()
        ax1.plot(latency, label="Latency (ms)", color="tab:blue")
        ax1.set_xlabel("Batch")
        ax1.set_ylabel("Latency (ms)", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(throughput, label="Throughput", color="tab:red")
        ax2.set_ylabel("Throughput", color="tab:red")
        fig.suptitle("ONNX Benchmarks")
        self._figure = fig
        self.logger.info("Rendered ONNXBenchmarksPlot length=%d", len(latency))
        return fig
