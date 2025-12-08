from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from quantum_twin.api.QuantumTwinAPI import QuantumTwinAPI
from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.plotting.PlotManager import PlotManager


class BaseDemo(BaseComponent):
    """Base class for demos providing shared API and plotting helpers."""

    def __init__(self, api_config: str = "quantum_twin/configs/api.yml", plotting_config: str = "quantum_twin/configs/plotting.yml") -> None:
        super().__init__()
        self.twin = QuantumTwinAPI(api_config)
        self.plot_manager = PlotManager(plotting_config)
        self.logger.info("BaseDemo initialized with api=%s plotting=%s", api_config, plotting_config)

    def run_demo(self) -> Dict[str, Any]:
        """Execute the demo workflow."""
        raise NotImplementedError

    def export_demo(self, output_dir: str | Path) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info("Exported demo artifacts to %s", output_dir)
