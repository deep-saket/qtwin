from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from quantum_twin.api.QuantumTwinAPI import QuantumTwinAPI
from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.plotting.PlotManager import PlotManager
from quantum_twin.demos.dash_app import launch_dash_async


class BaseDemo(BaseComponent):
    """Base class for demos providing shared API and plotting helpers."""

    def __init__(
        self,
        api_config: str = "quantum_twin/configs/api.yml",
        plotting_config: str = "quantum_twin/configs/plotting.yml",
        launch_dash: bool = False,
        dash_outputs_dir: str = "outputs",
        dash_port: int = 8050,
    ) -> None:
        super().__init__()
        self.twin = QuantumTwinAPI(api_config)
        self.plot_manager = PlotManager(plotting_config)
        self._dash_thread = None
        if launch_dash:
            self._dash_thread = launch_dash_async(outputs_dir=dash_outputs_dir, port=dash_port)
            if self._dash_thread is not None:
                self.logger.info("Dash viewer launched on port %d serving %s", dash_port, dash_outputs_dir)
            else:
                self.logger.warning("Dash not installed; skipping web viewer. Install 'dash' to enable it.")
        self.logger.info("BaseDemo initialized with api=%s plotting=%s", api_config, plotting_config)

    def run_demo(self) -> Dict[str, Any]:
        """Execute the demo workflow."""
        raise NotImplementedError

    def export_demo(self, output_dir: str | Path) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.logger.info("Exported demo artifacts to %s", output_dir)
