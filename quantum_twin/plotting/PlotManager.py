from __future__ import annotations

from typing import Any, Dict, List

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.core.ConfigLoader import ConfigLoader
from quantum_twin.core.StepInstantiator import StepInstantiator
from quantum_twin.plotting.PlotFactory import PlotFactory
from quantum_twin.plotting.BasePlot import BasePlot


class PlotManager(BaseComponent):
    """High-level orchestrator for plotting workflows."""

    def __init__(self, config_path: str) -> None:
        super().__init__()
        self._config_path = config_path
        loader = ConfigLoader(config_path)
        self._config = loader.load_yaml()
        self._factory = PlotFactory()
        self.logger.info("PlotManager initialized with %s", config_path)

    def _instantiate_group(self, key: str) -> List[BasePlot]:
        group_cfg = self._config.get(key, [])
        plotters: List[BasePlot] = []
        for cfg in group_cfg:
            plotters.append(self._factory.create(cfg["class"], cfg.get("params", {})))
        self.logger.info("Instantiated %d plots for group %s", len(plotters), key)
        return plotters

    def render_group(self, key: str, **data: Any) -> List[BasePlot]:
        plotters = self._instantiate_group(key)
        for plotter in plotters:
            plotter.render(**data)
            plotter.save()
        return plotters

    def render_all(self, datasets: Dict[str, Dict[str, Any]]) -> Dict[str, List[BasePlot]]:
        """Render all configured plot groups with provided datasets keyed by group name."""
        results: Dict[str, List[BasePlot]] = {}
        for key, data in datasets.items():
            self.logger.info("Rendering group %s", key)
            results[key] = self.render_group(key, **data)
        return results
