from __future__ import annotations

import importlib
from typing import Any, Dict, List

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.plotting.BasePlot import BasePlot


class PlotFactory(BaseComponent):
    """Factory that dynamically instantiates plot classes."""

    def __init__(self) -> None:
        super().__init__()
        self.logger.info("PlotFactory ready")

    def create(self, class_path: str, params: Dict[str, Any]) -> BasePlot:
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        plotter: BasePlot = cls(**params)
        self.logger.info("Created plotter %s with params=%s", class_path, params)
        return plotter

    def create_many(self, configs: List[Dict[str, Any]]) -> List[BasePlot]:
        plotters: List[BasePlot] = []
        for cfg in configs:
            plotters.append(self.create(cfg["class"], cfg.get("params", {})))
        return plotters
