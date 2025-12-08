from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from quantum_twin.core.BaseComponent import BaseComponent


@dataclass(frozen=True)
class ConfigStep:
    """Configuration step describing a component to instantiate."""

    name: str
    class_path: str
    params: Dict[str, Any]


class ConfigLoader(BaseComponent):
    """Loads YAML configuration files and exposes step metadata."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self._path = Path(path)
        self.logger.info("ConfigLoader initialized with %s", self._path)

    def load_yaml(self) -> Dict[str, Any]:
        """Load raw YAML content."""
        with self._path.open("r", encoding="utf-8") as handle:
            data: Dict[str, Any] = yaml.safe_load(handle) or {}
        self.logger.info("Loaded config keys: %s", list(data.keys()))
        return data

    def parse_steps(self, data: Dict[str, Any]) -> Tuple[ConfigStep, ...]:
        """Convert YAML mapping to ordered config steps (class path stays string)."""
        steps = []
        for name, cfg in data.items():
            class_path = cfg.get("class", "")
            params = cfg.get("params", {}) or {}
            step = ConfigStep(name=name, class_path=class_path, params=params)
            steps.append(step)
            self.logger.debug("Parsed step %s -> %s", name, class_path)
        return tuple(steps)
