from __future__ import annotations

from typing import Any, Dict

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.demos.physicist_demo.PhysicistDemo import PhysicistDemo
from quantum_twin.demos.control_demo.ControlCalibrationDemo import ControlCalibrationDemo
from quantum_twin.demos.stakeholder_demo.StakeholderDashboardDemo import StakeholderDashboardDemo
from quantum_twin.demos.algorithm_demo.AlgorithmEndUserDemo import AlgorithmEndUserDemo


class DemoFactory(BaseComponent):
    """Factory for creating demos by persona."""

    def __init__(self) -> None:
        super().__init__()
        self.logger.info("DemoFactory ready")

    def create(self, persona: str, **kwargs: Dict[str, Any]) -> Any:
        persona = persona.lower()
        if persona == "physicist":
            return PhysicistDemo(**kwargs)
        if persona == "control":
            return ControlCalibrationDemo(**kwargs)
        if persona == "stakeholder":
            return StakeholderDashboardDemo(**kwargs)
        if persona == "algorithm":
            return AlgorithmEndUserDemo(**kwargs)
        raise ValueError(f"Unknown persona {persona}")
