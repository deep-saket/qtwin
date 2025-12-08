from __future__ import annotations

from typing import Any, Dict

import torch

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.physics.DataSimulator import DataSimulator


class SimulatorAPI(BaseComponent):
    """API wrapper for physics solvers."""

    def __init__(self, params: Dict[str, Any]) -> None:
        super().__init__()
        self._params = params
        self._simulator = DataSimulator(params)
        self.logger.info("SimulatorAPI ready with params %s", params)

    def simulate_schrodinger(self, trajectories: int = 1) -> Dict[str, torch.Tensor]:
        return self._simulator.generate_dataset(trajectories=trajectories, use_lindblad=False)

    def simulate_lindblad(self, trajectories: int = 1) -> Dict[str, torch.Tensor]:
        return self._simulator.generate_dataset(trajectories=trajectories, use_lindblad=True)

    def run_pulse(self, trajectories: int = 1, pulse_params: Dict[str, Any] | None = None) -> Dict[str, torch.Tensor]:
        params = dict(self._params)
        if pulse_params:
            params.update(pulse_params)
        sim = DataSimulator(params)
        return sim.generate_dataset(trajectories=trajectories, use_lindblad=params.get("use_lindblad", True))

    def batch_simulate(self, batch_configs: list[Dict[str, Any]]) -> list[Dict[str, torch.Tensor]]:
        results = []
        for cfg in batch_configs:
            sim = DataSimulator(cfg)
            results.append(sim.generate_dataset(trajectories=int(cfg.get("trajectories", 1)), use_lindblad=cfg.get("use_lindblad", True)))
        self.logger.info("Batch simulated %d configs", len(batch_configs))
        return results
