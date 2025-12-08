from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch.utils.data import Dataset

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.dataloader.BaseDataLoader import BaseDataLoader
from quantum_twin.physics.DataSimulator import DataSimulator


class QuantumTrajectoryDataset(Dataset, BaseComponent):
    """Dataset wrapping simulated qubit trajectories."""

    def __init__(self, samples: Dict[str, torch.Tensor]) -> None:
        BaseComponent.__init__(self)
        self._t = samples["t"].double()
        self._controls = samples["controls"].double()
        self._rho = samples["rho"].to(torch.cdouble)
        self.logger.info("QuantumTrajectoryDataset created with %d samples", len(self._t))

    def __len__(self) -> int:
        return self._t.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._t[idx], self._controls[idx], self._rho[idx]


class QuantumTrajectoryLoader(BaseDataLoader):
    """Builds datasets using the physics DataSimulator."""

    def __init__(self, **params: Dict[str, object]) -> None:
        batch_size = int(params.get("batch_size", 32))
        shuffle = bool(params.get("shuffle", True))
        num_workers = int(params.get("num_workers", 0))
        self._device = params.get("device", "cpu")
        super().__init__(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        simulator_params = params.get("simulator", params)
        self._simulator = DataSimulator(simulator_params)  # type: ignore[arg-type]
        self._samples = self._simulator.generate_dataset(
            trajectories=int(params.get("trajectories", 4)),
            use_lindblad=bool(params.get("use_lindblad", True)),
        )
        self._dataset = QuantumTrajectoryDataset(self._samples)
        self.logger.info("QuantumTrajectoryLoader ready with params: %s", params)

    def dataset(self) -> Dataset:
        return self._dataset

    def loader(self) -> Iterable[Tuple[torch.Tensor, ...]]:
        return super().loader()
