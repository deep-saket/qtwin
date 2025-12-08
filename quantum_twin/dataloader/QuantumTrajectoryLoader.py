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
        device_str = str(samples["t"].device)
        use_mps = device_str.startswith("mps")
        float_dtype = torch.float32 if use_mps else torch.float64
        complex_dtype = torch.cfloat if use_mps else torch.cdouble
        self._t = samples["t"].to(dtype=float_dtype)
        self._controls = samples["controls"].to(dtype=float_dtype)
        self._rho = samples["rho"].to(dtype=complex_dtype)
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

        input_path = params.get("input_path")
        if input_path:
            map_location = "cpu" if str(self._device).startswith("mps") else self._device
            loaded = torch.load(input_path, map_location=map_location)
            if str(self._device).startswith("mps"):
                t = loaded["t"].float().to(self._device)
                controls = loaded["controls"].float().to(self._device)
                rho = loaded["rho"].to(torch.cfloat).to(self._device)
            else:
                t = loaded["t"].to(self._device)
                controls = loaded["controls"].to(self._device)
                rho = loaded["rho"].to(self._device)
            self._samples = {"t": t, "controls": controls, "rho": rho}
            self.logger.info("Loaded dataset from %s (map_location=%s)", input_path, map_location)
        else:
            simulator_params = params.get("simulator", params)
            self._simulator = DataSimulator(**simulator_params)  # type: ignore[arg-type]
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
