from __future__ import annotations

from typing import Callable, Tuple

import torch

from quantum_twin.core.BaseComponent import BaseComponent


class Hamiltonian(BaseComponent):
    """Constructs Hamiltonians with drift and control pulses."""

    def __init__(self, drift: float = 0.0) -> None:
        super().__init__()
        self._drift = drift
        self._pauli_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.cdouble)
        self._pauli_y = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.cdouble)
        self._pauli_z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.cdouble)
        self.logger.info("Hamiltonian initialized drift=%s", drift)

    def paulis(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._pauli_x, self._pauli_y, self._pauli_z

    def build(self, controls: torch.Tensor) -> torch.Tensor:
        """Builds H = drift*Z + c_x X + c_y Y + c_z Z."""
        hx, hy, hz = controls.unbind(-1)
        h_mat = (
            hx.unsqueeze(-1).unsqueeze(-1) * self._pauli_x
            + hy.unsqueeze(-1).unsqueeze(-1) * self._pauli_y
            + (hz + self._drift).unsqueeze(-1).unsqueeze(-1) * self._pauli_z
        )
        return h_mat

