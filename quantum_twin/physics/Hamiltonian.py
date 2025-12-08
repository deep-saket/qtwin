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

    def build(self, controls: torch.Tensor | "np.ndarray") -> torch.Tensor:
        """Builds H = drift*Z + c_x X + c_y Y + c_z Z."""
        if not hasattr(controls, "unbind"):
            import torch as _torch

            controls = _torch.tensor(controls, dtype=_torch.double)
        hx, hy, hz = controls.unbind(-1)
        complex_dtype = torch.cfloat if hx.dtype == torch.float32 else torch.cdouble
        px = self._pauli_x.to(complex_dtype)
        py = self._pauli_y.to(complex_dtype)
        pz = self._pauli_z.to(complex_dtype)
        h_mat = (
            hx.unsqueeze(-1).unsqueeze(-1) * px
            + hy.unsqueeze(-1).unsqueeze(-1) * py
            + (hz + self._drift).unsqueeze(-1).unsqueeze(-1) * pz
        )
        return h_mat
