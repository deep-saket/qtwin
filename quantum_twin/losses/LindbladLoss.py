from __future__ import annotations

from typing import Dict

import torch

from quantum_twin.losses.PhysicsLoss import PhysicsLoss
from quantum_twin.losses.RegularizationLoss import RegularizationLoss
from quantum_twin.physics.Hamiltonian import Hamiltonian
from quantum_twin.physics.LindbladOperators import LindbladOperators


class LindbladLoss(PhysicsLoss):
    """Physics loss for Lindblad master equation residuals."""

    def __init__(
        self, hamiltonian: Hamiltonian, lindblad_ops: LindbladOperators, weights: Dict[str, float]
    ) -> None:
        super().__init__(weights)
        self._hamiltonian = hamiltonian
        self._ops = lindblad_ops.operators()
        self._reg = RegularizationLoss(weights)
        self.logger.info("LindbladLoss initialized with %d collapse operators", len(self._ops))

    def _dissipator(self, rho: torch.Tensor) -> torch.Tensor:
        dissipator = torch.zeros_like(rho)
        for l_op in self._ops:
            l = l_op.to(device=rho.device, dtype=rho.dtype)
            term1 = l @ rho @ l.conj().transpose(-1, -2)
            term2 = 0.5 * (l.conj().transpose(-1, -2) @ l @ rho + rho @ l.conj().transpose(-1, -2) @ l)
            dissipator = dissipator + term1 - term2
        return dissipator

    def compute(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        t = batch["t"].requires_grad_(True)
        controls = batch["controls"]
        rho_pred = model(t, controls)

        h_mat = self._hamiltonian.build(controls)
        comm = h_mat @ rho_pred - rho_pred @ h_mat
        dissipator = self._dissipator(rho_pred)

        drho_dt = torch.autograd.grad(
            outputs=rho_pred,
            inputs=t,
            grad_outputs=torch.ones_like(rho_pred),
            retain_graph=True,
            create_graph=True,
        )[0].view(-1, 1, 1)

        residual = drho_dt + 1j * comm - dissipator
        physics = torch.mean(torch.abs(residual) ** 2)
        reg = self._reg.compute(rho_pred)
        total = self._weights.get("lindblad", 1.0) * physics + reg
        self.logger.debug("LindbladLoss physics=%.4f reg=%.4f", physics.item(), reg.item())
        return total
