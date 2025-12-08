from __future__ import annotations

from typing import Dict

import torch

from quantum_twin.losses.PhysicsLoss import PhysicsLoss
from quantum_twin.losses.RegularizationLoss import RegularizationLoss
from quantum_twin.physics.Hamiltonian import Hamiltonian


class SchrodingerLoss(PhysicsLoss):
    """Physics loss for Schrödinger equation residuals."""

    def __init__(self, hamiltonian: Hamiltonian, weights: Dict[str, float]) -> None:
        super().__init__(weights)
        self._hamiltonian = hamiltonian
        self.logger.info("SchrodingerLoss initialized")

    def compute(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        t = batch["t"].requires_grad_(True)
        controls = batch["controls"]
        rho_pred = model(t, controls)

        h_mat = self._hamiltonian.build(controls)
        comm = h_mat @ rho_pred - rho_pred @ h_mat

        drho_dt = torch.autograd.grad(
            outputs=rho_pred,
            inputs=t,
            grad_outputs=torch.ones_like(rho_pred),
            retain_graph=True,
            create_graph=True,
        )[0].view(-1, 1, 1)

        residual = drho_dt + 1j * comm
        physics = torch.mean(torch.abs(residual) ** 2)

        reg = RegularizationLoss(self._weights).compute(rho_pred)
        total = self._weights.get("schrodinger", 1.0) * physics + reg
        self.logger.debug("SchrodingerLoss physics=%.4f reg=%.4f", physics.item(), reg.item())
        return total
