from __future__ import annotations

from typing import Dict

import torch

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.physics.Hamiltonian import Hamiltonian
from quantum_twin.physics.PhysicsLoss import PhysicsLoss
from quantum_twin.utils.MathUtils import MathUtils


class SchrodingerLoss(PhysicsLoss):
    """Physics-informed loss enforcing the Schrödinger equation."""

    def __init__(self, hamiltonian: Hamiltonian, weights: Dict[str, float]) -> None:
        super().__init__(weights)
        self._hamiltonian = hamiltonian
        self.logger.info("SchrodingerLoss ready")

    def compute(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        t = batch["t"].requires_grad_(True)
        controls = batch["controls"]
        self.logger.debug("Computing Schrödinger loss on batch t shape=%s", t.shape)
        rho_pred: torch.Tensor = model(t, controls)

        h_mat = self._hamiltonian.build(t.squeeze(-1), controls)
        commutator = h_mat @ rho_pred - rho_pred @ h_mat

        drho_dt = torch.autograd.grad(
            outputs=rho_pred,
            inputs=t,
            grad_outputs=torch.ones_like(rho_pred),
            retain_graph=True,
            create_graph=True,
        )[0]
        drho_dt_matrix = drho_dt.view(-1, 1, 1)

        residual = drho_dt_matrix + 1j * commutator
        physics_loss = torch.mean(torch.abs(residual) ** 2)

        trace_penalty = torch.mean(
            torch.abs(torch.real(torch.diagonal(rho_pred, dim1=-2, dim2=-1).sum(dim=-1)) - 1.0)
        )
        hermitian_penalty = torch.mean(torch.abs(rho_pred - rho_pred.conj().transpose(-1, -2)))

        rho_psd = MathUtils.enforce_positive_semidefinite(rho_pred)
        positivity_penalty = torch.mean(torch.relu(-torch.real(torch.linalg.eigvals(rho_psd))))

        total = (
            self._weights.get("schrodinger", 1.0) * physics_loss
            + self._weights.get("trace", 0.0) * trace_penalty
            + self._weights.get("hermitian", 0.0) * hermitian_penalty
            + self._weights.get("positivity", 0.0) * positivity_penalty
        )
        self.logger.debug(
            "Schrödinger components physics=%.4f trace=%.4f hermitian=%.4f positivity=%.4f",
            physics_loss.item(),
            trace_penalty.item(),
            hermitian_penalty.item(),
            positivity_penalty.item(),
        )
        return total
