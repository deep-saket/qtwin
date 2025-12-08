from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.physics.Hamiltonian import Hamiltonian
from quantum_twin.physics.LindbladOperators import LindbladOperators


class LindbladSolver(BaseComponent):
    """Integrates the Lindblad master equation for density matrices."""

    def __init__(self, hamiltonian: Hamiltonian, operators: LindbladOperators) -> None:
        super().__init__()
        self._hamiltonian = hamiltonian
        self._ops = operators.operators()
        self.logger.info("LindbladSolver ready with %d operators", len(self._ops))

    def solve(self, controls: np.ndarray, t_max: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
        rho0 = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 0.0 + 0j]], dtype=np.complex128)
        times = np.linspace(0, t_max, steps)
        ops_np = [op.numpy() for op in self._ops]

        def rhs(t: float, rho_vec: np.ndarray) -> np.ndarray:
            rho = rho_vec.view(np.complex128).reshape(2, 2)
            h_mat = self._hamiltonian.build(
                controls=np.broadcast_to(controls, (1, controls.shape[-1]))
            ).squeeze(0)
            h_np = h_mat.numpy()
            comm = h_np @ rho - rho @ h_np
            dissipator = np.zeros_like(rho)
            for l in ops_np:
                term1 = l @ rho @ l.conj().T
                term2 = 0.5 * (l.conj().T @ l @ rho + rho @ l.conj().T @ l)
                dissipator += term1 - term2
            drho_dt = -1j * comm + dissipator
            return drho_dt.reshape(-1).view(np.float64)

        sol = solve_ivp(rhs, (0, t_max), rho0.reshape(-1).view(np.float64), t_eval=times, method="RK45")
        rho = sol.y.view(np.complex128).T.reshape(-1, 2, 2)
        return times, rho

