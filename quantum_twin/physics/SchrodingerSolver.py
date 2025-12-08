from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.physics.Hamiltonian import Hamiltonian


class SchrodingerSolver(BaseComponent):
    """Integrates the Schrödinger equation for pure states."""

    def __init__(self, hamiltonian: Hamiltonian) -> None:
        super().__init__()
        self._hamiltonian = hamiltonian
        self.logger.info("SchrodingerSolver ready")

    def solve(self, controls: np.ndarray, t_max: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
        psi0 = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        times = np.linspace(0, t_max, steps)

        def rhs(t: float, psi_real_imag: np.ndarray) -> np.ndarray:
            psi = psi_real_imag.view(np.complex128)
            h_mat = self._hamiltonian.build(
                controls=np.broadcast_to(controls, (1, controls.shape[-1]))
            ).squeeze(0)
            dpsi = -1j * h_mat.numpy() @ psi
            return dpsi.view(np.float64)

        sol = solve_ivp(rhs, (0, t_max), psi0.view(np.float64), t_eval=times, method="RK45")
        psi = sol.y.view(np.complex128).T
        density = np.einsum("bi,bj->bij", psi, np.conjugate(psi))
        return times, density

