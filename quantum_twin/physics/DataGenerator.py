from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp
from typing import Dict, Tuple

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.physics.Hamiltonian import Hamiltonian
from quantum_twin.physics.LindbladOperators import LindbladOperators


class DataGenerator(BaseComponent):
    """Generates qubit trajectories using SciPy solvers."""

    def __init__(self, config: Dict[str, float]) -> None:
        super().__init__()
        self._config = config
        self._hamiltonian = Hamiltonian(
            drift=float(config.get("drift", 0.0)), pulse_scale=float(config.get("pulse_scale", 1.0))
        )
        self._lindblad_ops = LindbladOperators(
            t1=float(config.get("t1", 30.0)),
            t2=float(config.get("t2", 20.0)),
            dephasing=float(config.get("dephasing", 0.0)),
        )
        self.logger.info("DataGenerator configured with %s", config)

    def _sample_controls(self) -> np.ndarray:
        pulse_scale = float(self._config.get("pulse_scale", 0.5))
        controls = np.random.uniform(-pulse_scale, pulse_scale, size=(3,))
        self.logger.debug("Sampled controls %s", controls)
        return controls

    def _solve_schrodinger(self, controls: np.ndarray, t_max: float, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        psi0 = np.array([1.0 + 0j, 0.0 + 0j], dtype=np.complex128)
        times = np.linspace(0, t_max, steps)

        def rhs(t: float, psi_real_imag: np.ndarray) -> np.ndarray:
            psi = psi_real_imag.view(np.complex128)
            t_tensor = torch.tensor(t, dtype=torch.double)
            h_mat = self._hamiltonian.build(t_tensor.unsqueeze(0), torch.tensor(controls).unsqueeze(0))
            h_np = h_mat.squeeze(0).numpy()
            dpsi = -1j * h_np @ psi
            return dpsi.view(np.float64)

        sol = solve_ivp(rhs, (0, t_max), psi0.view(np.float64), t_eval=times, method="RK45")
        psi = sol.y.view(np.complex128).T  # shape (steps, 2)
        density = np.einsum("bi,bj->bij", psi, np.conjugate(psi))
        return times, density

    def _solve_lindblad(self, controls: np.ndarray, t_max: float, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        rho0 = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 0.0 + 0j]], dtype=np.complex128)
        times = np.linspace(0, t_max, steps)
        ops = [op.numpy() for op in self._lindblad_ops.operators()]

        def rhs(t: float, rho_vec: np.ndarray) -> np.ndarray:
            rho = rho_vec.view(np.complex128).reshape(2, 2)
            t_tensor = torch.tensor(t, dtype=torch.double)
            h_mat = self._hamiltonian.build(t_tensor.unsqueeze(0), torch.tensor(controls).unsqueeze(0))
            h_np = h_mat.squeeze(0).numpy()

            comm = h_np @ rho - rho @ h_np
            dissipator = np.zeros_like(rho)
            for l in ops:
                term1 = l @ rho @ l.conj().T
                term2 = 0.5 * (l.conj().T @ l @ rho + rho @ l.conj().T @ l)
                dissipator += term1 - term2
            drho_dt = -1j * comm + dissipator
            return drho_dt.reshape(-1).view(np.float64)

        sol = solve_ivp(rhs, (0, t_max), rho0.reshape(-1).view(np.float64), t_eval=times, method="RK45")
        rho = sol.y.view(np.complex128).T.reshape(-1, 2, 2)
        return times, rho

    def generate(self, trajectories: int, use_lindblad: bool = True) -> Dict[str, torch.Tensor]:
        t_max = float(self._config.get("t_max", 1.0))
        steps = int(self._config.get("steps", 100))
        t_list, control_list, rho_list = [], [], []

        for _ in range(trajectories):
            controls = self._sample_controls()
            if use_lindblad:
                times, rho = self._solve_lindblad(controls, t_max, steps)
            else:
                times, rho = self._solve_schrodinger(controls, t_max, steps)
            for time_point, rho_point in zip(times, rho):
                t_list.append([time_point])
                control_list.append(controls)
                rho_list.append(rho_point)

        t_tensor = torch.tensor(np.array(t_list), dtype=torch.double)
        control_tensor = torch.tensor(np.array(control_list), dtype=torch.double)
        rho_tensor = torch.tensor(np.array(rho_list), dtype=torch.cdouble)
        self.logger.info(
            "Generated dataset with %d samples (lindblad=%s)", t_tensor.shape[0], use_lindblad
        )
        return {"t": t_tensor, "controls": control_tensor, "rho": rho_tensor}

