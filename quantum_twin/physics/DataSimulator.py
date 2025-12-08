from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.physics.Hamiltonian import Hamiltonian
from quantum_twin.physics.LindbladOperators import LindbladOperators
from quantum_twin.physics.LindbladSolver import LindbladSolver
from quantum_twin.physics.PulseGenerator import PulseGenerator
from quantum_twin.physics.SchrodingerSolver import SchrodingerSolver


class DataSimulator(BaseComponent):
    """Generates synthetic qubit trajectories via Schrödinger or Lindblad dynamics."""

    def __init__(self, **params) -> None:
        super().__init__()
        self._params = params
        self._device = params.get("device", "cpu")
        self._output_path = Path(params.get("output_path")) if params.get("output_path") else None
        self._pulse_gen = PulseGenerator(scale=float(params.get("pulse_scale", 0.5)))
        self._hamiltonian = Hamiltonian(drift=float(params.get("drift", 0.0)))
        self._lindblad_ops = LindbladOperators(
            t1=float(params.get("t1", 30.0)), t2=float(params.get("t2", 20.0)), tphi=float(params.get("tphi", 0.0))
        )
        self._sch_solver = SchrodingerSolver(self._hamiltonian)
        self._lin_solver = LindbladSolver(self._hamiltonian, self._lindblad_ops)
        self.logger.info("DataSimulator configured with %s", params)

    def _sample_controls(self) -> np.ndarray:
        return self._pulse_gen.sample_pulse()

    def generate_dataset(self, trajectories: int, use_lindblad: bool = True) -> Dict[str, torch.Tensor]:
        t_max = float(self._params.get("t_max", 1.0))
        steps = int(self._params.get("steps", 100))
        t_list, control_list, rho_list = [], [], []
        for _ in range(trajectories):
            controls = self._sample_controls()
            if use_lindblad:
                times, rho = self._lin_solver.solve(controls, t_max, steps)
            else:
                times, rho = self._sch_solver.solve(controls, t_max, steps)
            for time_point, rho_point in zip(times, rho):
                t_list.append([time_point])
                control_list.append(controls)
                rho_list.append(rho_point)
        device = torch.device(self._device)
        float_dtype = torch.float32 if str(device).startswith("mps") else torch.double
        complex_dtype = torch.cfloat if float_dtype == torch.float32 else torch.cdouble
        t_tensor = torch.tensor(np.array(t_list), dtype=float_dtype, device=device)
        control_tensor = torch.tensor(np.array(control_list), dtype=float_dtype, device=device)
        rho_tensor = torch.tensor(np.array(rho_list), dtype=complex_dtype, device=device)
        dataset = {"t": t_tensor, "controls": control_tensor, "rho": rho_tensor}
        if self._output_path:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dataset, self._output_path)
            self.logger.info("Saved dataset to %s", self._output_path)
        self.logger.info("Generated dataset total samples=%d", len(t_tensor))
        return dataset
