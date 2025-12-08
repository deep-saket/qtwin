from __future__ import annotations

from typing import List

import numpy as np
import torch

from quantum_twin.core.BaseComponent import BaseComponent


class LindbladOperators(BaseComponent):
    """Factory generating collapse operators for T1, T2, and pure dephasing."""

    def __init__(self, t1: float = 30.0, t2: float = 20.0, tphi: float = 0.0) -> None:
        super().__init__()
        self._t1 = t1
        self._t2 = t2
        self._tphi = tphi
        self.logger.info("LindbladOperators t1=%.3f t2=%.3f tphi=%.3f", t1, t2, tphi)

    def operators(self) -> List[torch.Tensor]:
        sigma_minus = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.cdouble)
        sigma_plus = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.cdouble)
        sigma_z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.cdouble)

        ops: List[torch.Tensor] = []
        if self._t1 > 0:
            gamma1 = 1.0 / self._t1
            ops.append(np.sqrt(gamma1) * sigma_minus)
        if self._t2 > 0:
            gamma2 = 1.0 / (2 * self._t2)
            ops.append(np.sqrt(gamma2) * sigma_plus)
        if self._tphi > 0:
            ops.append(np.sqrt(self._tphi) * sigma_z)
        self.logger.info("Generated %d collapse operators", len(ops))
        return ops

