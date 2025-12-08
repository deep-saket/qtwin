from __future__ import annotations

from typing import Any, Dict

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.losses.LindbladLoss import LindbladLoss
from quantum_twin.losses.SchrodingerLoss import SchrodingerLoss
from quantum_twin.physics.Hamiltonian import Hamiltonian
from quantum_twin.physics.LindbladOperators import LindbladOperators


class LossFactory(BaseComponent):
    """Factory that builds physics losses from inline configuration."""

    def __init__(self, **params: Dict[str, Any]) -> None:
        super().__init__()
        self._params = params
        self.logger.info("LossFactory initialized with params %s", params)

    def build(self) -> SchrodingerLoss | LindbladLoss:
        weights: Dict[str, float] = self._params.get("weights", {})
        use_lindblad = bool(self._params.get("use_lindblad", True))
        hamiltonian = Hamiltonian(drift=float(self._params.get("drift", 0.0)))
        if use_lindblad:
            ops = LindbladOperators(
                t1=float(self._params.get("t1", 30.0)),
                t2=float(self._params.get("t2", 20.0)),
                tphi=float(self._params.get("tphi", 0.0)),
            )
            return LindbladLoss(hamiltonian, ops, weights)
        return SchrodingerLoss(hamiltonian, weights)
