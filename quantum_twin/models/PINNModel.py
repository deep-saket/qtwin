from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn as nn

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.models.FourierFeatures import FourierFeatures
from quantum_twin.utils.DensityMatrixUtils import DensityMatrixUtils


class PINNModel(BaseComponent, nn.Module):
    """Physics-informed neural network with Cholesky parameterization."""

    def __init__(
        self,
        input_dim: int,
        layers: List[int],
        activation: str = "tanh",
        dropout: float = 0.0,
        fourier_features: int = 0,
    ) -> None:
        nn.Module.__init__(self)
        BaseComponent.__init__(self)
        self._activation = self._get_activation(activation)
        self._dropout = dropout
        self._fourier = FourierFeatures(fourier_features) if fourier_features > 0 else None
        effective_input = input_dim + (2 * fourier_features if fourier_features > 0 else 0)
        hidden_sizes = layers[:-1]
        output_dim = layers[-1]

        modules: List[nn.Module] = []
        prev = effective_input
        for size in hidden_sizes:
            modules.append(nn.Linear(prev, size))
            modules.append(self._activation)
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            prev = size
        modules.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*modules)
        self.logger.info(
            "PINNModel initialized in=%d hidden=%s out=%d fourier=%d",
            effective_input,
            hidden_sizes,
            output_dim,
            fourier_features,
        )

    def _get_activation(self, name: str) -> nn.Module:
        mapping: dict[str, Callable[[], nn.Module]] = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        return mapping.get(name.lower(), nn.Tanh)()

    def _encode(self, t: torch.Tensor) -> torch.Tensor:
        if self._fourier:
            return self._fourier.encode(t)
        return t

    def forward(self, t: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        t_feat = self._encode(t)
        x = torch.cat([t_feat, controls], dim=1)
        raw = self.network(x)
        l11 = torch.nn.functional.softplus(raw[:, 0]) + 1e-6
        l22 = torch.nn.functional.softplus(raw[:, 1]) + 1e-6
        l21_re = raw[:, 2]
        l21_im = raw[:, 3]

        l21 = l21_re + 1j * l21_im
        chol = torch.zeros((raw.shape[0], 2, 2), dtype=torch.cdouble, device=raw.device)
        chol[:, 0, 0] = l11
        chol[:, 1, 0] = l21
        chol[:, 1, 1] = l22

        rho = chol @ chol.conj().transpose(-1, -2)
        rho = DensityMatrixUtils.clamp_physical(rho)
        return rho
