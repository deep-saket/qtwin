from __future__ import annotations

from typing import Callable, List

import torch
import torch.nn as nn

from quantum_twin.core.BaseComponent import BaseComponent
from quantum_twin.utils.MathUtils import MathUtils


class PINNModel(BaseComponent, nn.Module):
    """Physics-informed neural network modeling qubit density evolution."""

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
        self._fourier_features = fourier_features
        self._input_dim = input_dim + 2 * fourier_features
        self._layers = layers
        self._activation = self._get_activation(activation)
        self._dropout = dropout

        hidden_sizes = layers[:-1] if len(layers) > 1 else [64]
        output_dim = layers[-1] if len(layers) > 0 else 4

        modules: List[nn.Module] = []
        prev = self._input_dim
        for size in hidden_sizes:
            modules.append(nn.Linear(prev, size))
            modules.append(self._activation)
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
            prev = size
        modules.append(nn.Linear(prev, output_dim))

        self.network = nn.Sequential(*modules)
        self.logger.info(
            "PINNModel initialized input_dim=%d hidden=%s output=%d fourier=%d",
            self._input_dim,
            hidden_sizes,
            output_dim,
            fourier_features,
        )

    def _get_activation(self, name: str) -> nn.Module:
        activations: dict[str, Callable[[], nn.Module]] = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        if name.lower() not in activations:
            self.logger.warning("Unknown activation %s, defaulting to tanh", name)
        return activations.get(name.lower(), nn.Tanh)()

    def _encode_time(self, t: torch.Tensor) -> torch.Tensor:
        if self._fourier_features <= 0:
            return t
        features = [t]
        for k in range(self._fourier_features):
            freq = 2.0 ** k * torch.pi
            features.append(torch.sin(freq * t))
            features.append(torch.cos(freq * t))
        encoded = torch.cat(features, dim=1)
        self.logger.debug("Fourier-encoded time shape %s", encoded.shape)
        return encoded

    def forward(self, t: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        self.logger.debug("Forward pass t=%s controls=%s", t.shape, controls.shape)
        t_feat = self._encode_time(t)
        x = torch.cat([t_feat, controls], dim=1)
        out = self.network(x)

        diag1 = torch.nn.functional.softplus(out[:, 0])
        diag2 = torch.nn.functional.softplus(out[:, 1])
        off_re = out[:, 2]
        off_im = out[:, 3]

        rho = torch.zeros((out.shape[0], 2, 2), dtype=torch.cdouble, device=out.device)
        rho[:, 0, 0] = diag1
        rho[:, 1, 1] = diag2
        rho[:, 0, 1] = off_re + 1j * off_im
        rho[:, 1, 0] = off_re - 1j * off_im

        rho = MathUtils.enforce_hermitian(rho)
        rho = MathUtils.enforce_positive_semidefinite(rho)
        self.logger.debug("Produced density matrix shape %s", rho.shape)
        return rho

