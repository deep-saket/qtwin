from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from quantum_twin.core.BaseComponent import BaseComponent


class FourierFeatures(BaseComponent):
    """Applies Fourier feature mapping to time inputs."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self._num_features = num_features
        self.logger.info("FourierFeatures num=%d", num_features)

    def encode(self, t: torch.Tensor) -> torch.Tensor:
        features: List[torch.Tensor] = [t]
        for k in range(self._num_features):
            freq = 2.0 ** k * torch.pi
            features.append(torch.sin(freq * t))
            features.append(torch.cos(freq * t))
        encoded = torch.cat(features, dim=1)
        self.logger.debug("Encoded time shape=%s", encoded.shape)
        return encoded

