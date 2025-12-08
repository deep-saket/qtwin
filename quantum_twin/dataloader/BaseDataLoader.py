from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from quantum_twin.core.BaseComponent import BaseComponent


class BaseDataLoader(BaseComponent, ABC):
    """Abstract base for data loader wrappers."""

    def __init__(self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0) -> None:
        super().__init__()
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = num_workers
        self.logger.info(
            "BaseDataLoader init batch_size=%d shuffle=%s workers=%d",
            batch_size,
            shuffle,
            num_workers,
        )

    @abstractmethod
    def dataset(self) -> Dataset:
        """Return the underlying dataset."""
        raise NotImplementedError

    def loader(self) -> Iterable[Tuple[torch.Tensor, ...]]:
        """Return a torch DataLoader iterable."""
        ds = self.dataset()
        self.logger.info("Creating DataLoader for dataset of length %d", len(ds))
        return DataLoader(ds, batch_size=self._batch_size, shuffle=self._shuffle, num_workers=self._num_workers)

