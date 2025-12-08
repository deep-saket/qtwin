from __future__ import annotations

from quantum_twin.core.BaseLogger import BaseLogger
from quantum_twin.core.LoggerFactory import LoggerFactory


class BaseComponent(BaseLogger):
    """Base class for all components providing logging support."""

    def __init__(self) -> None:
        super().__init__(self.__class__.__name__)
        self._logger = LoggerFactory.get_logger(self.__class__)
        self.logger.info("Constructed %s", self.__class__.__name__)

