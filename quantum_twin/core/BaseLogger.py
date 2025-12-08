from __future__ import annotations

import logging
from abc import ABC
from typing import ClassVar


class BaseLogger(ABC):
    """Abstract base class providing configured instance and class loggers."""

    _class_logger: ClassVar[logging.Logger]

    def __init__(self, name: str) -> None:
        self._logger = self._configure_logger(name)
        self.__class__._class_logger = self._logger
        self._logger.debug("BaseLogger initialized for %s", name)

    @classmethod
    def _configure_logger(cls, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    @property
    def logger(self) -> logging.Logger:
        """Return the configured instance logger."""
        return self._logger

    @classmethod
    def class_logger(cls) -> logging.Logger:
        """Return the class-level logger."""
        if hasattr(cls, "_class_logger"):
            return cls._class_logger
        return cls._configure_logger(f"{cls.__module__}.{cls.__name__}")
