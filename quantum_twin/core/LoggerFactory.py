from __future__ import annotations

import logging
from typing import Type

from quantum_twin.core.BaseLogger import BaseLogger


class LoggerFactory:
    """Factory for creating class-scoped loggers."""

    @staticmethod
    def get_logger(klass: Type[BaseLogger]) -> logging.Logger:
        logger = logging.getLogger(f"{klass.__module__}.{klass.__name__}")
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
        logger.debug("LoggerFactory created logger for %s", klass.__name__)
        return logger

