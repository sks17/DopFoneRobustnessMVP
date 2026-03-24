"""Simple logging helpers."""

from __future__ import annotations

import logging


# Spec:
# - General description: Configure and return a standard module logger.
# - Params: `logger_name`, logger name string.
# - Pre: `logger_name` is non-empty.
# - Post: Returns a logger configured at INFO level with at least one handler.
# - Mathematical definition: Not applicable; this is a side-effectful logging configuration helper.
def configure_logger(logger_name: str) -> logging.Logger:
    """Return a basic configured logger."""
    if logger_name.strip() == "":
        raise ValueError("logger_name must be non-empty.")
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if len(logger.handlers) == 0:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
