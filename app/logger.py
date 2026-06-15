"""
Structured logging setup.
All pipeline stages log with stock + date context for easy grep.
"""
import logging
import sys
from typing import Optional

from config.settings import settings


def get_logger(name: str, stock: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s"
        if stock:
            fmt += f" | {stock}"
        fmt += " | %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
        logger.propagate = False
    return logger
