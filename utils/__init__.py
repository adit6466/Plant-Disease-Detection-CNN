"""
utils/__init__.py
-----------------
Makes `utils` a Python package and exposes a convenience logger
that every sub-module can import with:

    from utils import get_logger
    logger = get_logger(__name__)
"""

import logging
import sys
from config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """
    Return a consistently configured logger.

    Parameters
    ----------
    name : str
        Typically __name__ from the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger

