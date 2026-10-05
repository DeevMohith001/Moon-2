"""
Standardized logging utility for the Reddit Market Sentiment project.
"""

import sys
import logging
from typing import Optional


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger with standard formatting.
    """
    logger = logging.getLogger(name or "LLM_FA")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s • %(levelname)s • %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
