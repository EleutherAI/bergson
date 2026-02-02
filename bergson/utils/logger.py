import logging
import sys
from typing import Optional

import torch.distributed as dist

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(funcName)s:%(lineno)d:%(levelname)s:  %(message)s",
    stream=sys.stdout,  # Ensure output goes to stdout
    force=True,  # Override any existing configuration
)


class RankFilter(logging.Filter):
    """Filter that only allows logging from rank 0."""

    def filter(self, record):
        rank = dist.get_rank() if dist.is_initialized() else 0
        return rank == 0


# Create a function to get loggers with consistent naming
def get_logger(
    name: Optional[str] = None, level: Optional[str] = None
) -> logging.Logger:
    """
    Get a logger with the configured format.

    Args:
        name: Logger name. If None, uses the calling module's __name__
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
               If None, uses the global default.

    Returns:
        Configured logger instance
    """
    if name is None:
        # Get the calling module's name
        import inspect

        frame = inspect.currentframe().f_back  # type: ignore
        assert frame is not None, "Frame should not be None"
        name = frame.f_globals.get("__name__", "unknown")

    logger = logging.getLogger(name)

    # Add rank filter if not already present
    if not any(isinstance(f, RankFilter) for f in logger.filters):
        logger.addFilter(RankFilter())

    if level is not None:
        # Convert string level to logging constant
        level_mapping = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        log_level = level_mapping.get(level.upper(), logging.INFO)
        logger.setLevel(log_level)

    return logger
