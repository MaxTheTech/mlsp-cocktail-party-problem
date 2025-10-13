import logging
import sys
from pathlib import Path


def setup_logger(name = __name__, log_file = None, level = "INFO"):
    """
    Configure and return a logger instance.

    Args:
        name (str): Name of the logger (usually __name__).
        log_file (str | Path | None): Optional file path for saving logs.
        level (str): Log level (e.g. "DEBUG", "INFO", "WARNING", "ERROR").

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    # avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    # logging format: timestamp | logging level | logger name | message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # optional file handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
