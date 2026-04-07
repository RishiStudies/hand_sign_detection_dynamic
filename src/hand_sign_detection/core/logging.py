"""Logging setup for the application.

Provides structured logging with optional file rotation.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

from hand_sign_detection.core.config import get_settings


def setup_logging(
    logger_name: str = "hand_sign_detection",
    log_level: str | None = None,
    log_to_file: bool | None = None,
    logs_dir: str | None = None,
) -> logging.Logger:
    """Configure structured logging with optional file rotation.

    Args:
        logger_name: Name for the logger
        log_level: Override log level (uses settings if None)
        log_to_file: Override file logging (uses settings if None)
        logs_dir: Override logs directory (uses settings if None)

    Returns:
        Configured logger instance

    Logs to:
    - stderr (always)
    - logs/{logger_name}.log (if LOG_TO_FILE enabled)
    """
    settings = get_settings()

    level = log_level or settings.log_level
    to_file = log_to_file if log_to_file is not None else settings.log_to_file
    directory = logs_dir or settings.logs_dir

    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level, logging.INFO))

    # Clear existing handlers
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional, with rotation)
    if to_file:
        os.makedirs(directory, exist_ok=True)
        log_file = os.path.join(directory, f"{logger_name}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=50_000_000,  # 50MB
            backupCount=5,
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("File logging enabled at %s", log_file)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the application prefix.

    Args:
        name: Logger name suffix (e.g., "api", "training")

    Returns:
        Logger instance
    """
    return logging.getLogger(f"hand_sign_detection.{name}")
