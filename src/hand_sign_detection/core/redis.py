"""Redis client factory and utilities.

Provides a singleton Redis client with connection management.
"""

import importlib
import logging
from typing import Any

from hand_sign_detection.core.config import get_settings

logger = logging.getLogger("hand_sign_detection.redis")

_redis_client: Any | None = None
_redis_initialized: bool = False


def get_redis_client(force_reconnect: bool = False) -> Any | None:
    """Get Redis client singleton.

    Args:
        force_reconnect: Force re-establishing connection

    Returns:
        Redis client instance or None if unavailable
    """
    global _redis_client, _redis_initialized

    if _redis_initialized and not force_reconnect:
        return _redis_client

    settings = get_settings()

    if not settings.redis_url:
        logger.debug("REDIS_URL not configured - Redis unavailable")
        _redis_initialized = True
        _redis_client = None
        return None

    try:
        redis_module = importlib.import_module("redis")
        client = redis_module.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        client.ping()
        _redis_client = client
        _redis_initialized = True
        logger.info("Redis connected successfully")
        return client
    except ImportError:
        logger.warning("redis package not installed")
        _redis_client = None
        _redis_initialized = True
        return None
    except Exception as exc:
        logger.warning("Redis connection failed: %s", exc)
        _redis_client = None
        _redis_initialized = True
        return None


def is_redis_available() -> bool:
    """Check if Redis is available and connected."""
    client = get_redis_client()
    if client is None:
        return False
    try:
        client.ping()
        return True
    except Exception:
        return False


def reset_redis_client() -> None:
    """Reset Redis client state (for testing)."""
    global _redis_client, _redis_initialized
    _redis_client = None
    _redis_initialized = False
