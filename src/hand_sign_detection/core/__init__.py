"""Core infrastructure modules.

Contains:
- config: Centralized Pydantic settings
- logging: Logging setup
- redis: Redis client factory
- shared_state: Shared artifacts state management
"""

from hand_sign_detection.core.config import Settings, get_settings
from hand_sign_detection.core.logging import setup_logging
from hand_sign_detection.core.redis import get_redis_client
from hand_sign_detection.core.shared_state import (
    load_shared_state,
    resolve_shared_path,
    save_shared_state,
    update_shared_state,
)

__all__ = [
    "Settings",
    "get_settings",
    "setup_logging",
    "get_redis_client",
    "load_shared_state",
    "save_shared_state",
    "update_shared_state",
    "resolve_shared_path",
]
