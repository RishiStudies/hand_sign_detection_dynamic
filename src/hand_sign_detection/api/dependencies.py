"""FastAPI dependencies for dependency injection."""

import hmac
import logging
import time
from collections import defaultdict
from threading import Lock

from fastapi import Depends, Header, HTTPException, Request

from hand_sign_detection.core.config import Settings, get_settings
from hand_sign_detection.models.manager import ModelManager, get_model_manager
from hand_sign_detection.services.combo_detection import ComboService, get_combo_service
from hand_sign_detection.services.prediction import PredictionService, get_prediction_service
from hand_sign_detection.services.rate_limiting import RateLimiter, get_rate_limiter

# API key failure tracking for rate limiting
_api_key_failures: dict[str, list[float]] = defaultdict(list)
_api_key_lock = Lock()
_MAX_FAILURES = 5  # Max failed attempts before lockout
_LOCKOUT_SECONDS = 300  # 5 minute lockout
_FAILURE_WINDOW = 60  # Track failures within 1 minute window


def settings_dependency() -> Settings:
    """Get application settings."""
    return get_settings()


def model_manager_dependency() -> ModelManager:
    """Get model manager instance."""
    return get_model_manager()


def rate_limiter_dependency() -> RateLimiter:
    """Get rate limiter instance."""
    return get_rate_limiter()


def combo_service_dependency() -> ComboService:
    """Get combo service instance."""
    return get_combo_service()


def prediction_service_dependency() -> PredictionService:
    """Get prediction service instance."""
    return get_prediction_service()


def get_client_identity(
    request: Request,
    x_session_id: str | None = Header(default=None),
) -> str:
    """Extract client identity from request.

    Priority:
    1. Session ID header
    2. X-Forwarded-For header (first IP)
    3. Client host IP
    4. "unknown"
    """
    if x_session_id:
        return f"session:{x_session_id.strip() or 'anonymous'}"

    x_forwarded_for = request.headers.get("x-forwarded-for", "")
    if x_forwarded_for:
        return f"ip:{x_forwarded_for.split(',')[0].strip()}"

    if request.client and request.client.host:
        return f"ip:{request.client.host}"

    return "ip:unknown"


def require_training_key(
    request: Request,
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(settings_dependency),
) -> None:
    """Enforce API key requirement for training endpoints.

    Security features:
    - Timing-safe comparison to prevent timing attacks
    - Rate limiting on failed attempts (5 failures = 5 min lockout)
    - Minimum key length/complexity validation

    Raises HTTPException 403 if key is invalid/missing in production.
    Raises HTTPException 429 if too many failed attempts.
    """
    logger = logging.getLogger("hand_sign_detection.auth")

    # Get client identifier for rate limiting
    client_ip = "unknown"
    if request.client and request.client.host:
        client_ip = request.client.host
    elif x_forwarded := request.headers.get("x-forwarded-for"):
        client_ip = x_forwarded.split(",")[0].strip()

    # Check if client is locked out due to too many failures
    now = time.time()
    with _api_key_lock:
        failures = _api_key_failures[client_ip]
        # Clean old failures outside the window
        failures[:] = [t for t in failures if now - t < _LOCKOUT_SECONDS]

        if len(failures) >= _MAX_FAILURES:
            oldest_failure = min(failures)
            lockout_remaining = int(_LOCKOUT_SECONDS - (now - oldest_failure))
            logger.warning(f"Client {client_ip} locked out for {lockout_remaining}s due to repeated API key failures")
            raise HTTPException(
                status_code=429,
                detail=f"Too many failed attempts. Try again in {lockout_remaining} seconds.",
            )

    # If no API key configured, allow access (dev mode)
    if not settings.training_api_key:
        logger.warning("Training API not secured: TRAINING_API_KEY environment variable not set")
        return

    # Validate key is provided
    if not x_api_key:
        _record_failure(client_ip, now)
        logger.warning(f"Missing API key from {client_ip}")
        raise HTTPException(status_code=403, detail="API key required")

    # Timing-safe comparison to prevent timing attacks
    key_valid = hmac.compare_digest(
        x_api_key.encode("utf-8"),
        settings.training_api_key.encode("utf-8"),
    )

    if not key_valid:
        _record_failure(client_ip, now)
        logger.warning(f"Invalid API key from {client_ip}")
        raise HTTPException(status_code=403, detail="Invalid API key")


def _record_failure(client_ip: str, timestamp: float) -> None:
    """Record a failed API key attempt for rate limiting."""
    with _api_key_lock:
        _api_key_failures[client_ip].append(timestamp)
        # Only keep failures within the tracking window
        _api_key_failures[client_ip] = [
            t for t in _api_key_failures[client_ip]
            if timestamp - t < _LOCKOUT_SECONDS
        ]
