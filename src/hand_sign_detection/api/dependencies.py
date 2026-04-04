"""FastAPI dependencies for dependency injection."""

from fastapi import Depends, Header, HTTPException, Request

from hand_sign_detection.core.config import Settings, get_settings
from hand_sign_detection.models.manager import ModelManager, get_model_manager
from hand_sign_detection.services.combo_detection import ComboService, get_combo_service
from hand_sign_detection.services.prediction import PredictionService, get_prediction_service
from hand_sign_detection.services.rate_limiting import RateLimiter, get_rate_limiter


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
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(settings_dependency),
) -> None:
    """Enforce API key requirement for training endpoints.

    Raises HTTPException 403 if key is invalid/missing in production.
    """
    import logging

    logger = logging.getLogger("hand_sign_detection.auth")

    if not settings.training_api_key:
        logger.warning("Training API not secured: TRAINING_API_KEY environment variable not set")
        return

    if x_api_key != settings.training_api_key:
        logger.warning("Unauthorized training attempt with invalid/missing API key")
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
