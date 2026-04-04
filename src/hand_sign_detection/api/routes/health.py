"""Health check endpoints."""

# Import lazily to avoid circular imports
import importlib
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from hand_sign_detection.api.dependencies import (
    model_manager_dependency,
    prediction_service_dependency,
    rate_limiter_dependency,
    settings_dependency,
)
from hand_sign_detection.core.config import Settings
from hand_sign_detection.core.shared_state import load_shared_state
from hand_sign_detection.models.manager import ModelManager
from hand_sign_detection.services.prediction import PredictionService
from hand_sign_detection.services.rate_limiting import RateLimiter

router = APIRouter(tags=["Health"])


@router.get("/health/live")
def health_live() -> dict[str, str]:
    """Liveness probe endpoint.

    Returns:
        {"status": "ok"}
    """
    return {"status": "ok"}


@router.get("/health/ready")
def health_ready(
    model_manager: ModelManager = Depends(model_manager_dependency),
) -> dict[str, str]:
    """Readiness probe endpoint.

    Checks if models are loaded and ready to serve.

    Returns:
        Status of each model type

    Raises:
        HTTPException: 503 if primary model unavailable
    """
    if not model_manager.rf_available:
        raise HTTPException(status_code=503, detail="RandomForest model unavailable")

    return {
        "status": "ready",
        "random_forest": "available",
        "lstm": "available" if model_manager.lstm_available else "unavailable",
    }


@router.get("/health/details")
def health_details(
    model_manager: ModelManager = Depends(model_manager_dependency),
    rate_limiter: RateLimiter = Depends(rate_limiter_dependency),
    settings: Settings = Depends(settings_dependency),
) -> dict[str, Any]:
    """Detailed health status with component information.

    Returns:
        Detailed status of all components
    """
    # Check job queue availability
    job_queue_available = False
    try:
        job_queue_module = importlib.import_module("hand_sign_detection.training.job_queue")
        job_queue_available = job_queue_module.is_job_queue_available()
    except (ImportError, AttributeError):
        pass

    state = load_shared_state()

    return {
        "status": "ready" if model_manager.rf_available else "degraded",
        "rate_limit_backend": rate_limiter.backend,
        "combo_state_backend": rate_limiter.backend,  # Same backend
        "job_queue_backend": "redis_rq" if job_queue_available else "unavailable",
        "model_status": model_manager.get_status(),
        "limits": {
            "max_upload_bytes": settings.max_upload_bytes,
            "max_csv_upload_bytes": settings.max_csv_upload_bytes,
            "max_sequence_frames": settings.max_sequence_frames,
            "max_frame": {
                "width": settings.max_frame_width,
                "height": settings.max_frame_height,
            },
        },
        "shared_artifacts": state,
    }


@router.get("/health/metrics")
def health_metrics(
    prediction_service: PredictionService = Depends(prediction_service_dependency),
) -> dict[str, Any]:
    """Get prediction performance metrics.

    Returns:
        Performance metrics including latency, cache stats, etc.
    """
    return {
        "status": "ok",
        "timestamp": time.time(),
        "prediction_metrics": prediction_service.get_metrics(),
    }


@router.post("/health/metrics/reset")
def reset_metrics(
    prediction_service: PredictionService = Depends(prediction_service_dependency),
) -> dict[str, str]:
    """Reset prediction metrics.

    Returns:
        Confirmation message
    """
    prediction_service.reset_metrics()
    return {"status": "ok", "message": "Metrics reset"}


@router.get("/health")
def health() -> dict[str, Any]:
    """Simple health check endpoint.

    Returns:
        Basic health status
    """
    return {
        "status": "ok",
        "message": "API is running",
        "timestamp": time.time(),
    }
