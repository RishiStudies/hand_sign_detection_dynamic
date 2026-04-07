"""Prediction endpoints."""

import logging
from threading import Lock

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile

from hand_sign_detection.api.dependencies import (
    get_client_identity,
    prediction_service_dependency,
    rate_limiter_dependency,
    settings_dependency,
)
from hand_sign_detection.core.config import Settings
from hand_sign_detection.services.prediction import PredictionService
from hand_sign_detection.services.rate_limiting import RateLimiter

logger = logging.getLogger("hand_sign_detection.api.predict")

router = APIRouter(tags=["Prediction"])

# Concurrency control for sequence inference
_sequence_inflight_lock = Lock()
_sequence_inflight_count = 0


def _validate_upload(
    file: UploadFile,
    data: bytes,
    settings: Settings,
    field_name: str = "file",
) -> None:
    """Validate uploaded file."""
    if file.content_type and file.content_type.lower() not in settings.allowed_image_content_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type for {field_name}: {file.content_type}",
        )
    if not data:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{field_name} exceeds max size {settings.max_upload_bytes} bytes",
        )


def _acquire_sequence_slot(settings: Settings) -> None:
    """Acquire slot for sequence inference."""
    global _sequence_inflight_count
    with _sequence_inflight_lock:
        if _sequence_inflight_count >= settings.max_concurrent_sequence_requests:
            raise HTTPException(
                status_code=503,
                detail="Sequence inference busy. Retry shortly.",
                headers={"Retry-After": "2"},
            )
        _sequence_inflight_count += 1


def _release_sequence_slot() -> None:
    """Release sequence inference slot."""
    global _sequence_inflight_count
    with _sequence_inflight_lock:
        if _sequence_inflight_count > 0:
            _sequence_inflight_count -= 1


@router.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    x_session_id: str | None = Header(default=None),
    settings: Settings = Depends(settings_dependency),
    rate_limiter: RateLimiter = Depends(rate_limiter_dependency),
    prediction_service: PredictionService = Depends(prediction_service_dependency),
):
    """Predict gesture from a single image.

    Args:
        file: Image file (JPEG, PNG)
        x_session_id: Optional session ID for combo tracking

    Returns:
        Prediction result with label, confidence, and optional combo
    """
    client_key = get_client_identity(request, x_session_id)
    rate_limiter.check_limit("predict", client_key)

    data = await file.read()
    _validate_upload(file, data, settings)

    return prediction_service.predict_single(
        image_data=data,
        session_id=x_session_id,
    )


@router.post("/predict_sequence")
async def predict_sequence(
    request: Request,
    files: list[UploadFile] = File(...),
    x_session_id: str | None = Header(default=None),
    settings: Settings = Depends(settings_dependency),
    rate_limiter: RateLimiter = Depends(rate_limiter_dependency),
    prediction_service: PredictionService = Depends(prediction_service_dependency),
):
    """Predict gesture from a sequence of frames using LSTM.

    Args:
        files: List of image files (must match max_sequence_frames)
        x_session_id: Optional session ID for combo tracking

    Returns:
        Prediction result with label, confidence, and optional combo
    """
    client_key = get_client_identity(request, x_session_id)
    rate_limiter.check_limit("predict_sequence", client_key)

    _acquire_sequence_slot(settings)

    try:
        if len(files) != settings.max_sequence_frames:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {settings.max_sequence_frames} frames, got {len(files)}",
            )

        frames_data = []
        for i, file in enumerate(files):
            data = await file.read()
            _validate_upload(file, data, settings, field_name=f"files[{i}]")
            frames_data.append(data)

        return prediction_service.predict_sequence(
            frames_data=frames_data,
            session_id=x_session_id,
        )
    finally:
        _release_sequence_slot()
