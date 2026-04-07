"""Training endpoints."""

import contextlib
import json
import logging
import os
import uuid
from threading import Lock

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile

from hand_sign_detection.api.dependencies import (
    get_client_identity,
    rate_limiter_dependency,
    require_training_key,
    settings_dependency,
)
from hand_sign_detection.core.config import Settings
from hand_sign_detection.core.shared_state import load_shared_state
from hand_sign_detection.services.rate_limiting import RateLimiter

logger = logging.getLogger("hand_sign_detection.api.training")

router = APIRouter(tags=["Training"])

# Training lock to prevent concurrent training
_training_lock = Lock()


def _acquire_training_slot() -> None:
    """Acquire training slot."""
    if not _training_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Training already in progress")


def _release_training_slot() -> None:
    """Release training slot."""
    with contextlib.suppress(RuntimeError):
        _training_lock.release()


def _get_job_queue():
    """Get job queue module (lazy import)."""
    try:
        from hand_sign_detection.training.job_queue import (
            enqueue_named_job,
            get_job_status,
            is_job_queue_available,
        )

        return enqueue_named_job, get_job_status, is_job_queue_available
    except ImportError:
        return None, None, lambda: False


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


def _persist_sample_training_inputs(
    samples: list[UploadFile],
    sample_payloads: list[bytes],
    labels_input: list[str],
    settings: Settings,
) -> str:
    """Persist training samples to disk for background job."""
    job_inputs_dir = os.path.join(settings.data_dir, "job_inputs")
    job_dir = os.path.join(job_inputs_dir, f"train-{uuid.uuid4().hex}")
    os.makedirs(job_dir, exist_ok=True)

    manifest_samples = []
    for index, (_sample, data, label) in enumerate(
        zip(samples, sample_payloads, labels_input)
    ):
        file_path = os.path.join(job_dir, f"sample_{index:04d}.jpg")
        with open(file_path, "wb") as file_obj:
            file_obj.write(data)
        manifest_samples.append({"label": label, "file_path": file_path})

    manifest_path = os.path.join(job_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file_obj:
        json.dump({"samples": manifest_samples}, file_obj)

    return manifest_path


def _persist_csv_training_input(data: bytes, settings: Settings) -> str:
    """Persist CSV data to disk for background job."""
    job_inputs_dir = os.path.join(settings.data_dir, "job_inputs")
    os.makedirs(job_inputs_dir, exist_ok=True)
    csv_path = os.path.join(job_inputs_dir, f"train_csv_{uuid.uuid4().hex}.csv")
    with open(csv_path, "wb") as file_obj:
        file_obj.write(data)
    return csv_path


def _validate_csv_schema(csv_path: str, max_rows_to_check: int = 100) -> None:
    """Validate CSV structure before training."""
    try:
        df = pd.read_csv(csv_path, nrows=max_rows_to_check)

        if df.empty:
            raise ValueError("CSV file is empty")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            raise ValueError("CSV has no numeric feature columns")

        logger.info(
            "CSV validation passed: %d numeric features, %d non-numeric columns",
            len(numeric_cols),
            len(non_numeric_cols),
        )

    except pd.errors.ParserError as exc:
        logger.warning("CSV parsing failed: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CSV format: {str(exc)}",
        ) from exc
    except ValueError as exc:
        logger.warning("CSV structure validation failed: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"CSV validation failed: {str(exc)}",
        ) from exc
    except OSError as exc:
        logger.error("CSV file I/O error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"CSV file error: {str(exc)}",
        ) from exc


@router.get("/artifacts")
def get_artifacts():
    """Get shared artifact state.

    Returns:
        Current shared state configuration
    """
    return load_shared_state()


@router.get("/training")
def training_info(
    settings: Settings = Depends(settings_dependency),
):
    """Get training UI information.

    Returns:
        Frontend URL for training UI
    """
    return {
        "message": "Use the Next.js frontend for training UI",
        "frontend_url": settings.frontend_url,
    }


@router.get("/jobs/{job_id}")
def get_training_job(
    job_id: str,
):
    """Get status of a training job.

    Args:
        job_id: Job identifier

    Returns:
        Job status and result if complete
    """
    enqueue_named_job, get_job_status, is_job_queue_available = _get_job_queue()

    if not is_job_queue_available():
        raise HTTPException(status_code=503, detail="Job queue unavailable")

    try:
        job_state = get_job_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc
    except (ConnectionError, TimeoutError) as exc:
        logger.error("Job queue connection error: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Job queue unavailable") from exc

    # Reload models if training completed
    result = job_state.get("result")
    if job_state.get("status") == "finished" and isinstance(result, dict):
        from hand_sign_detection.models.manager import get_model_manager

        manager = get_model_manager()

        job_name = result.get("job_name")
        if job_name == "train_lstm":
            manager.load_lstm_model()
        elif job_name in {"train_rf_samples", "train_rf_csv"}:
            manager.load_rf_model()

    return job_state


@router.post("/train", status_code=202)
async def train(
    request: Request,
    samples: list[UploadFile] = File(...),
    labels_input: list[str] = Form(...),
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(settings_dependency),
    rate_limiter: RateLimiter = Depends(rate_limiter_dependency),
    _: None = Depends(require_training_key),
):
    """Train RandomForest model with image samples.

    Args:
        samples: List of training images
        labels_input: Labels for each sample
        x_api_key: Training API key

    Returns:
        Job info with status and job_id
    """
    enqueue_named_job, get_job_status, is_job_queue_available = _get_job_queue()

    rate_limiter.check_limit("train", get_client_identity(request))

    if not is_job_queue_available():
        raise HTTPException(status_code=503, detail="Job queue unavailable")

    if len(samples) == 0:
        raise HTTPException(status_code=400, detail="No samples provided")

    if len(samples) != len(labels_input):
        raise HTTPException(
            status_code=400,
            detail="samples and labels_input size mismatch",
        )

    sample_payloads = []
    for i, (sample, label) in enumerate(zip(samples, labels_input)):
        if not label or len(label.strip()) == 0 or len(label) > 64:
            logger.warning("Invalid label rejected: %s", label)
            raise HTTPException(status_code=400, detail=f"Invalid label at index {i}")
        data = await sample.read()
        _validate_upload(sample, data, settings, field_name="sample")
        sample_payloads.append(data)

    manifest_path = _persist_sample_training_inputs(
        samples, sample_payloads, labels_input, settings
    )

    try:
        job = enqueue_named_job("train_rf_samples", manifest_path=manifest_path)
        logger.info("Sample training job queued: %s with %d samples", job.id, len(samples))
    except (ConnectionError, TimeoutError) as exc:
        logger.error("Job queue connection error: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Job queue unavailable") from exc
    except ValueError as exc:
        logger.error("Invalid job parameters: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "queued",
        "job_id": job.id,
        "job_type": "train_rf_samples",
    }


@router.post("/train_csv", status_code=202)
async def train_csv(
    request: Request,
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(settings_dependency),
    rate_limiter: RateLimiter = Depends(rate_limiter_dependency),
    _: None = Depends(require_training_key),
):
    """Train RandomForest model from CSV data.

    Args:
        file: CSV file with features and labels
        x_api_key: Training API key

    Returns:
        Job info with status and job_id
    """
    enqueue_named_job, get_job_status, is_job_queue_available = _get_job_queue()

    rate_limiter.check_limit("train", get_client_identity(request))

    if not is_job_queue_available():
        raise HTTPException(status_code=503, detail="Job queue unavailable")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    if len(data) > settings.max_csv_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"CSV file exceeds max size {settings.max_csv_upload_bytes} bytes",
        )
    if file.content_type and "csv" not in file.content_type.lower():
        logger.warning("Rejected CSV upload with wrong content type: %s", file.content_type)
        raise HTTPException(status_code=415, detail="Expected CSV content type")

    csv_path = _persist_csv_training_input(data, settings)
    _validate_csv_schema(csv_path)

    try:
        job = enqueue_named_job("train_rf_csv", csv_path=csv_path)
        logger.info("CSV training job queued: %s", job.id)
    except (ConnectionError, TimeoutError) as exc:
        logger.error("Job queue connection error: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Job queue unavailable") from exc
    except ValueError as exc:
        logger.error("Invalid job parameters: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "queued",
        "job_id": job.id,
        "job_type": "train_rf_csv",
    }


@router.post("/process_wlasl", status_code=202)
async def process_wlasl(
    request: Request,
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(settings_dependency),
    rate_limiter: RateLimiter = Depends(rate_limiter_dependency),
    _: None = Depends(require_training_key),
):
    """Process WLASL videos for LSTM training.

    Args:
        x_api_key: Training API key

    Returns:
        Job info with status and job_id
    """
    enqueue_named_job, get_job_status, is_job_queue_available = _get_job_queue()

    rate_limiter.check_limit("train", get_client_identity(request))

    if not is_job_queue_available():
        raise HTTPException(status_code=503, detail="Job queue unavailable")

    try:
        job = enqueue_named_job("process_wlasl")
    except (ConnectionError, TimeoutError) as exc:
        logger.error("Job queue connection error: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Job queue unavailable") from exc

    return {
        "status": "queued",
        "job_id": job.id,
        "job_type": "process_wlasl",
    }


@router.post("/train_lstm", status_code=202)
async def train_lstm(
    request: Request,
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(settings_dependency),
    rate_limiter: RateLimiter = Depends(rate_limiter_dependency),
    _: None = Depends(require_training_key),
):
    """Train LSTM model on preprocessed data.

    Args:
        x_api_key: Training API key

    Returns:
        Job info with status and job_id
    """
    enqueue_named_job, get_job_status, is_job_queue_available = _get_job_queue()

    rate_limiter.check_limit("train", get_client_identity(request))

    if not is_job_queue_available():
        raise HTTPException(status_code=503, detail="Job queue unavailable")

    try:
        job = enqueue_named_job("train_lstm")
    except (ConnectionError, TimeoutError) as exc:
        logger.error("Job queue connection error: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Job queue unavailable") from exc

    return {
        "status": "queued",
        "job_id": job.id,
        "job_type": "train_lstm",
    }
