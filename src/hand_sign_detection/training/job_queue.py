"""Job queue management for background training tasks.

Provides RQ (Redis Queue) integration for async training.
"""

import importlib
import logging
from typing import Any

from hand_sign_detection.core.config import get_settings
from hand_sign_detection.core.redis import get_redis_client
from hand_sign_detection.training.jobs import JOB_HANDLERS

logger = logging.getLogger("hand_sign_detection.training.job_queue")

# Queue name constant for worker compatibility
JOB_QUEUE_NAME = get_settings().job_queue_name


def _get_redis_connection():
    """Get Redis connection for RQ worker.

    Returns:
        Redis connection instance

    Raises:
        RuntimeError: If Redis is not configured
    """
    redis_client = get_redis_client()
    if redis_client is None:
        raise RuntimeError("Redis connection failed - check REDIS_URL")
    return redis_client


def is_job_queue_available() -> bool:
    """Check if job queue (Redis + RQ) is available.

    Returns:
        True if job queue is operational
    """
    settings = get_settings()

    if not settings.redis_url:
        logger.debug("REDIS_URL not configured - job queue unavailable")
        return False

    try:
        redis_client = get_redis_client()
        if redis_client is None:
            return False

        redis_client.ping()
        importlib.import_module("rq")
        logger.info("Job queue is available and operational")
        return True
    except Exception as e:
        logger.warning("Job queue unavailable: %s: %s", type(e).__name__, e)
        return False


def _get_queue():
    """Get RQ queue instance.

    Returns:
        RQ Queue instance

    Raises:
        RuntimeError: If Redis not available
    """
    settings = get_settings()

    if not settings.redis_url:
        raise RuntimeError("REDIS_URL is required for queued training jobs")

    redis_client = get_redis_client()
    if redis_client is None:
        raise RuntimeError("Redis connection failed")

    rq_module = importlib.import_module("rq")
    queue = rq_module.Queue(
        settings.job_queue_name,
        connection=redis_client,
        default_timeout=settings.job_timeout_seconds,
    )

    logger.debug(
        "Job queue '%s' obtained with timeout=%ds",
        settings.job_queue_name,
        settings.job_timeout_seconds,
    )

    return queue


def enqueue_named_job(job_name: str, **kwargs):
    """Enqueue a named training job.

    Args:
        job_name: Name of the job (must be in JOB_HANDLERS)
        **kwargs: Arguments to pass to the job handler

    Returns:
        RQ Job instance

    Raises:
        ValueError: If job_name is not recognized
    """
    logger.info("Enqueuing job: %s with args %s", job_name, list(kwargs.keys()))

    handler = JOB_HANDLERS.get(job_name)
    if handler is None:
        logger.error(
            "Unknown job name: %s. Available jobs: %s",
            job_name,
            list(JOB_HANDLERS.keys()),
        )
        raise ValueError(f"Unsupported job name: {job_name}")

    queue = _get_queue()
    job = queue.enqueue(handler, kwargs=kwargs)

    logger.info("Job enqueued successfully: %s (job_id=%s)", job_name, job.id)
    return job


def get_job_status(job_id: str) -> dict[str, Any]:
    """Get status of a job by ID.

    Args:
        job_id: Job identifier

    Returns:
        Job status dictionary
    """
    logger.debug("Fetching status for job_id=%s", job_id)

    redis_client = get_redis_client()
    if redis_client is None:
        raise RuntimeError("Redis connection failed")

    rq_job_module = importlib.import_module("rq.job")
    job = rq_job_module.Job.fetch(job_id, connection=redis_client)

    status = job.get_status()
    result = job.result if status == "finished" else None

    error = None
    if job.exc_info:
        error = "\n".join(job.exc_info.strip().splitlines()[-10:])

    logger.debug(
        "Job %s status=%s (created=%s, started=%s)",
        job_id,
        status,
        job.created_at,
        job.started_at,
    )

    if status == "failed":
        logger.warning("Job %s failed with error: %s", job_id, error)
    elif status == "finished":
        logger.info("Job %s completed successfully", job_id)

    return {
        "id": job.id,
        "status": status,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "ended_at": job.ended_at.isoformat() if job.ended_at else None,
        "description": job.description,
        "result": result,
        "error": error,
    }
