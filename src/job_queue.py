import importlib
import logging
import os
from typing import Any, Dict

from .training_module.jobs import JOB_HANDLERS


logger = logging.getLogger("job_queue")

REDIS_URL = os.getenv("REDIS_URL", "").strip()
JOB_QUEUE_NAME = os.getenv("JOB_QUEUE_NAME", "training")
JOB_TIMEOUT_SECONDS = int(os.getenv("JOB_TIMEOUT_SECONDS", "7200"))


def _require_redis_url() -> str:
    if not REDIS_URL:
        logger.error("REDIS_URL environment variable is not set")
        raise RuntimeError("REDIS_URL is required for queued training jobs")
    logger.debug(f"REDIS_URL is configured (length={len(REDIS_URL)})")
    return REDIS_URL


def _get_redis_connection():
    """Get Redis connection with logging."""
    try:
        redis_module = importlib.import_module("redis")
        connection = redis_module.from_url(_require_redis_url(), decode_responses=True)
        logger.debug("Redis connection established successfully")
        return connection
    except ImportError as e:
        logger.error(f"Failed to import redis module: {e}")
        raise RuntimeError("redis package is required but not installed") from e
    except Exception as e:
        logger.error(f"Failed to establish Redis connection: {e}", exc_info=True)
        raise


def _get_queue():
    """Get RQ queue with logging."""
    try:
        rq_module = importlib.import_module("rq")
        queue = rq_module.Queue(
            JOB_QUEUE_NAME,
            connection=_get_redis_connection(),
            default_timeout=JOB_TIMEOUT_SECONDS,
        )
        logger.debug(f"Job queue '{JOB_QUEUE_NAME}' obtained with timeout={JOB_TIMEOUT_SECONDS}s")
        return queue
    except Exception as e:
        logger.error(f"Failed to obtain job queue: {e}", exc_info=True)
        raise


def is_job_queue_available() -> bool:
    """Check if job queue (Redis + RQ) is available with logging."""
    if not REDIS_URL:
        logger.debug("REDIS_URL not configured - job queue unavailable")
        return False
    try:
        connection = _get_redis_connection()
        connection.ping()
        importlib.import_module("rq")
        logger.info("Job queue is available and operational")
        return True
    except Exception as e:
        logger.warning(f"Job queue unavailable: {type(e).__name__}: {e}")
        return False


def enqueue_named_job(job_name: str, **kwargs):
    """Enqueue a named training job with comprehensive logging."""
    try:
        logger.info(f"Enqueuing job: {job_name} with args {list(kwargs.keys())}")
        queue = _get_queue()
        handler = JOB_HANDLERS.get(job_name)
        if handler is None:
            logger.error(f"Unknown job name: {job_name}. Available jobs: {list(JOB_HANDLERS.keys())}")
            raise ValueError(f"Unsupported job name: {job_name}")
        
        job = queue.enqueue(handler, kwargs=kwargs)
        logger.info(f"Job enqueued successfully: {job_name} (job_id={job.id})")
        return job
    except Exception as e:
        logger.error(f"Failed to enqueue job '{job_name}': {e}", exc_info=True)
        raise


def get_job_status(job_id: str) -> Dict[str, Any]:
    """Get job status with comprehensive logging."""
    try:
        logger.debug(f"Fetching status for job_id={job_id}")
        rq_job_module = importlib.import_module("rq.job")
        job = rq_job_module.Job.fetch(job_id, connection=_get_redis_connection())
        status = job.get_status()
        result = job.result if status == "finished" else None
        error = None
        if job.exc_info:
            error = "\n".join(job.exc_info.strip().splitlines()[-10:])
        
        logger.debug(f"Job {job_id} status={status} (created={job.created_at}, started={job.started_at})")
        if status == "failed":
            logger.warning(f"Job {job_id} failed with error: {error}")
        elif status == "finished":
            logger.info(f"Job {job_id} completed successfully")

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
    except Exception as e:
        logger.error(f"Failed to fetch job status for {job_id}: {e}", exc_info=True)
        raise

