import importlib
import os
from typing import Any, Dict

from .training_module.jobs import JOB_HANDLERS


REDIS_URL = os.getenv("REDIS_URL", "").strip()
JOB_QUEUE_NAME = os.getenv("JOB_QUEUE_NAME", "training")
JOB_TIMEOUT_SECONDS = int(os.getenv("JOB_TIMEOUT_SECONDS", "7200"))


def _require_redis_url() -> str:
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL is required for queued training jobs")
    return REDIS_URL


def _get_redis_connection():
    redis_module = importlib.import_module("redis")
    return redis_module.from_url(_require_redis_url(), decode_responses=True)


def _get_queue():
    rq_module = importlib.import_module("rq")
    return rq_module.Queue(
        JOB_QUEUE_NAME,
        connection=_get_redis_connection(),
        default_timeout=JOB_TIMEOUT_SECONDS,
    )


def is_job_queue_available() -> bool:
    if not REDIS_URL:
        return False
    try:
        _get_redis_connection().ping()
        importlib.import_module("rq")
        return True
    except Exception:
        return False


def enqueue_named_job(job_name: str, **kwargs):
    queue = _get_queue()
    handler = JOB_HANDLERS.get(job_name)
    if handler is None:
        raise ValueError(f"Unsupported job name: {job_name}")
    return queue.enqueue(handler, kwargs=kwargs)


def get_job_status(job_id: str) -> Dict[str, Any]:
    rq_job_module = importlib.import_module("rq.job")
    job = rq_job_module.Job.fetch(job_id, connection=_get_redis_connection())
    status = job.get_status()
    result = job.result if status == "finished" else None
    error = None
    if job.exc_info:
        error = "\n".join(job.exc_info.strip().splitlines()[-10:])

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

