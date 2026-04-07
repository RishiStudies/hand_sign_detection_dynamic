"""Rate limiting service.

Provides request rate limiting with Redis or in-memory fallback.
"""

import logging
import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import HTTPException

from hand_sign_detection.core.config import get_settings
from hand_sign_detection.core.redis import get_redis_client

logger = logging.getLogger("hand_sign_detection.rate_limiting")


class RateLimiter:
    """Rate limiter with Redis or in-memory backend.

    Supports per-bucket rate limiting with configurable windows.
    Automatically falls back to in-memory if Redis is unavailable.

    Usage:
        limiter = RateLimiter()
        limiter.check_limit("predict", client_key)  # Raises HTTPException if exceeded
    """

    def __init__(self):
        self._lock = Lock()
        self._store: dict = defaultdict(deque)
        self._settings = get_settings()

    @property
    def backend(self) -> str:
        """Get current backend type."""
        return "redis" if get_redis_client() is not None else "in_memory"

    def check_limit(
        self,
        bucket: str,
        key: str,
        max_requests: int | None = None,
        window_seconds: int | None = None,
    ) -> None:
        """Check rate limit and raise exception if exceeded.

        Args:
            bucket: Rate limit bucket name (e.g., "predict", "train")
            key: Client identity key
            max_requests: Max requests per window (uses settings default if None)
            window_seconds: Window duration in seconds (uses settings default if None)

        Raises:
            HTTPException: 429 if rate limit exceeded
        """
        if max_requests is None:
            max_requests = self._get_default_limit(bucket)
        if window_seconds is None:
            window_seconds = self._settings.rate_limit_window_seconds

        redis_client = get_redis_client()
        if redis_client is not None:
            try:
                self._check_limit_redis(redis_client, bucket, key, max_requests, window_seconds)
                return
            except HTTPException:
                raise
            except Exception as exc:
                logger.warning("Redis rate limit failed, using in-memory: %s", exc)

        self._check_limit_memory(bucket, key, max_requests, window_seconds)

    def _get_default_limit(self, bucket: str) -> int:
        """Get default rate limit for a bucket."""
        limits = {
            "predict": self._settings.max_predict_requests_per_window,
            "predict_sequence": self._settings.max_sequence_requests_per_window,
            "train": self._settings.max_train_requests_per_window,
        }
        return limits.get(bucket, 100)

    def _check_limit_memory(
        self,
        bucket: str,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> None:
        """Check rate limit using in-memory store."""
        now = time.time()
        composite_key = f"{bucket}:{key}"

        with self._lock:
            request_times = self._store[composite_key]

            # Remove expired entries
            while request_times and (now - request_times[0]) > window_seconds:
                request_times.popleft()

            if len(request_times) >= max_requests:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(window_seconds)},
                )

            request_times.append(now)

    def _check_limit_redis(
        self,
        redis_client,
        bucket: str,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> None:
        """Check rate limit using Redis."""
        redis_key = f"{self._settings.redis_rate_limit_prefix}:{bucket}:{key}"

        count = int(redis_client.incr(redis_key))
        if count == 1:
            redis_client.expire(redis_key, window_seconds)

        if count > max_requests:
            ttl = redis_client.ttl(redis_key)
            retry_after = str(ttl if isinstance(ttl, int) and ttl > 0 else window_seconds)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": retry_after},
            )

    def reset(self) -> None:
        """Reset in-memory rate limit store (for testing)."""
        with self._lock:
            self._store.clear()


# Module-level singleton
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get or create RateLimiter singleton."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def reset_rate_limiter() -> None:
    """Reset rate limiter singleton (for testing)."""
    global _rate_limiter
    _rate_limiter = None
