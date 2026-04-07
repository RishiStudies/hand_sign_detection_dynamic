"""Combo detection service.

Detects gesture combinations/sequences from prediction history.
Supports Redis or in-memory backend for session state.
"""

import json
import logging
import time
from collections import deque
from threading import Lock
from typing import Any

from hand_sign_detection.core.config import get_settings
from hand_sign_detection.core.redis import get_redis_client

logger = logging.getLogger("hand_sign_detection.combo_detection")


# Default combo patterns
DEFAULT_COMBOS = {
    "HELLO_WORLD": ["HELLO", "WORLD"],
    "THANK_YOU": ["THANK", "YOU"],
    "GOOD_MORNING": ["GOOD", "MORNING"],
    "HOW_ARE_YOU": ["HOW", "ARE", "YOU"],
    "I_LOVE_YOU": ["I", "LOVE", "YOU"],
    "PLEASE": ["PLEASE"],
    "SORRY": ["SORRY"],
    "YES_NO": ["YES", "NO"],
    "ABC": ["A", "B", "C"],
    "COUNTING": ["ONE", "TWO", "THREE"],
}


class ComboDetector:
    """Detects gesture combinations from a prediction buffer.

    Maintains a buffer of recent predictions and checks for combo patterns.
    """

    def __init__(self, combos: dict[str, list[str]] | None = None):
        """Initialize combo detector.

        Args:
            combos: Custom combo patterns (uses defaults if None)
        """
        self.combos = combos or dict(DEFAULT_COMBOS)
        self.prediction_buffer: deque = deque(maxlen=10)
        self.buffer_timeout = 5.0  # seconds

    def add_prediction(
        self,
        gesture: str,
        confidence: float,
        model_type: str = "rf",
    ) -> None:
        """Add a prediction to the buffer.

        Args:
            gesture: Predicted gesture label
            confidence: Prediction confidence
            model_type: Model type ("rf" or "lstm")
        """
        self.prediction_buffer.append(
            {
                "gesture": gesture,
                "confidence": confidence,
                "timestamp": time.time(),
                "model": model_type,
            }
        )

    def check_combos(self, min_confidence: float = 0.7) -> dict[str, Any] | None:
        """Check for matching combo patterns.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            Dict with combo info if matched, None otherwise
        """
        if len(self.prediction_buffer) < 2:
            return None

        current_time = time.time()
        recent_predictions = [
            p
            for p in self.prediction_buffer
            if current_time - p["timestamp"] <= self.buffer_timeout
            and p["confidence"] >= min_confidence
        ]

        if len(recent_predictions) < 2:
            return None

        gesture_sequence = [p["gesture"] for p in recent_predictions]

        for combo_name, combo_sequence in self.combos.items():
            if self._matches_combo(gesture_sequence, combo_sequence):
                combo_predictions = recent_predictions[-len(combo_sequence) :]
                avg_confidence = sum(p["confidence"] for p in combo_predictions) / len(
                    combo_predictions
                )

                return {
                    "combo": combo_name,
                    "sequence": combo_sequence,
                    "confidence": avg_confidence,
                    "timestamp": current_time,
                }

        return None

    def _matches_combo(
        self,
        gesture_sequence: list[str],
        combo_sequence: list[str],
    ) -> bool:
        """Check if gesture sequence ends with combo pattern."""
        if len(gesture_sequence) < len(combo_sequence):
            return False
        recent_gestures = gesture_sequence[-len(combo_sequence) :]
        return recent_gestures == combo_sequence

    def clear(self) -> None:
        """Clear the prediction buffer."""
        self.prediction_buffer.clear()

    def get_available_combos(self) -> list[str]:
        """Get list of available combo names."""
        return list(self.combos.keys())


class ComboService:
    """Service for managing combo detection across sessions.

    Provides session-based combo detection with Redis or in-memory backend.
    """

    def __init__(self):
        self._lock = Lock()
        self._detectors: dict[str, ComboDetector] = {}
        self._settings = get_settings()
        self._catalog = ComboDetector()  # For getting available combos

    @property
    def backend(self) -> str:
        """Get current backend type."""
        return "redis" if get_redis_client() is not None else "in_memory"

    def get_available_combos(self) -> dict[str, Any]:
        """Get catalog of available combos.

        Returns:
            Dict with combo names and patterns
        """
        return {
            "combos": self._catalog.get_available_combos(),
            "patterns": self._catalog.combos,
        }

    def add_prediction(
        self,
        session_id: str | None,
        gesture: str,
        confidence: float,
        model_type: str = "rf",
    ) -> None:
        """Add prediction for a session.

        Args:
            session_id: Client session ID
            gesture: Predicted gesture
            confidence: Prediction confidence
            model_type: Model type
        """
        prediction = {
            "gesture": gesture,
            "confidence": confidence,
            "timestamp": time.time(),
            "model": model_type,
        }

        redis_client = get_redis_client()
        if redis_client is not None:
            try:
                self._add_prediction_redis(redis_client, session_id, prediction)
                return
            except (ConnectionError, TimeoutError, OSError) as exc:
                logger.warning("Redis combo write failed (connection): %s", exc)
            except ValueError as exc:
                logger.warning("Redis combo write failed (serialization): %s", exc)

        self._add_prediction_memory(session_id, prediction)

    def check_combos(
        self,
        session_id: str | None,
        min_confidence: float = 0.7,
    ) -> dict[str, Any] | None:
        """Check for combos in a session.

        Args:
            session_id: Client session ID
            min_confidence: Minimum confidence threshold

        Returns:
            Combo info if matched, None otherwise
        """
        predictions = self._load_predictions(session_id)
        if len(predictions) < 2:
            return None

        detector = ComboDetector()
        detector.prediction_buffer.extend(predictions)
        return detector.check_combos(min_confidence=min_confidence)

    def clear_session(self, session_id: str | None) -> None:
        """Clear combo state for a session.

        Args:
            session_id: Client session ID
        """
        redis_client = get_redis_client()
        if redis_client is not None:
            try:
                redis_client.delete(self._get_redis_key(session_id))
                return
            except (ConnectionError, TimeoutError, OSError) as exc:
                logger.warning("Redis combo delete failed (connection): %s", exc)

        session_key = self._get_session_key(session_id)
        with self._lock:
            if session_key in self._detectors:
                self._detectors[session_key].clear()

    def _get_session_key(self, session_id: str | None) -> str:
        """Get normalized session key."""
        return (session_id or "anonymous").strip() or "anonymous"

    def _get_redis_key(self, session_id: str | None) -> str:
        """Get Redis key for session."""
        session_key = self._get_session_key(session_id)
        return f"{self._settings.redis_combo_prefix}:{session_key}"

    def _get_detector(self, session_id: str | None) -> ComboDetector:
        """Get or create detector for session."""
        session_key = self._get_session_key(session_id)
        with self._lock:
            if session_key not in self._detectors:
                self._detectors[session_key] = ComboDetector()
            return self._detectors[session_key]

    def _add_prediction_memory(
        self,
        session_id: str | None,
        prediction: dict[str, Any],
    ) -> None:
        """Add prediction to in-memory detector."""
        detector = self._get_detector(session_id)
        detector.prediction_buffer.append(prediction)

    def _add_prediction_redis(
        self,
        redis_client,
        session_id: str | None,
        prediction: dict[str, Any],
    ) -> None:
        """Add prediction to Redis."""
        redis_key = self._get_redis_key(session_id)
        redis_client.rpush(redis_key, json.dumps(prediction))
        redis_client.ltrim(redis_key, -10, -1)  # Keep last 10
        redis_client.expire(redis_key, self._settings.combo_state_ttl_seconds)

    def _load_predictions(
        self,
        session_id: str | None,
    ) -> list[dict[str, Any]]:
        """Load predictions for session."""
        redis_client = get_redis_client()
        if redis_client is not None:
            try:
                serialized = redis_client.lrange(self._get_redis_key(session_id), 0, -1)
                return [json.loads(item) for item in serialized]
            except (ConnectionError, TimeoutError, OSError) as exc:
                logger.warning("Redis combo read failed (connection): %s", exc)
            except (ValueError, json.JSONDecodeError) as exc:
                logger.warning("Redis combo read failed (parsing): %s", exc)

        detector = self._get_detector(session_id)
        return list(detector.prediction_buffer)

    def reset(self) -> None:
        """Reset all in-memory state (for testing)."""
        with self._lock:
            self._detectors.clear()


# Module-level singleton
_combo_service: ComboService | None = None


def get_combo_service() -> ComboService:
    """Get or create ComboService singleton."""
    global _combo_service
    if _combo_service is None:
        _combo_service = ComboService()
    return _combo_service


def reset_combo_service() -> None:
    """Reset combo service singleton (for testing)."""
    global _combo_service
    _combo_service = None
