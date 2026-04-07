"""Prediction service.

Handles inference requests and coordinates with models and combos.
Includes optimizations for fast, accurate predictions.
"""

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException

from hand_sign_detection.core.config import get_settings
from hand_sign_detection.models.features import (
    extract_features_with_confidence,
    preprocess_frame,
    validate_feature_contract,
)
from hand_sign_detection.models.manager import ModelManager, get_model_manager
from hand_sign_detection.services.combo_detection import ComboService, get_combo_service

logger = logging.getLogger("hand_sign_detection.prediction")


@dataclass
class PredictionMetrics:
    """Metrics for prediction performance monitoring."""

    total_predictions: int = 0
    total_latency_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    low_confidence_count: int = 0
    no_hand_detected_count: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.total_latency_ms / self.total_predictions

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_predictions": self.total_predictions,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "low_confidence_count": self.low_confidence_count,
            "no_hand_detected_count": self.no_hand_detected_count,
        }


class LRUCache:
    """Thread-safe LRU cache for prediction results."""

    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._lock = Lock()

    def get(self, key: str) -> dict | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def set(self, key: str, value: dict) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class PredictionService:
    """Service for handling prediction requests.

    Coordinates between model inference and combo detection.
    Includes optimizations:
    - Feature caching for repeated frames
    - Batch processing support
    - Performance metrics tracking
    - Confidence-based filtering
    """

    LOW_CONFIDENCE_THRESHOLD = 0.3
    CACHE_ENABLED = True

    def __init__(
        self,
        model_manager: ModelManager | None = None,
        combo_service: ComboService | None = None,
        enable_cache: bool = True,
        cache_size: int = 100,
    ):
        """Initialize prediction service.

        Args:
            model_manager: ModelManager instance (uses singleton if None)
            combo_service: ComboService instance (uses singleton if None)
            enable_cache: Whether to enable prediction caching
            cache_size: Maximum cache entries
        """
        self._model_manager = model_manager
        self._combo_service = combo_service
        self._settings = get_settings()
        self._metrics = PredictionMetrics()
        self._cache = LRUCache(cache_size) if enable_cache else None

    @property
    def model_manager(self) -> ModelManager:
        """Get model manager instance."""
        if self._model_manager is None:
            self._model_manager = get_model_manager()
        return self._model_manager

    @property
    def combo_service(self) -> ComboService:
        """Get combo service instance."""
        if self._combo_service is None:
            self._combo_service = get_combo_service()
        return self._combo_service

    @property
    def metrics(self) -> PredictionMetrics:
        """Get current metrics."""
        return self._metrics

    def _compute_image_hash(self, data: bytes) -> str:
        """Compute hash for cache key using blake2b (faster, non-cryptographic)."""
        return hashlib.blake2b(data, digest_size=16).hexdigest()

    def predict_single(
        self,
        image_data: bytes,
        session_id: str | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Predict gesture from single image.

        Args:
            image_data: Image bytes (JPEG, PNG, etc.)
            session_id: Client session ID for combo tracking
            use_cache: Whether to use prediction cache

        Returns:
            Dict with prediction results

        Raises:
            HTTPException: 503 if model unavailable, 400 if invalid image
        """
        start_time = time.perf_counter()

        if not self.model_manager.rf_available:
            raise HTTPException(status_code=503, detail="RandomForest model unavailable")

        # Check cache
        cache_key = None
        if self._cache and use_cache:
            cache_key = self._compute_image_hash(image_data)
            cached = self._cache.get(cache_key)
            if cached:
                self._metrics.cache_hits += 1
                # Still track for combos even with cached result
                self.combo_service.add_prediction(
                    session_id=session_id,
                    gesture=cached["label"],
                    confidence=cached["prob"],
                    model_type="rf",
                )
                return cached
            self._metrics.cache_misses += 1

        features, confidence, hand_detected = self._extract_features_with_confidence(image_data)

        if not hand_detected:
            self._metrics.no_hand_detected_count += 1

        # Validate feature dimensions
        validate_feature_contract(
            actual_dimension=int(features.shape[1]),
            expected_dimension=self.model_manager.rf_n_features,
            context="Inference request",
        )

        predicted_label, pred_confidence = self.model_manager.predict_rf(features)

        # Track low confidence
        if pred_confidence < self.LOW_CONFIDENCE_THRESHOLD:
            self._metrics.low_confidence_count += 1

        # Track for combo detection
        self.combo_service.add_prediction(
            session_id=session_id,
            gesture=predicted_label,
            confidence=pred_confidence,
            model_type="rf",
        )

        combo_result = self.combo_service.check_combos(session_id)

        response = {
            "label": predicted_label,
            "prob": pred_confidence,
            "backend_mode": "random_forest",
            "feature_schema_version": self.model_manager.rf_feature_schema_version,
            "hand_detected": hand_detected,
        }

        if combo_result:
            response["combo"] = combo_result

        # Update metrics
        latency = (time.perf_counter() - start_time) * 1000
        self._metrics.total_predictions += 1
        self._metrics.total_latency_ms += latency
        response["latency_ms"] = round(latency, 2)

        # Cache result (without latency)
        if self._cache and cache_key:
            cache_response = {k: v for k, v in response.items() if k != "latency_ms"}
            self._cache.set(cache_key, cache_response)

        return response

    def predict_batch(
        self,
        images_data: list[bytes],
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Predict gestures from multiple images in batch.

        Args:
            images_data: List of image bytes
            session_id: Client session ID

        Returns:
            List of prediction results
        """
        if not self.model_manager.rf_available:
            raise HTTPException(status_code=503, detail="RandomForest model unavailable")

        results = []
        for data in images_data:
            try:
                result = self.predict_single(data, session_id, use_cache=True)
                results.append(result)
            except HTTPException as e:
                results.append({"error": e.detail, "status_code": e.status_code})

        return results

    def predict_sequence(
        self,
        frames_data: list,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Predict gesture from frame sequence using LSTM.

        Args:
            frames_data: List of image bytes
            session_id: Client session ID for combo tracking

        Returns:
            Dict with prediction results

        Raises:
            HTTPException: 501 if LSTM unavailable, 400 if invalid data
        """
        start_time = time.perf_counter()

        if not self.model_manager.lstm_available:
            logger.warning("LSTM prediction requested but model unavailable")
            raise HTTPException(
                status_code=501,
                detail="LSTM model not available. Sequence inference disabled.",
            )

        expected_frames = self._settings.max_sequence_frames
        if len(frames_data) != expected_frames:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected_frames} frames, got {len(frames_data)}",
            )

        # Extract features for all frames
        sequence_features = []
        hands_detected = 0

        for data in frames_data:
            features, _, detected = self._extract_features_with_confidence(data)
            sequence_features.append(features.flatten())
            if detected:
                hands_detected += 1

        x_sequence = np.array(sequence_features).reshape(1, expected_frames, -1)

        predicted_label, confidence = self.model_manager.predict_lstm(x_sequence)

        # Track for combo detection
        self.combo_service.add_prediction(
            session_id=session_id,
            gesture=predicted_label,
            confidence=confidence,
            model_type="lstm",
        )

        combo_result = self.combo_service.check_combos(session_id)

        latency = (time.perf_counter() - start_time) * 1000
        self._metrics.total_predictions += 1
        self._metrics.total_latency_ms += latency

        response = {
            "label": predicted_label,
            "prob": confidence,
            "model": "lstm",
            "backend_mode": "lstm",
            "feature_schema_version": self._settings.feature_schema_version,
            "frames_with_hands": hands_detected,
            "total_frames": expected_frames,
            "latency_ms": round(latency, 2),
        }

        if combo_result:
            response["combo"] = combo_result

        return response

    def _extract_features_with_confidence(self, data: bytes) -> tuple[np.ndarray, float, bool]:
        """Extract features from image bytes with confidence.

        Args:
            data: Image bytes

        Returns:
            Tuple of (features, confidence, hand_detected)

        Raises:
            HTTPException: 400 if invalid image
        """
        npimg = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image payload")

        frame = preprocess_frame(frame)

        try:
            features, confidence, detected = extract_features_with_confidence(frame)
            return features.reshape(1, -1), confidence, detected
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def _extract_features_from_bytes(self, data: bytes) -> np.ndarray:
        """Extract features from image bytes (legacy method).

        Args:
            data: Image bytes

        Returns:
            Feature array (1, n_features)

        Raises:
            HTTPException: 400 if invalid image
        """
        features, _, _ = self._extract_features_with_confidence(data)
        return features

    def get_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        return self._metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._metrics = PredictionMetrics()

    def clear_cache(self) -> None:
        """Clear prediction cache."""
        if self._cache:
            self._cache.clear()


# Module-level singleton
_prediction_service: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    """Get or create PredictionService singleton."""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service


def reset_prediction_service() -> None:
    """Reset prediction service singleton (for testing)."""
    global _prediction_service
    _prediction_service = None
