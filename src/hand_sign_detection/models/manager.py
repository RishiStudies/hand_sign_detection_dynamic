"""Thread-safe model management.

Provides ModelManager class for loading and accessing ML models
with proper thread synchronization.
"""

import importlib
import logging
import os
from threading import Lock
from typing import Any

import joblib
import numpy as np

from hand_sign_detection.core.config import get_settings
from hand_sign_detection.core.shared_state import load_shared_state, resolve_shared_path
from hand_sign_detection.models.features import (
    get_feature_schema,
    get_feature_schema_version,
    validate_feature_contract,
)

logger = logging.getLogger("hand_sign_detection.models")


class ModelManager:
    """Thread-safe manager for ML models.

    Handles loading, accessing, and reloading RandomForest and LSTM models
    with proper thread synchronization.

    Usage:
        manager = ModelManager()
        manager.load_all()

        if manager.rf_available:
            label, confidence = manager.predict_rf(features)
    """

    def __init__(self):
        self._lock = Lock()

        # RandomForest state
        self._rf_model: Any | None = None
        self._rf_labels: np.ndarray | None = None
        self._rf_n_features: int = 8
        self._rf_feature_schema: str = "histogram"
        self._rf_feature_schema_version: str = "histogram_v1"

        # LSTM state
        self._lstm_model: Any | None = None
        self._lstm_labels: np.ndarray | None = None
        self._lstm_available: bool = False

        # Track loading state
        self._rf_loaded: bool = False
        self._lstm_loaded: bool = False

    @property
    def rf_available(self) -> bool:
        """Check if RandomForest model is available."""
        with self._lock:
            return self._rf_model is not None and self._rf_labels is not None

    @property
    def lstm_available(self) -> bool:
        """Check if LSTM model is available."""
        with self._lock:
            return self._lstm_available

    @property
    def rf_feature_schema(self) -> str:
        """Get RF model's feature schema."""
        with self._lock:
            return self._rf_feature_schema

    @property
    def rf_feature_schema_version(self) -> str:
        """Get RF model's feature schema version."""
        with self._lock:
            return self._rf_feature_schema_version

    @property
    def rf_n_features(self) -> int:
        """Get RF model's expected feature count."""
        with self._lock:
            return self._rf_n_features

    def load_all(self) -> dict[str, bool]:
        """Load all models.

        Returns:
            Dict with load status for each model type
        """
        return {
            "random_forest": self.load_rf_model(),
            "lstm": self.load_lstm_model(),
        }

    def load_rf_model(self) -> bool:
        """Load RandomForest model from shared artifacts.

        Returns:
            True if model loaded successfully
        """
        settings = get_settings()

        try:
            rf_model_path = resolve_shared_path("random_forest", "model_path")
            rf_labels_path = resolve_shared_path("random_forest", "labels_path")

            if not os.path.exists(rf_model_path):
                logger.warning("RF model file not found: %s", rf_model_path)
                return False

            if not os.path.exists(rf_labels_path):
                logger.warning("RF labels file not found: %s", rf_labels_path)
                return False

            loaded_model = joblib.load(rf_model_path)
            loaded_labels = np.load(rf_labels_path, allow_pickle=True)

            # Get feature contract from shared state
            feature_contract = self._get_rf_feature_contract()
            expected_dimension = feature_contract.get("feature_dimension")
            if expected_dimension is None:
                expected_dimension = int(loaded_model.n_features_in_)
            expected_dimension = int(expected_dimension)

            # Infer schema from dimension
            inferred_schema = settings.feature_schema_by_dimension.get(
                expected_dimension,
                feature_contract["feature_schema"],
            )

            validate_feature_contract(
                actual_dimension=int(loaded_model.n_features_in_),
                expected_dimension=expected_dimension,
                context="RandomForest model",
            )

            with self._lock:
                self._rf_model = loaded_model
                self._rf_labels = loaded_labels
                self._rf_n_features = loaded_model.n_features_in_
                self._rf_feature_schema = inferred_schema
                self._rf_feature_schema_version = str(
                    feature_contract.get("feature_schema_version") or f"{inferred_schema}_v1"
                )
                self._rf_loaded = True

            logger.info("RandomForest model loaded successfully")
            return True

        except Exception as exc:
            logger.warning("RandomForest load failed: %s", exc)
            return False

    def load_lstm_model(self) -> bool:
        """Load LSTM model from shared artifacts.

        Returns:
            True if model loaded successfully
        """
        settings = get_settings()

        if not settings.tensorflow_available:
            logger.warning("TensorFlow not available. LSTM inference disabled.")
            with self._lock:
                self._lstm_available = False
            return False

        try:
            load_model = importlib.import_module("tensorflow.keras.models").load_model
            lstm_model_path = resolve_shared_path("lstm", "model_path")

            if not os.path.exists(lstm_model_path):
                logger.warning("LSTM model file not found: %s", lstm_model_path)
                with self._lock:
                    self._lstm_available = False
                return False

            lstm_labels_path = resolve_shared_path("lstm", "labels_path")

            loaded_model = load_model(lstm_model_path)
            loaded_labels = np.load(lstm_labels_path, allow_pickle=True)

            with self._lock:
                self._lstm_model = loaded_model
                self._lstm_labels = loaded_labels
                self._lstm_available = True
                self._lstm_loaded = True

            logger.info("LSTM model loaded successfully")
            return True

        except Exception as exc:
            logger.error("LSTM load failed: %s", exc)
            with self._lock:
                self._lstm_model = None
                self._lstm_labels = None
                self._lstm_available = False
            return False

    def predict_rf(self, features: np.ndarray) -> tuple[str, float]:
        """Predict using RandomForest model.

        Args:
            features: Feature vector (1, n_features)

        Returns:
            Tuple of (label, confidence)

        Raises:
            RuntimeError: If model not available
        """
        with self._lock:
            if self._rf_model is None or self._rf_labels is None:
                raise RuntimeError("RandomForest model not available")

            probabilities = self._rf_model.predict_proba(features)[0]
            local_labels = self._rf_labels

        idx = int(np.argmax(probabilities))
        predicted_label = str(local_labels[idx])
        confidence = float(probabilities[idx])

        return predicted_label, confidence

    def predict_lstm(self, sequence: np.ndarray) -> tuple[str, float]:
        """Predict using LSTM model.

        Args:
            sequence: Sequence features (1, sequence_length, n_features)

        Returns:
            Tuple of (label, confidence)

        Raises:
            RuntimeError: If model not available
        """
        with self._lock:
            if not self._lstm_available or self._lstm_model is None:
                raise RuntimeError("LSTM model not available")

            predictions = self._lstm_model.predict(sequence, batch_size=1, verbose=0)
            local_labels = self._lstm_labels

        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
        predicted_label = (
            str(local_labels[predicted_class]) if local_labels is not None else str(predicted_class)
        )

        return predicted_label, confidence

    def get_status(self) -> dict[str, Any]:
        """Get current model status.

        Returns:
            Dict with model availability and metadata
        """
        settings = get_settings()

        with self._lock:
            return {
                "random_forest": self._rf_model is not None,
                "lstm": self._lstm_available,
                "tensorflow": settings.tensorflow_available,
                "rf_feature_schema": self._rf_feature_schema,
                "rf_feature_schema_version": self._rf_feature_schema_version,
                "rf_feature_dimension": self._rf_n_features,
            }

    def _get_rf_feature_contract(self) -> dict[str, Any]:
        """Get RF feature contract from shared state."""
        state = load_shared_state()
        rf_state = state.get("random_forest", {}) if isinstance(state, dict) else {}

        return {
            "feature_schema": rf_state.get("feature_schema") or get_feature_schema(),
            "feature_schema_version": rf_state.get("feature_schema_version")
            or get_feature_schema_version(),
            "feature_dimension": rf_state.get("feature_dimension"),
        }


# Module-level singleton for dependency injection
_model_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Get or create the ModelManager singleton.

    Returns:
        ModelManager instance
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def reset_model_manager() -> None:
    """Reset model manager singleton (for testing)."""
    global _model_manager
    _model_manager = None
