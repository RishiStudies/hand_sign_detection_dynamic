"""Model management and feature extraction.

Contains:
- ModelManager: Thread-safe model loading and inference
- Feature extraction utilities with MediaPipe integration
- Hand detection with confidence scoring
"""

from hand_sign_detection.models.features import (
    HandDetectionResult,
    batch_extract_features,
    detect_hand,
    draw_hand_landmarks,
    extract_features_from_bytes,
    extract_features_from_frame,
    extract_features_with_confidence,
    get_expected_feature_dimension,
    get_feature_schema,
    preprocess_frame,
    reset_mediapipe,
)
from hand_sign_detection.models.manager import ModelManager, get_model_manager, reset_model_manager

__all__ = [
    "ModelManager",
    "get_model_manager",
    "reset_model_manager",
    "HandDetectionResult",
    "detect_hand",
    "draw_hand_landmarks",
    "extract_features_from_frame",
    "extract_features_from_bytes",
    "extract_features_with_confidence",
    "batch_extract_features",
    "preprocess_frame",
    "get_feature_schema",
    "get_expected_feature_dimension",
    "reset_mediapipe",
]
