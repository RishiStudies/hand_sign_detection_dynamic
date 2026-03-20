import importlib
import os

try:
    from ..shared_artifacts import DATA_DIR, MODELS_DIR, REPORTS_DIR, SHARED_STATE_PATH
except ImportError:
    from shared_artifacts import DATA_DIR, MODELS_DIR, REPORTS_DIR, SHARED_STATE_PATH

TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
MEDIAPIPE_SPEC = importlib.util.find_spec("mediapipe")
MEDIAPIPE_AVAILABLE = MEDIAPIPE_SPEC is not None

FEATURE_SCHEMA = os.getenv("FEATURE_SCHEMA", "histogram").strip().lower() or "histogram"
FEATURE_SCHEMA_DIMENSIONS = {
    "histogram": 8,
    "mediapipe": 63,
}
FEATURE_SCHEMA_BY_DIMENSION = {
    dimension: schema for schema, dimension in FEATURE_SCHEMA_DIMENSIONS.items()
}

if FEATURE_SCHEMA not in FEATURE_SCHEMA_DIMENSIONS:
    raise ValueError(
        f"Unsupported FEATURE_SCHEMA '{FEATURE_SCHEMA}'. Expected one of: {', '.join(FEATURE_SCHEMA_DIMENSIONS)}"
    )

FEATURE_SCHEMA_VERSION = f"{FEATURE_SCHEMA}_v1"

DEVICE_PROFILES = {
    "pi_zero": {
        "rf_estimators": 64,
        "rf_max_depth": 16,
        "rf_n_jobs": 1,
        "max_classes": 8,
        "max_videos_per_class": 3,
        "sequence_length": 20,
        "frame_stride": 3,
        "lstm_low_end": True,
    },
    "full": {
        "rf_estimators": 300,
        "rf_max_depth": 20,
        "rf_n_jobs": -1,
        "max_classes": 50,
        "max_videos_per_class": 8,
        "sequence_length": 30,
        "frame_stride": 1,
        "lstm_low_end": False,
    },
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

__all__ = [
    "DATA_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "SHARED_STATE_PATH",
    "DEVICE_PROFILES",
    "TENSORFLOW_AVAILABLE",
    "MEDIAPIPE_AVAILABLE",
    "FEATURE_SCHEMA",
    "FEATURE_SCHEMA_DIMENSIONS",
    "FEATURE_SCHEMA_BY_DIMENSION",
    "FEATURE_SCHEMA_VERSION",
    "PROJECT_ROOT",
]
