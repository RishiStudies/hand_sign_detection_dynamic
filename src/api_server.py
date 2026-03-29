import json
import logging
import os
import time
import importlib
import uuid
import warnings
from collections import defaultdict, deque
from logging.handlers import RotatingFileHandler
from threading import Lock
from typing import Any, Dict, List, Optional

import cv2
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

try:
    from .shared_artifacts import load_shared_state, resolve_shared_path, update_shared_state
    from .job_queue import enqueue_named_job, get_job_status, is_job_queue_available
    from .training_module.config import (
        FEATURE_SCHEMA,
        FEATURE_SCHEMA_BY_DIMENSION,
        FEATURE_SCHEMA_VERSION,
    )
    from .training_module.features import extract_features_from_frame as extract_shared_features_from_frame
    from .training_module.features import get_expected_feature_dimension
except ImportError:
    from shared_artifacts import load_shared_state, resolve_shared_path, update_shared_state
    from job_queue import enqueue_named_job, get_job_status, is_job_queue_available
    from training_module.config import (
        FEATURE_SCHEMA,
        FEATURE_SCHEMA_BY_DIMENSION,
        FEATURE_SCHEMA_VERSION,
    )
    from training_module.features import extract_features_from_frame as extract_shared_features_from_frame
    from training_module.features import get_expected_feature_dimension

warnings.filterwarnings("ignore")

# ========== LOGGING SETUP ==========
def setup_logging():
    """Configure structured logging with optional file rotation.
    
    Logs to:
    - stderr (always)
    - logs/api_server.log (if LOG_TO_FILE enabled)
    
    Environment variables:
    - LOG_LEVEL: DEBUG, INFO, WARNING, ERROR (default: INFO)
    - LOG_TO_FILE: true/1 to enable file logging (default: false)
    - LOGS_DIR: directory for log files (default: ./logs)
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_to_file = os.getenv("LOG_TO_FILE", "false").lower() in ("true", "1", "yes")
    logs_dir = os.getenv("LOGS_DIR", "logs")
    
    logger_obj = logging.getLogger("api_server")
    logger_obj.setLevel(getattr(logging, log_level, logging.INFO))
    
    # Clear existing handlers
    logger_obj.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger_obj.addHandler(console_handler)
    
    # File handler (optional, with rotation)
    if log_to_file:
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, "api_server.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=50_000_000,  # 50MB
            backupCount=5  # Keep 5 backups
        )
        file_handler.setFormatter(formatter)
        logger_obj.addHandler(file_handler)
        logger_obj.info("File logging enabled at %s", log_file)
    
    return logger_obj

logger = setup_logging()
logger.info("API Server starting up...")

# Performance and safety limits.
MAX_FRAME_WIDTH = 320
MAX_FRAME_HEIGHT = 240
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024)))
MAX_SEQUENCE_FRAMES = int(os.getenv("MAX_SEQUENCE_FRAMES", "30"))
MAX_CSV_UPLOAD_BYTES = int(os.getenv("MAX_CSV_UPLOAD_BYTES", str(10 * 1024 * 1024)))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
MAX_PREDICT_REQUESTS_PER_WINDOW = int(
    os.getenv("MAX_PREDICT_REQUESTS_PER_WINDOW", "180")
)
MAX_SEQUENCE_REQUESTS_PER_WINDOW = int(
    os.getenv("MAX_SEQUENCE_REQUESTS_PER_WINDOW", "30")
)
MAX_TRAIN_REQUESTS_PER_WINDOW = int(os.getenv("MAX_TRAIN_REQUESTS_PER_WINDOW", "6"))
MAX_CONCURRENT_SEQUENCE_REQUESTS = int(
    os.getenv("MAX_CONCURRENT_SEQUENCE_REQUESTS", "2")
)
ALLOWED_IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "application/octet-stream",
}
TRAINING_API_KEY = os.getenv("TRAINING_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "").strip()
REDIS_RATE_LIMIT_PREFIX = os.getenv("REDIS_RATE_LIMIT_PREFIX", "hsd:ratelimit")
REDIS_COMBO_PREFIX = os.getenv("REDIS_COMBO_PREFIX", "hsd:combo")
COMBO_STATE_TTL_SECONDS = int(os.getenv("COMBO_STATE_TTL_SECONDS", "300"))

TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None
if not TENSORFLOW_AVAILABLE:
    logger.warning("TensorFlow not available. LSTM predictions are disabled.")

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=500)

cors_origins_env = os.getenv("CORS_ORIGINS")
cors_origins = (
    [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
    if cors_origins_env
    else [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== ENVIRONMENT VALIDATION ==========
def validate_environment():
    """Validate critical environment variables at startup."""
    warnings_list = []
    errors_list = []
    
    # Check TRAINING_API_KEY in production
    if not TRAINING_API_KEY:
        warnings_list.append(
            "TRAINING_API_KEY not set - training endpoints are OPEN to anyone. "
            "Required for production deployments."
        )
    elif len(TRAINING_API_KEY) < 16:
        warnings_list.append("TRAINING_API_KEY is weak (less than 16 characters)")
    
    # Check CORS configuration
    if not cors_origins_env:
        logger.info("CORS_ORIGINS not set, using default localhost origins")
    
    # Check Redis availability
    if REDIS_URL:
        logger.info("Redis URL configured for distributed backend")
    else:
        logger.info("Redis not configured, using in-memory fallback")
    
    # Log warnings and errors
    for warning in warnings_list:
        logger.warning("⚠ %s", warning)
    for error in errors_list:
        logger.error("✗ %s", error)
    
    if errors_list:
        raise RuntimeError(f"Environment validation failed: {'; '.join(errors_list)}")

validate_environment()

script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "../models")
data_dir = os.path.join(script_dir, "../data")
job_inputs_dir = os.path.join(data_dir, "job_inputs")

model = None
labels = None
n_features = 8
rf_feature_schema = FEATURE_SCHEMA
rf_feature_schema_version = FEATURE_SCHEMA_VERSION
model_lock = Lock()
training_lock = Lock()
rate_limit_lock = Lock()
rate_limit_store: Dict[str, deque] = defaultdict(deque)
sequence_inflight_lock = Lock()
sequence_inflight_count = 0
redis_client = None

if REDIS_URL:
    try:
        redis_module = importlib.import_module("redis")
        redis_client = redis_module.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        redis_client.ping()
        logger.info("Redis connected for distributed rate limiting")
    except Exception as exc:
        redis_client = None
        logger.warning("Redis unavailable, falling back to in-memory rate limits: %s", exc)


class ComboDetector:
    def __init__(self):
        self.combos = {
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
        self.prediction_buffer = deque(maxlen=10)
        self.buffer_timeout = 5.0

    def add_prediction(self, gesture: str, confidence: float, model_type: str = "rf"):
        timestamp = time.time()
        self.prediction_buffer.append(
            {
                "gesture": gesture,
                "confidence": confidence,
                "timestamp": timestamp,
                "model": model_type,
            }
        )

    def check_combos(self, min_confidence: float = 0.7):
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
        self, gesture_sequence: List[str], combo_sequence: List[str]
    ) -> bool:
        if len(gesture_sequence) < len(combo_sequence):
            return False

        recent_gestures = gesture_sequence[-len(combo_sequence) :]
        return recent_gestures == combo_sequence

    def get_available_combos(self):
        return list(self.combos.keys())


combo_detectors: Dict[str, ComboDetector] = {}
combo_detectors_lock = Lock()
combo_catalog = ComboDetector()


lstm_model = None
lstm_labels = None
lstm_available = False  # Track LSTM availability for startup checks


def resize_frame_for_inference(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= MAX_FRAME_WIDTH and h <= MAX_FRAME_HEIGHT:
        return frame

    scale = min(MAX_FRAME_WIDTH / w, MAX_FRAME_HEIGHT / h)
    return cv2.resize(
        frame,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def validate_feature_contract(actual_dimension: int, expected_dimension: int, context: str) -> None:
    if actual_dimension != expected_dimension:
        raise ValueError(
            f"{context} feature mismatch: expected {expected_dimension}, got {actual_dimension}"
        )


def get_rf_feature_contract() -> Dict[str, Any]:
    state = load_shared_state()
    rf_state = state.get("random_forest", {}) if isinstance(state, dict) else {}
    feature_schema = rf_state.get("feature_schema") or FEATURE_SCHEMA
    feature_schema_version = rf_state.get("feature_schema_version") or FEATURE_SCHEMA_VERSION
    expected_dimension = rf_state.get("feature_dimension")

    return {
        "feature_schema": str(feature_schema),
        "feature_schema_version": str(feature_schema_version),
        "feature_dimension": int(expected_dimension) if expected_dimension is not None else None,
    }


def load_rf_model() -> bool:
    global model, labels, n_features, rf_feature_schema, rf_feature_schema_version
    try:
        rf_model_path = resolve_shared_path("random_forest", "model_path")
        rf_labels_path = resolve_shared_path("random_forest", "labels_path")
        loaded_model = joblib.load(rf_model_path)
        loaded_labels = np.load(rf_labels_path, allow_pickle=True)
        feature_contract = get_rf_feature_contract()
        expected_dimension = feature_contract.get("feature_dimension")
        if expected_dimension is None:
            expected_dimension = int(loaded_model.n_features_in_)
        expected_dimension = int(expected_dimension)
        inferred_schema = FEATURE_SCHEMA_BY_DIMENSION.get(expected_dimension, feature_contract["feature_schema"])
        validate_feature_contract(
            actual_dimension=int(loaded_model.n_features_in_),
            expected_dimension=expected_dimension,
            context="RandomForest model",
        )
        with model_lock:
            model = loaded_model
            labels = loaded_labels
            n_features = loaded_model.n_features_in_
            rf_feature_schema = inferred_schema
            rf_feature_schema_version = str(
                feature_contract.get("feature_schema_version")
                or f"{inferred_schema}_v1"
            )
        logger.info("RandomForest model loaded successfully")
        return True
    except Exception as exc:
        logger.warning("RandomForest load failed: %s", exc)
        return False


def load_lstm_model() -> bool:
    global lstm_model, lstm_labels, lstm_available
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow not available. LSTM inference disabled.")
        lstm_available = False
        return False

    try:
        load_model = importlib.import_module("tensorflow.keras.models").load_model
        lstm_model_path = resolve_shared_path("lstm", "model_path")
        if not os.path.exists(lstm_model_path):
            logger.error("LSTM model path does not exist: %s", lstm_model_path)
            lstm_available = False
            return False
        lstm_model = load_model(lstm_model_path)
        lstm_labels = np.load(resolve_shared_path("lstm", "labels_path"), allow_pickle=True)
        lstm_available = True
        logger.info("LSTM model loaded successfully")
        return True
    except Exception as exc:
        logger.error("LSTM load failed: %s", exc)
        lstm_model = None
        lstm_labels = None
        lstm_available = False
        return False


def get_combo_detector(session_id: Optional[str]) -> ComboDetector:
    session_key = (session_id or "anonymous").strip() or "anonymous"
    with combo_detectors_lock:
        if session_key not in combo_detectors:
            combo_detectors[session_key] = ComboDetector()
        return combo_detectors[session_key]


def get_combo_session_key(session_id: Optional[str]) -> str:
    return (session_id or "anonymous").strip() or "anonymous"


def get_combo_redis_key(session_id: Optional[str]) -> str:
    session_key = get_combo_session_key(session_id)
    return f"{REDIS_COMBO_PREFIX}:{session_key}"


def load_combo_predictions(session_id: Optional[str]) -> List[Dict[str, Any]]:
    if redis_client is not None:
        try:
            serialized_predictions = redis_client.lrange(get_combo_redis_key(session_id), 0, -1)
            return [json.loads(item) for item in serialized_predictions]
        except Exception as exc:
            logger.warning("Redis combo read failed, using in-memory fallback: %s", exc)

    return list(get_combo_detector(session_id).prediction_buffer)


def add_prediction_for_session(
    session_id: Optional[str],
    gesture: str,
    confidence: float,
    model_type: str,
) -> None:
    prediction = {
        "gesture": gesture,
        "confidence": confidence,
        "timestamp": time.time(),
        "model": model_type,
    }

    if redis_client is not None:
        try:
            redis_key = get_combo_redis_key(session_id)
            redis_client.rpush(redis_key, json.dumps(prediction))
            redis_client.ltrim(redis_key, -10, -1)
            redis_client.expire(redis_key, COMBO_STATE_TTL_SECONDS)
            return
        except Exception as exc:
            logger.warning("Redis combo write failed, using in-memory fallback: %s", exc)

    detector = get_combo_detector(session_id)
    detector.prediction_buffer.append(prediction)


def check_combos_for_session(
    session_id: Optional[str],
    min_confidence: float = 0.7,
):
    predictions = load_combo_predictions(session_id)
    if len(predictions) < 2:
        return None

    detector = ComboDetector()
    detector.prediction_buffer.extend(predictions)
    return detector.check_combos(min_confidence=min_confidence)


def clear_combo_state(session_id: Optional[str]) -> None:
    if redis_client is not None:
        try:
            redis_client.delete(get_combo_redis_key(session_id))
            return
        except Exception as exc:
            logger.warning("Redis combo delete failed, using in-memory fallback: %s", exc)

    detector = get_combo_detector(session_id)
    detector.prediction_buffer.clear()


def validate_upload(file: UploadFile, data: bytes, field_name: str = "file") -> None:
    if file.content_type and file.content_type.lower() not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type for {field_name}: {file.content_type}",
        )
    if not data:
        raise HTTPException(status_code=400, detail=f"{field_name} is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{field_name} exceeds max size {MAX_UPLOAD_BYTES} bytes",
        )


def persist_sample_training_inputs(
    samples: List[UploadFile], sample_payloads: List[bytes], labels_input: List[str]
) -> str:
    job_dir = os.path.join(job_inputs_dir, f"train-{uuid.uuid4().hex}")
    os.makedirs(job_dir, exist_ok=True)

    manifest_samples = []
    for index, (sample, data, label) in enumerate(zip(samples, sample_payloads, labels_input)):
        file_path = os.path.join(job_dir, f"sample_{index:04d}.jpg")
        with open(file_path, "wb") as file_obj:
            file_obj.write(data)
        manifest_samples.append({"label": label, "file_path": file_path})

    manifest_path = os.path.join(job_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file_obj:
        json.dump({"samples": manifest_samples}, file_obj)
    return manifest_path


def persist_csv_training_input(data: bytes) -> str:
    os.makedirs(job_inputs_dir, exist_ok=True)
    csv_path = os.path.join(job_inputs_dir, f"train_csv_{uuid.uuid4().hex}.csv")
    with open(csv_path, "wb") as file_obj:
        file_obj.write(data)
    return csv_path


def validate_csv_schema(csv_path: str, max_rows_to_check: int = 100) -> None:
    """Validate CSV structure before training.
    
    Checks:
    - File is readable as CSV
    - Has expected columns (numeric features + label column)
    - Data types are numeric
    - Not empty
    
    Raises HTTPException on validation failure.
    """
    try:
        df = pd.read_csv(csv_path, nrows=max_rows_to_check)
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        # CSV should have numeric columns (features) and a label column
        # Expected pattern: columns of floats/ints with possible 'label', 'class', 'gesture' column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            raise ValueError("CSV has no numeric feature columns")
        
        # At least one label/class/gesture column expected, or numeric column names
        logger.info(
            "CSV validation passed: %d numeric features, "
            "%d non-numeric columns", 
            len(numeric_cols), 
            len(non_numeric_cols)
        )
        
    except pd.errors.ParserError as exc:
        logger.warning("CSV parsing failed: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CSV format: {str(exc)}"
        ) from exc
    except ValueError as exc:
        logger.warning("CSV structure validation failed: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"CSV validation failed: {str(exc)}"
        ) from exc
    except Exception as exc:
        logger.warning("Unexpected CSV validation error: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"CSV validation error: {str(exc)}"
        ) from exc


def require_training_key(x_api_key: Optional[str]) -> None:
    """Enforce API key requirement for training endpoints.
    
    In production (TRAINING_API_KEY set), this is mandatory.
    Logs auth attempts for audit trail.
    """
    if not TRAINING_API_KEY:
        logger.warning(
            "Training API not secured: TRAINING_API_KEY environment variable not set"
        )
        return
    
    if x_api_key != TRAINING_API_KEY:
        logger.warning("Unauthorized training attempt with invalid/missing API key")
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


def acquire_training_slot() -> None:
    if not training_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Training already in progress")


def get_client_identity(request: Request, session_id: Optional[str] = None) -> str:
    if session_id:
        return f"session:{session_id.strip() or 'anonymous'}"

    x_forwarded_for = request.headers.get("x-forwarded-for", "")
    if x_forwarded_for:
        return f"ip:{x_forwarded_for.split(',')[0].strip()}"
    if request.client and request.client.host:
        return f"ip:{request.client.host}"
    return "ip:unknown"


def enforce_rate_limit(
    bucket: str,
    key: str,
    max_requests: int,
    window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
) -> None:
    if redis_client is not None:
        try:
            enforce_rate_limit_redis(bucket, key, max_requests, window_seconds)
            return
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Redis rate limit failed, using in-memory fallback: %s", exc)

    now = time.time()
    composite_key = f"{bucket}:{key}"

    with rate_limit_lock:
        request_times = rate_limit_store[composite_key]
        while request_times and (now - request_times[0]) > window_seconds:
            request_times.popleft()

        if len(request_times) >= max_requests:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(window_seconds)},
            )

        request_times.append(now)


def enforce_rate_limit_redis(
    bucket: str,
    key: str,
    max_requests: int,
    window_seconds: int,
) -> None:
    redis_key = f"{REDIS_RATE_LIMIT_PREFIX}:{bucket}:{key}"
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


def acquire_sequence_slot() -> None:
    global sequence_inflight_count
    with sequence_inflight_lock:
        if sequence_inflight_count >= MAX_CONCURRENT_SEQUENCE_REQUESTS:
            raise HTTPException(
                status_code=503,
                detail="Sequence inference busy. Retry shortly.",
                headers={"Retry-After": "2"},
            )
        sequence_inflight_count += 1


def release_sequence_slot() -> None:
    global sequence_inflight_count
    with sequence_inflight_lock:
        if sequence_inflight_count > 0:
            sequence_inflight_count -= 1


def extract_features_from_frame(frame: np.ndarray) -> np.ndarray:
    frame = resize_frame_for_inference(frame)
    features = extract_shared_features_from_frame(frame).flatten()
    validate_feature_contract(
        actual_dimension=int(features.shape[0]),
        expected_dimension=int(n_features),
        context="Inference request",
    )
    return features.reshape(1, -1)


def extract_features_from_bytes(data: bytes) -> np.ndarray:
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image payload")
    try:
        return extract_features_from_frame(frame)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health/live")
def health_live():
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready():
    if model is None or labels is None:
        raise HTTPException(status_code=503, detail="RandomForest model unavailable")
    return {
        "status": "ready",
        "random_forest": "available",
        "lstm": "available" if lstm_available else "unavailable",
    }


@app.get("/health/details")
def health_details():
    state = load_shared_state()
    return {
        "status": "ready" if model is not None and labels is not None else "degraded",
        "rate_limit_backend": "redis" if redis_client is not None else "in_memory",
        "combo_state_backend": "redis" if redis_client is not None else "in_memory",
        "job_queue_backend": "redis_rq" if is_job_queue_available() else "unavailable",
        "model_status": {
            "random_forest": model is not None and labels is not None,
            "lstm": lstm_available,
            "tensorflow": TENSORFLOW_AVAILABLE,
            "rf_feature_schema": rf_feature_schema,
            "rf_feature_schema_version": rf_feature_schema_version,
            "rf_feature_dimension": n_features,
        },
        "limits": {
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "max_csv_upload_bytes": MAX_CSV_UPLOAD_BYTES,
            "max_sequence_frames": MAX_SEQUENCE_FRAMES,
            "max_frame": {
                "width": MAX_FRAME_WIDTH,
                "height": MAX_FRAME_HEIGHT,
            },
        },
        "shared_artifacts": state,
    }


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(default=None),
):
    client_key = get_client_identity(request, session_id=x_session_id)
    enforce_rate_limit("predict", client_key, MAX_PREDICT_REQUESTS_PER_WINDOW)

    if model is None or labels is None:
        raise HTTPException(status_code=503, detail="RandomForest model unavailable")

    data = await file.read()
    validate_upload(file, data)
    features = extract_features_from_bytes(data)

    with model_lock:
        probabilities = model.predict_proba(features)[0]
        local_labels = labels

    idx = int(np.argmax(probabilities))
    predicted_label = str(local_labels[idx])
    confidence = float(probabilities[idx])

    add_prediction_for_session(x_session_id, predicted_label, confidence, "rf")
    combo_result = check_combos_for_session(x_session_id)

    response = {
        "label": predicted_label,
        "prob": confidence,
        "backend_mode": "random_forest",
        "feature_schema_version": rf_feature_schema_version,
    }
    if combo_result:
        response["combo"] = combo_result

    return response


@app.post("/predict_sequence")
async def predict_sequence(
    request: Request,
    files: List[UploadFile] = File(...),
    x_session_id: Optional[str] = Header(default=None),
):
    client_key = get_client_identity(request, session_id=x_session_id)
    enforce_rate_limit("predict_sequence", client_key, MAX_SEQUENCE_REQUESTS_PER_WINDOW)
    acquire_sequence_slot()

    try:
        if not lstm_available:
            logger.warning("LSTM prediction requested but model unavailable")
            raise HTTPException(
                status_code=501,
                detail="LSTM model not available. Sequence inference disabled."
            )

        if len(files) != MAX_SEQUENCE_FRAMES:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {MAX_SEQUENCE_FRAMES} frames, got {len(files)}",
            )

        sequence_features = []
        for i, file in enumerate(files):
            data = await file.read()
            validate_upload(file, data, field_name=f"files[{i}]")
            features = extract_features_from_bytes(data)
            sequence_features.append(features.flatten())

        x_sequence = np.array(sequence_features).reshape(1, MAX_SEQUENCE_FRAMES, -1)
        predictions = lstm_model.predict(x_sequence, batch_size=1, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])

        predicted_label = (
            str(lstm_labels[predicted_class])
            if lstm_labels is not None
            else str(predicted_class)
        )

        add_prediction_for_session(x_session_id, predicted_label, confidence, "lstm")
        combo_result = check_combos_for_session(x_session_id)

        response = {
            "label": predicted_label,
            "prob": confidence,
            "model": "lstm",
            "backend_mode": "lstm",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        }
        if combo_result:
            response["combo"] = combo_result

        return response
    finally:
        release_sequence_slot()


@app.get("/combos")
def get_combos(x_session_id: Optional[str] = Header(default=None)):
    return {
        "combos": combo_catalog.get_available_combos(),
        "patterns": combo_catalog.combos,
    }


@app.post("/clear_combos")
def clear_combo_history(x_session_id: Optional[str] = Header(default=None)):
    clear_combo_state(x_session_id)
    return {"status": "cleared"}


@app.get("/")
def index():
    frontend_url = os.getenv("FRONTEND_URL", "http://127.0.0.1:3000")
    return {
        "message": "Frontend moved to Next.js app",
        "frontend_url": frontend_url,
        "docs": "/docs",
    }


@app.get("/artifacts")
def artifacts():
    return load_shared_state()


@app.get("/training")
def training():
    frontend_url = os.getenv("FRONTEND_URL", "http://127.0.0.1:3000")
    return {
        "message": "Use the Next.js frontend for training UI",
        "frontend_url": frontend_url,
    }


@app.get("/jobs/{job_id}")
def get_training_job(job_id: str):
    if not is_job_queue_available():
        raise HTTPException(status_code=503, detail="Job queue unavailable")

    try:
        job_state = get_job_status(job_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    result = job_state.get("result")
    if job_state.get("status") == "finished" and isinstance(result, dict):
        if result.get("job_name") == "train_lstm":
            load_lstm_model()
        if result.get("job_name") in {"train_rf_samples", "train_rf_csv"}:
            load_rf_model()

    return job_state


@app.post("/train", status_code=202)
async def train(
    request: Request,
    samples: List[UploadFile] = File(...),
    labels_input: List[str] = Form(...),
    x_api_key: Optional[str] = Header(default=None),
):
    require_training_key(x_api_key)
    enforce_rate_limit(
        "train", get_client_identity(request), MAX_TRAIN_REQUESTS_PER_WINDOW
    )
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
            logging.warning("Invalid label rejected in train endpoint: %s", label)
            raise HTTPException(status_code=400, detail=f"Invalid label at index {i}")
        data = await sample.read()
        validate_upload(sample, data, field_name="sample")
        sample_payloads.append(data)

    manifest_path = persist_sample_training_inputs(samples, sample_payloads, labels_input)
    try:
        job = enqueue_named_job("train_rf_samples", manifest_path=manifest_path)
        logger.info("Sample training job queued: %s with %d samples", job.id, len(samples))
    except Exception as exc:
        logger.error("Failed to queue sample training job: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "queued",
        "job_id": job.id,
        "job_type": "train_rf_samples",
    }


@app.post("/train_csv", status_code=202)
async def train_csv(
    request: Request,
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None),
):
    require_training_key(x_api_key)
    enforce_rate_limit(
        "train", get_client_identity(request), MAX_TRAIN_REQUESTS_PER_WINDOW
    )
    if not is_job_queue_available():
        raise HTTPException(status_code=503, detail="Job queue unavailable")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    if len(data) > MAX_CSV_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"CSV file exceeds max size {MAX_CSV_UPLOAD_BYTES} bytes",
        )
    if file.content_type and "csv" not in file.content_type.lower():
        logger.warning("Rejected CSV upload with wrong content type: %s", file.content_type)
        raise HTTPException(status_code=415, detail="Expected CSV content type")

    csv_path = persist_csv_training_input(data)
    
    # Validate CSV schema before queuing job
    validate_csv_schema(csv_path)
    
    try:
        job = enqueue_named_job("train_rf_csv", csv_path=csv_path)
        logger.info("CSV training job queued: %s", job.id)
    except Exception as exc:
        logger.error("Failed to queue CSV training job: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "queued",
        "job_id": job.id,
        "job_type": "train_rf_csv",
    }


@app.post("/process_wlasl", status_code=202)
async def process_wlasl(
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
):
    require_training_key(x_api_key)
    enforce_rate_limit(
        "train", get_client_identity(request), MAX_TRAIN_REQUESTS_PER_WINDOW
    )
    if not is_job_queue_available():
        raise HTTPException(status_code=503, detail="Job queue unavailable")

    try:
        job = enqueue_named_job("process_wlasl")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "queued",
        "job_id": job.id,
        "job_type": "process_wlasl",
    }


@app.post("/train_lstm", status_code=202)
async def train_lstm(
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
):
    require_training_key(x_api_key)
    enforce_rate_limit(
        "train", get_client_identity(request), MAX_TRAIN_REQUESTS_PER_WINDOW
    )
    if not is_job_queue_available():
        raise HTTPException(status_code=503, detail="Job queue unavailable")

    try:
        job = enqueue_named_job("train_lstm")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "queued",
        "job_id": job.id,
        "job_type": "train_lstm",
    }


load_rf_model()
load_lstm_model()

if model is None or labels is None:
    logger.error("CRITICAL: RandomForest model failed to load at startup")
else:
    logger.info("API Server initialized successfully. RandomForest model available.")
    
if not lstm_available:
    logger.warning("LSTM model not available - sequence inference disabled")
else:
    logger.info("LSTM model available for sequence inference")
