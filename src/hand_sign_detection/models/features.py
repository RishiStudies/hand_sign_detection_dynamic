"""Feature extraction utilities.

Provides feature extraction from images using different schemas:
- histogram: Color histogram features (8 dimensions)
- mediapipe: Hand landmark features (63 dimensions)
"""

import contextlib
import logging
from dataclasses import dataclass

import cv2
import numpy as np

from hand_sign_detection.core.config import get_settings

logger = logging.getLogger("hand_sign_detection.models.features")

# Lazy-loaded MediaPipe components
_mp_hands = None
_mp_drawing = None
_hands = None
_mediapipe_initialized = False


@dataclass
class HandDetectionResult:
    """Result of hand detection with landmarks."""

    detected: bool
    confidence: float
    landmarks: np.ndarray | None  # 21x3 array if detected
    bounding_box: tuple[int, int, int, int] | None  # x, y, w, h
    handedness: str  # "Left" or "Right"

    def to_feature_vector(self) -> np.ndarray:
        """Convert landmarks to flattened feature vector (63 dims)."""
        if self.landmarks is None:
            return np.zeros(63, dtype=np.float32)
        return self.landmarks.flatten().astype(np.float32)


def _init_mediapipe(
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> bool:
    """Lazily initialize MediaPipe hands detector.

    Args:
        max_num_hands: Maximum number of hands to detect
        min_detection_confidence: Minimum confidence for detection
        min_tracking_confidence: Minimum confidence for tracking

    Returns:
        True if MediaPipe initialized successfully
    """
    global _mp_hands, _mp_drawing, _hands, _mediapipe_initialized

    if _mediapipe_initialized:
        return _hands is not None

    settings = get_settings()
    if not settings.mediapipe_available:
        logger.warning("MediaPipe not available - install with: pip install mediapipe")
        _mediapipe_initialized = True
        return False

    try:
        import mediapipe as mp

        _mp_hands = mp.solutions.hands
        _mp_drawing = mp.solutions.drawing_utils
        _hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        _mediapipe_initialized = True
        logger.info("MediaPipe hands detector initialized successfully")
        return True
    except Exception as e:
        logger.error("Failed to initialize MediaPipe: %s", e)
        _mp_hands = None
        _mp_drawing = None
        _hands = None
        _mediapipe_initialized = True
        return False


def reset_mediapipe() -> None:
    """Reset MediaPipe state (useful for testing or reconfiguration)."""
    global _mp_hands, _mp_drawing, _hands, _mediapipe_initialized
    if _hands is not None:
        with contextlib.suppress(Exception):
            _hands.close()
    _mp_hands = None
    _mp_drawing = None
    _hands = None
    _mediapipe_initialized = False


def detect_hand(
    frame: np.ndarray,
    return_all_hands: bool = False,
) -> list[HandDetectionResult]:
    """Detect hands in a frame using MediaPipe.

    Args:
        frame: BGR image frame
        return_all_hands: If True, return all detected hands; else only best

    Returns:
        List of HandDetectionResult (empty if no hands detected)
    """
    if not _init_mediapipe():
        return []

    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _hands.process(rgb)

        if not results.multi_hand_landmarks:
            return []

        detections = []
        h, w = frame.shape[:2]

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Extract landmarks as 21x3 array
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
            )

            # Calculate bounding box from landmarks
            x_coords = landmarks[:, 0] * w
            y_coords = landmarks[:, 1] * h
            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            y_min, y_max = int(y_coords.min()), int(y_coords.max())
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

            # Get handedness
            handedness = "Right"
            if results.multi_handedness and idx < len(results.multi_handedness):
                handedness = results.multi_handedness[idx].classification[0].label
                confidence = results.multi_handedness[idx].classification[0].score
            else:
                confidence = 0.5

            detections.append(
                HandDetectionResult(
                    detected=True,
                    confidence=confidence,
                    landmarks=landmarks,
                    bounding_box=bbox,
                    handedness=handedness,
                )
            )

        if not return_all_hands and len(detections) > 1:
            # Return only the highest confidence detection
            detections = [max(detections, key=lambda d: d.confidence)]

        return detections

    except Exception as e:
        logger.debug("Hand detection failed: %s", e)
        return []


def draw_hand_landmarks(
    frame: np.ndarray,
    detection: HandDetectionResult,
    draw_bbox: bool = True,
    draw_landmarks: bool = True,
) -> np.ndarray:
    """Draw hand landmarks and bounding box on frame.

    Args:
        frame: BGR image frame (will be modified in place)
        detection: HandDetectionResult to draw
        draw_bbox: Whether to draw bounding box
        draw_landmarks: Whether to draw landmark points

    Returns:
        Frame with annotations drawn
    """
    if not detection.detected or detection.landmarks is None:
        return frame

    h, w = frame.shape[:2]

    # Draw bounding box
    if draw_bbox and detection.bounding_box:
        x, y, bw, bh = detection.bounding_box
        color = (0, 255, 0) if detection.handedness == "Right" else (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        label = f"{detection.handedness}: {detection.confidence:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw landmarks
    if draw_landmarks:
        for landmark in detection.landmarks:
            px = int(landmark[0] * w)
            py = int(landmark[1] * h)
            cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)

        # Draw connections (simplified - just connect adjacent landmarks)
        connections = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),  # Thumb
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),  # Index
            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),  # Middle
            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),  # Ring
            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),  # Pinky
            (5, 9),
            (9, 13),
            (13, 17),  # Palm
        ]
        for start, end in connections:
            p1 = (int(detection.landmarks[start][0] * w), int(detection.landmarks[start][1] * h))
            p2 = (int(detection.landmarks[end][0] * w), int(detection.landmarks[end][1] * h))
            cv2.line(frame, p1, p2, (0, 200, 0), 1)

    return frame


def extract_features_from_frame(frame: np.ndarray) -> np.ndarray:
    """Extract features from a video/image frame.

    Uses the configured feature schema (histogram or mediapipe).

    Args:
        frame: BGR image frame (numpy array)

    Returns:
        Feature vector as numpy array
    """
    settings = get_settings()
    schema = settings.feature_schema

    if schema == "histogram":
        return _extract_histogram_features(frame)

    # MediaPipe schema
    detections = detect_hand(frame, return_all_hands=False)
    if detections:
        return detections[0].to_feature_vector()

    # Fallback to zeros if no hand detected
    return np.zeros(get_expected_feature_dimension(), dtype=np.float32)


def extract_features_with_confidence(
    frame: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    """Extract features with detection confidence.

    Args:
        frame: BGR image frame

    Returns:
        Tuple of (features, confidence, hand_detected)
    """
    settings = get_settings()

    if settings.feature_schema == "histogram":
        features = _extract_histogram_features(frame)
        return features, 1.0, True  # Histogram always "detects"

    detections = detect_hand(frame, return_all_hands=False)
    if detections:
        return detections[0].to_feature_vector(), detections[0].confidence, True

    return np.zeros(63, dtype=np.float32), 0.0, False


def _extract_histogram_features(frame: np.ndarray) -> np.ndarray:
    """Extract color histogram features.

    Args:
        frame: BGR image frame

    Returns:
        8-dimensional histogram feature vector
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
    hist = hist.flatten() / (hist.sum() + 1e-10)
    return hist.astype(np.float32)


def extract_features_from_bytes(data: bytes) -> np.ndarray:
    """Extract features from image bytes.

    Args:
        data: Image data as bytes (JPEG, PNG, etc.)

    Returns:
        Feature vector as numpy array

    Raises:
        ValueError: If image cannot be decoded
    """
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid image payload")
    return extract_features_from_frame(frame)


def preprocess_frame(
    frame: np.ndarray,
    target_size: tuple[int, int] | None = None,
    normalize: bool = False,
) -> np.ndarray:
    """Preprocess frame for optimal inference.

    Args:
        frame: Input BGR frame
        target_size: Optional (width, height) to resize to
        normalize: Whether to normalize pixel values to [0, 1]

    Returns:
        Preprocessed frame
    """
    settings = get_settings()
    h, w = frame.shape[:2]

    # Resize if needed
    if target_size:
        tw, th = target_size
        if w != tw or h != th:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    elif w > settings.max_frame_width or h > settings.max_frame_height:
        scale = min(settings.max_frame_width / w, settings.max_frame_height / h)
        frame = cv2.resize(
            frame,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    if normalize:
        frame = frame.astype(np.float32) / 255.0

    return frame


def resize_frame_for_inference(frame: np.ndarray) -> np.ndarray:
    """Resize frame to maximum dimensions for efficient inference.

    Args:
        frame: Input frame

    Returns:
        Resized frame (or original if within limits)
    """
    return preprocess_frame(frame)


def batch_extract_features(frames: list[np.ndarray]) -> np.ndarray:
    """Extract features from multiple frames.

    Args:
        frames: List of BGR frames

    Returns:
        Array of shape (n_frames, feature_dim)
    """
    if not frames:
        return np.array([])

    features = [extract_features_from_frame(f) for f in frames]
    return np.array(features, dtype=np.float32)


def get_feature_schema() -> str:
    """Get current feature schema name."""
    return get_settings().feature_schema


def get_feature_schema_version() -> str:
    """Get current feature schema version string."""
    return get_settings().feature_schema_version


def get_expected_feature_dimension(schema: str | None = None) -> int:
    """Get expected feature dimension for a schema.

    Args:
        schema: Schema name (uses current if None)

    Returns:
        Feature dimension
    """
    settings = get_settings()
    selected_schema = schema or settings.feature_schema
    return settings.feature_schema_dimensions.get(selected_schema, 8)


def validate_feature_contract(
    actual_dimension: int,
    expected_dimension: int,
    context: str,
) -> None:
    """Validate feature dimensions match expected contract.

    Args:
        actual_dimension: Actual feature dimension
        expected_dimension: Expected feature dimension
        context: Context string for error message

    Raises:
        ValueError: If dimensions don't match
    """
    if actual_dimension != expected_dimension:
        raise ValueError(
            f"{context} feature mismatch: expected {expected_dimension}, got {actual_dimension}"
        )
