import cv2
import numpy as np

from .config import (
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_DIMENSIONS,
    FEATURE_SCHEMA_VERSION,
    MEDIAPIPE_AVAILABLE,
)

mp_hands = None
hands = None
if MEDIAPIPE_AVAILABLE:
    try:
        import mediapipe as mp

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )
    except Exception:
        mp_hands = None
        hands = None


def extract_features_from_frame(frame: np.ndarray) -> np.ndarray:
    if FEATURE_SCHEMA == "histogram":
        return _extract_histogram_features(frame)

    if hands is not None:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                row = []
                for landmark in hand_landmarks.landmark:
                    row += [landmark.x, landmark.y, landmark.z]
                return np.array(row)
        except Exception:
            pass

    return np.zeros(get_expected_feature_dimension(), dtype=np.float32)


def _extract_histogram_features(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
    hist = hist.flatten() / (hist.sum() + 1e-10)
    return hist


def extract_features_from_bytes(data: bytes) -> np.ndarray:
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid image payload")
    return extract_features_from_frame(frame)


def get_feature_schema() -> str:
    return FEATURE_SCHEMA


def get_feature_schema_version() -> str:
    return FEATURE_SCHEMA_VERSION


def get_expected_feature_dimension(schema: str | None = None) -> int:
    selected_schema = schema or FEATURE_SCHEMA
    return int(FEATURE_SCHEMA_DIMENSIONS[selected_schema])
