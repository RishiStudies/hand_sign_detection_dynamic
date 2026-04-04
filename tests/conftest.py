"""Pytest configuration and shared fixtures for Hand Sign Detection tests."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all singletons before each test."""
    # Reset before test
    from hand_sign_detection.core.redis import reset_redis_client
    from hand_sign_detection.models.manager import reset_model_manager
    from hand_sign_detection.services.combo_detection import reset_combo_service
    from hand_sign_detection.services.prediction import reset_prediction_service
    from hand_sign_detection.services.rate_limiting import reset_rate_limiter

    reset_model_manager()
    reset_rate_limiter()
    reset_combo_service()
    reset_prediction_service()
    reset_redis_client()

    yield

    # Reset after test
    reset_model_manager()
    reset_rate_limiter()
    reset_combo_service()
    reset_prediction_service()
    reset_redis_client()


@pytest.fixture
def temp_models_dir(tmp_path):
    """Create a temporary directory for test model artifacts."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return str(models_dir)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return str(data_dir)


@pytest.fixture
def mock_rf_model(temp_models_dir):
    """Create a minimal mock RandomForest model for testing."""
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    # Create a simple RF model with 8 features (histogram schema)
    X_train = np.random.rand(10, 8).astype(np.float32)
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)

    model_path = os.path.join(temp_models_dir, "hand_alphabet_model.pkl")
    joblib.dump(model, model_path)

    # Create labels
    labels = np.array(["gesture_1", "gesture_2"], dtype=object)
    labels_path = os.path.join(temp_models_dir, "class_labels.npy")
    np.save(labels_path, labels)

    return model_path, labels_path


@pytest.fixture
def mock_shared_state(temp_models_dir, temp_data_dir, mock_rf_model):
    """Create a mock shared_backend_state.json."""
    model_path, labels_path = mock_rf_model

    state = {
        "random_forest": {
            "model_path": model_path,
            "labels_path": labels_path,
            "feature_dimension": 8,
            "feature_schema": "histogram",
            "feature_schema_version": "histogram_v1",
        },
        "lstm": {
            "model_path": os.path.join(temp_models_dir, "gesture_model.h5"),
            "labels_path": os.path.join(temp_models_dir, "wlasl_labels.npy"),
        },
        "dynamic_data": {
            "x_path": os.path.join(temp_data_dir, "X_data.npy"),
            "y_path": os.path.join(temp_data_dir, "y_data.npy"),
            "labels_path": os.path.join(temp_models_dir, "wlasl_labels.npy"),
        },
        "last_updated": "2026-03-29T00:00:00Z",
        "publisher": "test",
    }

    state_path = os.path.join(temp_models_dir, "shared_backend_state.json")
    with open(state_path, "w") as f:
        json.dump(state, f)

    return state_path, state


@pytest.fixture
def sample_csv_data(temp_data_dir):
    """Create sample CSV training data."""
    import csv

    csv_path = os.path.join(temp_data_dir, "sample_data.csv")

    # Write CSV with numeric features and label
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header with 8 features + label
        writer.writerow(["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "label"])
        # Sample rows
        for i in range(10):
            features = [str(np.random.rand()) for _ in range(8)]
            label = f"gesture_{i % 2}"
            writer.writerow(features + [label])

    return csv_path


@pytest.fixture
def invalid_csv_data(temp_data_dir):
    """Create invalid CSV for testing error handling."""
    csv_path = os.path.join(temp_data_dir, "invalid_data.csv")

    with open(csv_path, "w") as f:
        f.write("This is not valid CSV data\nBar baz qux\n")

    return csv_path


@pytest.fixture
def sample_image_data():
    """Create a small sample image for testing."""
    import cv2

    # Create a simple image (100x100 BGR)
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Encode to JPEG bytes
    is_success, buffer = cv2.imencode(".jpg", image)
    if is_success:
        return buffer.tobytes()
    else:
        raise RuntimeError("Failed to encode test image")
