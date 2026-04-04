"""Integration tests for API endpoints."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def api_client(monkeypatch, mock_shared_state, temp_models_dir, temp_data_dir):
    """Create a FastAPI test client with mocked models."""

    # Mock environment
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000")
    monkeypatch.setenv("FEATURE_SCHEMA", "histogram")
    monkeypatch.setenv("TRAINING_API_KEY", "test-api-key-123456")

    state_path, state_dict = mock_shared_state

    # Need to clear settings cache before modifying environment
    from hand_sign_detection.core.config import get_settings

    get_settings.cache_clear()

    # Mock resolve_shared_path
    def mock_resolve_shared_path(section, key):
        return state_dict.get(section, {}).get(key, "")

    # Mock load_shared_state
    def mock_load_shared_state():
        return state_dict

    with patch("hand_sign_detection.core.shared_state.load_shared_state", mock_load_shared_state):
        with patch(
            "hand_sign_detection.models.manager.resolve_shared_path", mock_resolve_shared_path
        ):
            with patch(
                "hand_sign_detection.models.manager.load_shared_state", mock_load_shared_state
            ):
                # Create app without auto-loading models
                from hand_sign_detection.api.app import create_app

                app = create_app(auto_load_models=False, validate_env=False)

                # Mock model manager with available RF model
                from hand_sign_detection.models.manager import get_model_manager

                manager = get_model_manager()

                # Load the mock model
                model_path, labels_path = (
                    state_dict["random_forest"]["model_path"],
                    state_dict["random_forest"]["labels_path"],
                )
                if os.path.exists(model_path):
                    import joblib

                    manager._rf_model = joblib.load(model_path)
                    manager._rf_labels = np.load(labels_path, allow_pickle=True)
                    manager._rf_n_features = 8
                else:
                    # Create mock model
                    manager._rf_model = MagicMock()
                    manager._rf_model.predict_proba = MagicMock(return_value=np.array([[0.9, 0.1]]))
                    manager._rf_labels = np.array(["gesture_1", "gesture_2"])
                    manager._rf_n_features = 8

                yield TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_live(self, api_client):
        """Test /health/live endpoint."""
        response = api_client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_ready_with_model(self, api_client):
        """Test /health/ready when models are available."""
        response = api_client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["random_forest"] == "available"

    def test_health_endpoint(self, api_client):
        """Test /health endpoint."""
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestPredictEndpoint:
    """Test /predict inference endpoint."""

    def test_predict_success(self, api_client, sample_image_data):
        """Test successful prediction."""
        response = api_client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_data, "image/jpeg")},
            headers={"x-session-id": "test-session"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "prob" in data
        assert "backend_mode" in data

    def test_predict_empty_file(self, api_client):
        """Test prediction with empty file."""
        response = api_client.post(
            "/predict",
            files={"file": ("test.jpg", b"", "image/jpeg")},
        )

        assert response.status_code == 400


class TestComboDetection:
    """Test combo detection functionality."""

    def test_get_available_combos(self, api_client):
        """Test /combos endpoint returns available combos."""
        response = api_client.get("/combos")
        assert response.status_code == 200
        data = response.json()
        assert "combos" in data
        assert isinstance(data["combos"], list)

    def test_clear_combo_state(self, api_client):
        """Test /clear_combos endpoint."""
        response = api_client.post("/clear_combos", headers={"x-session-id": "test-session"})
        assert response.status_code == 200
        assert response.json()["status"] == "cleared"


class TestTrainEndpoint:
    """Test /train training endpoint."""

    def test_train_requires_api_key(self, api_client, sample_image_data):
        """Test that /train endpoint requires API key."""
        response = api_client.post(
            "/train",
            data={"labels_input": ["gesture_1"]},
            files={"samples": ("test.jpg", sample_image_data, "image/jpeg")},
        )

        # Should be rejected without API key (403)
        assert response.status_code == 403

    def test_train_with_valid_api_key_no_queue(self, api_client, sample_image_data):
        """Test /train with valid API key but no job queue."""
        response = api_client.post(
            "/train",
            data={"labels_input": ["gesture_1", "gesture_2"]},
            files=[
                ("samples", ("test1.jpg", sample_image_data, "image/jpeg")),
                ("samples", ("test2.jpg", sample_image_data, "image/jpeg")),
            ],
            headers={"x-api-key": "test-api-key-123456"},
        )

        # Without Redis, job queue is unavailable
        assert response.status_code == 503


class TestArtifactsEndpoint:
    """Test artifacts endpoint."""

    def test_get_artifacts(self, api_client):
        """Test /artifacts endpoint."""
        response = api_client.get("/artifacts")
        assert response.status_code == 200
        data = response.json()
        assert "random_forest" in data or "last_updated" in data
