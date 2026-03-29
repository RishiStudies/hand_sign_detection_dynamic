"""Integration tests for API endpoints."""

import json
import os
import sys
from pathlib import Path
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def api_client(monkeypatch, mock_shared_state, temp_models_dir):
    """Create a FastAPI test client with mocked models."""
    
    # Mock environment
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000")
    monkeypatch.setenv("FEATURE_SCHEMA", "histogram")
    monkeypatch.setenv("TRAINING_API_KEY", "test-api-key-123456")
    
    # Mock load_shared_state to return our test state
    state_path, state_dict = mock_shared_state
    
    def mock_load_shared_state():
        return state_dict
    
    def mock_resolve_shared_path(model_type, path_key):
        return state_dict[model_type][path_key]
    
    # Patch before importing app
    with patch("api_server.load_shared_state", mock_load_shared_state):
        with patch("api_server.resolve_shared_path", mock_resolve_shared_path):
            # Import and setup app
            import api_server
            
            # Mock model loading
            api_server.model = MagicMock()
            api_server.labels = np.array(["gesture_1", "gesture_2"])
            api_server.n_features = 8
            
            yield TestClient(api_server.app)


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


class TestPredictEndpoint:
    """Test /predict inference endpoint."""
    
    def test_predict_success(self, api_client, sample_image_data, mocker):
        """Test successful prediction."""
        # Mock RF model prediction
        api_client.app.extra["model"].predict_proba.return_value = np.array([[0.9, 0.1]])
        
        response = api_client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_data, "image/jpeg")},
            headers={"x-session-id": "test-session"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "prob" in data
        assert "backend_mode" in data
    
    def test_predict_rate_limit(self, api_client, sample_image_data, monkeypatch):
        """Test rate limiting on /predict endpoint."""
        # Set very low rate limit
        monkeypatch.setenv("MAX_PREDICT_REQUESTS_PER_WINDOW", "1")
        
        # Make multiple requests
        for i in range(3):
            response = api_client.post(
                "/predict",
                files={"file": ("test.jpg", sample_image_data, "image/jpeg")}
            )
            
            if i == 0:
                assert response.status_code == 200
            else:
                assert response.status_code == 429  # Too Many Requests


class TestTrainEndpoint:
    """Test /train training endpoint."""
    
    def test_train_requires_api_key(self, api_client):
        """Test that /train endpoint requires API key."""
        response = api_client.post(
            "/train",
            data={"labels_input": ["gesture_1"]},
            files={"samples": ("test.jpg", b"fake image data", "image/jpeg")}
        )
        
        # Should be rejected without API key (403 or 401)
        assert response.status_code in (401, 403)
    
    def test_train_with_valid_api_key(self, api_client, sample_image_data, mocker):
        """Test /train with valid API key."""
        # Mock job queue
        from io import BytesIO
        
        mock_job = MagicMock()
        mock_job.id = "job-123"
        
        mocker.patch("api_server.enqueue_named_job", return_value=mock_job)
        mocker.patch("api_server.is_job_queue_available", return_value=True)
        
        response = api_client.post(
            "/train",
            data={"labels_input": ["gesture_1", "gesture_2"]},
            files=[
                ("samples", ("test1.jpg", sample_image_data, "image/jpeg")),
                ("samples", ("test2.jpg", sample_image_data, "image/jpeg"))
            ],
            headers={"x-api-key": "test-api-key-123456"}
        )
        
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "queued"
        assert data["job_id"] == "job-123"


class TestTrainCSVEndpoint:
    """Test /train_csv endpoint."""
    
    def test_train_csv_requires_api_key(self, api_client, sample_csv_data):
        """Test that /train_csv requires API key."""
        with open(sample_csv_data, "rb") as f:
            response = api_client.post(
                "/train_csv",
                files={"file": ("data.csv", f, "text/csv")}
            )
        
        assert response.status_code in (401, 403)
    
    def test_train_csv_valid(self, api_client, sample_csv_data, mocker):
        """Test /train_csv with valid CSV."""
        mock_job = MagicMock()
        mock_job.id = "job-456"
        
        mocker.patch("api_server.enqueue_named_job", return_value=mock_job)
        mocker.patch("api_server.is_job_queue_available", return_value=True)
        
        with open(sample_csv_data, "rb") as f:
            response = api_client.post(
                "/train_csv",
                files={"file": ("data.csv", f, "text/csv")},
                headers={"x-api-key": "test-api-key-123456"}
            )
        
        assert response.status_code == 202
        data = response.json()
        assert data["job_type"] == "train_rf_csv"


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
        response = api_client.post(
            "/clear_combos",
            headers={"x-session-id": "test-session"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "cleared"


class TestJobTracking:
    """Test job tracking endpoint."""
    
    def test_get_job_status(self, api_client, mocker):
        """Test /jobs/{job_id} endpoint."""
        mock_status = {
            "status": "finished",
            "result": {
                "job_name": "train_rf_samples",
                "success": True
            }
        }
        
        mocker.patch(
            "api_server.get_job_status",
            return_value=mock_status
        )
        mocker.patch("api_server.is_job_queue_available", return_value=True)
        
        response = api_client.get("/jobs/test-job-id")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "finished"
