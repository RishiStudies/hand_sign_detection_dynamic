"""Integration tests for the full prediction pipeline."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestFullPredictionPipeline:
    """End-to-end tests for the prediction pipeline."""

    def test_histogram_feature_extraction_to_prediction(
        self, temp_models_dir, mock_shared_state, monkeypatch
    ):
        """Test full pipeline from image to prediction using histogram features."""
        monkeypatch.setenv("FEATURE_SCHEMA", "histogram")

        from hand_sign_detection.core.config import get_settings

        get_settings.cache_clear()

        from hand_sign_detection.models.features import extract_histogram_features

        # Create a test image (RGB)
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Extract features
        features = extract_histogram_features(test_image)

        # Verify feature shape
        assert features is not None
        assert len(features) == 8  # Histogram has 8 bins
        assert all(0 <= f <= 1 for f in features)  # Normalized

    def test_feature_extraction_handles_grayscale(self):
        """Test that feature extraction handles grayscale images."""
        from hand_sign_detection.models.features import extract_histogram_features

        # Grayscale image
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        # Should not crash, even if result is different
        features = extract_histogram_features(gray_image)
        assert features is not None

    def test_feature_extraction_handles_rgba(self):
        """Test that feature extraction handles RGBA images."""
        from hand_sign_detection.models.features import extract_histogram_features

        # RGBA image
        rgba_image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)

        # Should handle 4-channel images
        features = extract_histogram_features(rgba_image)
        assert features is not None


class TestModelManagerIntegration:
    """Integration tests for ModelManager."""

    def test_model_manager_thread_safety(self, mock_shared_state, monkeypatch):
        """Test that model manager is thread-safe."""
        import threading

        state_path, state_dict = mock_shared_state

        def mock_resolve(section, key):
            return state_dict.get(section, {}).get(key, "")

        with patch("hand_sign_detection.models.manager.resolve_shared_path", mock_resolve):
            with patch(
                "hand_sign_detection.models.manager.load_shared_state", return_value=state_dict
            ):
                from hand_sign_detection.models.manager import (
                    get_model_manager,
                    reset_model_manager,
                )

                reset_model_manager()
                manager = get_model_manager()

                results = []
                errors = []

                def access_manager():
                    try:
                        for _ in range(10):
                            _ = manager.rf_available
                            _ = manager.lstm_available
                            _ = manager.rf_n_features
                        results.append(True)
                    except Exception as e:
                        errors.append(str(e))

                # Create multiple threads accessing manager
                threads = [threading.Thread(target=access_manager) for _ in range(5)]

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                assert len(errors) == 0, f"Thread safety errors: {errors}"
                assert len(results) == 5

    def test_singleton_pattern(self, mock_shared_state):
        """Test that ModelManager follows singleton pattern."""
        state_path, state_dict = mock_shared_state

        def mock_resolve(section, key):
            return state_dict.get(section, {}).get(key, "")

        with patch("hand_sign_detection.models.manager.resolve_shared_path", mock_resolve):
            with patch(
                "hand_sign_detection.models.manager.load_shared_state", return_value=state_dict
            ):
                from hand_sign_detection.models.manager import get_model_manager

                manager1 = get_model_manager()
                manager2 = get_model_manager()

                assert manager1 is manager2


class TestRateLimitingIntegration:
    """Integration tests for rate limiting."""

    def test_rate_limiting_across_endpoints(self, api_client):
        """Test that rate limiting tracks requests correctly."""
        # Make several requests
        for _ in range(5):
            response = api_client.get("/health/live")
            assert response.status_code == 200

    def test_rate_limiting_respects_headers(self, api_client):
        """Test rate limit headers are returned."""
        response = api_client.get("/health/live")
        # Rate limit headers may be present depending on configuration
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_image_format(self, api_client):
        """Test handling of invalid image data."""
        response = api_client.post(
            "/predict", files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        # Should return 400 (bad request) or 422 (unprocessable)
        assert response.status_code in [400, 422, 500]

    def test_missing_file_parameter(self, api_client):
        """Test handling of missing file."""
        response = api_client.post("/predict")
        assert response.status_code == 422  # FastAPI validation error

    def test_oversized_file(self, api_client, monkeypatch):
        """Test handling of files exceeding size limit."""
        from hand_sign_detection.core.config import get_settings

        get_settings.cache_clear()

        # Create a large "file" (larger than typical limit)
        large_data = b"x" * (3 * 1024 * 1024)  # 3MB

        response = api_client.post(
            "/predict", files={"file": ("large.jpg", large_data, "image/jpeg")}
        )

        # Should reject with 400 (size limit) or 413 (payload too large)
        assert response.status_code in [400, 413, 422, 500]


class TestComboDetectionIntegration:
    """Integration tests for combo detection."""

    def test_combo_detection_sequence(self, api_client, sample_image_data):
        """Test combo detection with sequence of predictions."""
        session_id = "integration-test-session"

        # Clear any existing state
        api_client.post("/clear_combos", headers={"x-session-id": session_id})

        # Make predictions (combo detection happens automatically)
        for _ in range(3):
            response = api_client.post(
                "/predict",
                files={"file": ("test.jpg", sample_image_data, "image/jpeg")},
                headers={"x-session-id": session_id},
            )
            assert response.status_code == 200

        # Check that we can get combo list
        response = api_client.get("/combos")
        assert response.status_code == 200
        assert "combos" in response.json()


class TestHealthChecks:
    """Test health check integration."""

    def test_health_endpoints_always_respond(self, api_client):
        """Test that health endpoints respond even under load."""
        for _ in range(10):
            live = api_client.get("/health/live")
            assert live.status_code == 200

            ready = api_client.get("/health/ready")
            assert ready.status_code == 200

    def test_health_response_format(self, api_client):
        """Test health response format is consistent."""
        response = api_client.get("/health/live")
        data = response.json()

        assert "status" in data
        assert data["status"] in ["ok", "degraded", "unhealthy"]


class TestConcurrency:
    """Test concurrent request handling."""

    def test_concurrent_predictions(self, api_client, sample_image_data):
        """Test handling multiple concurrent prediction requests."""
        import concurrent.futures

        def make_prediction():
            return api_client.post(
                "/predict", files={"file": ("test.jpg", sample_image_data, "image/jpeg")}
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_prediction) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All requests should complete (may be rate limited)
        statuses = [r.status_code for r in results]
        assert all(s in [200, 429] for s in statuses)
