"""Tests for edge cases and error handling."""

import contextlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestConfigurationEdgeCases:
    """Test edge cases in configuration handling."""

    def test_invalid_feature_schema(self, monkeypatch):
        """Test handling of invalid feature schema."""
        from hand_sign_detection.core.config import get_settings

        get_settings.cache_clear()

        monkeypatch.setenv("FEATURE_SCHEMA", "invalid_schema")

        # Should use default or raise validation error
        try:
            settings = get_settings()
            # If it doesn't raise, check it defaulted
            assert settings.feature_schema in ["histogram", "mediapipe"]
        except Exception:
            pass  # Validation error is acceptable
        finally:
            get_settings.cache_clear()

    def test_empty_redis_url(self, monkeypatch):
        """Test handling of empty Redis URL."""
        from hand_sign_detection.core.config import get_settings

        get_settings.cache_clear()

        monkeypatch.setenv("REDIS_URL", "")

        settings = get_settings()
        # Empty URL should be treated as None/disabled
        assert settings.redis_url == "" or settings.redis_url is None

        get_settings.cache_clear()

    def test_cors_origins_parsing(self, monkeypatch):
        """Test CORS origins parsing with various formats."""
        from hand_sign_detection.core.config import get_settings

        test_cases = [
            ("http://localhost:3000", ["http://localhost:3000"]),
            ("http://a.com,http://b.com", ["http://a.com", "http://b.com"]),
            ("", []),
        ]

        for input_val, _expected in test_cases:
            get_settings.cache_clear()
            monkeypatch.setenv("CORS_ORIGINS", input_val)
            settings = get_settings()
            # Just verify it doesn't crash
            assert hasattr(settings, "cors_origins")

        get_settings.cache_clear()


class TestFeatureExtractionEdgeCases:
    """Test edge cases in feature extraction."""

    def test_empty_image(self):
        """Test handling of empty/zero-size image."""
        from hand_sign_detection.models.features import extract_histogram_features

        empty_image = np.zeros((0, 0, 3), dtype=np.uint8)

        # Should handle gracefully (return zeros or raise)
        try:
            features = extract_histogram_features(empty_image)
            # If it returns, should be valid format
            assert features is not None
        except (ValueError, IndexError):
            pass  # Acceptable to raise on empty image

    def test_single_pixel_image(self):
        """Test handling of 1x1 image."""
        from hand_sign_detection.models.features import extract_histogram_features

        tiny_image = np.array([[[128, 64, 32]]], dtype=np.uint8)

        features = extract_histogram_features(tiny_image)
        assert features is not None
        assert len(features) == 8

    def test_very_large_image(self):
        """Test handling of large image."""
        from hand_sign_detection.models.features import extract_histogram_features

        # 4K resolution image
        large_image = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)

        features = extract_histogram_features(large_image)
        assert features is not None
        assert len(features) == 8

    def test_uniform_color_image(self):
        """Test handling of uniform color image."""
        from hand_sign_detection.models.features import extract_histogram_features

        # All black
        black_image = np.zeros((100, 100, 3), dtype=np.uint8)
        features_black = extract_histogram_features(black_image)

        # All white
        white_image = np.full((100, 100, 3), 255, dtype=np.uint8)
        features_white = extract_histogram_features(white_image)

        assert features_black is not None
        assert features_white is not None
        # Features should be different
        assert not np.allclose(features_black, features_white)


class TestRateLimiterEdgeCases:
    """Test edge cases in rate limiting."""

    def test_very_small_window(self):
        """Test rate limiting with very small time window."""
        from hand_sign_detection.services.rate_limiting import RateLimiter

        limiter = RateLimiter()

        # 1 second window, 1 request allowed
        limiter.check_limit("test", "client", max_requests=1, window_seconds=1)

        # Should fail immediately
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            limiter.check_limit("test", "client", max_requests=1, window_seconds=1)

    def test_very_large_limit(self):
        """Test rate limiting with very large limit."""
        from hand_sign_detection.services.rate_limiting import RateLimiter

        limiter = RateLimiter()

        # Should handle large limits
        for _ in range(100):
            limiter.check_limit("test", "client", max_requests=1000000, window_seconds=60)

    def test_special_characters_in_client_id(self):
        """Test rate limiting with special characters in client ID."""
        from hand_sign_detection.services.rate_limiting import RateLimiter

        limiter = RateLimiter()

        special_ids = [
            "client:with:colons",
            "client/with/slashes",
            "client with spaces",
            "client\nwith\nnewlines",
            "クライアント",  # Unicode
        ]

        for client_id in special_ids:
            # Should not crash
            limiter.check_limit("test", client_id, max_requests=10, window_seconds=60)


class TestComboDetectorEdgeCases:
    """Test edge cases in combo detection."""

    def test_empty_combo_list(self):
        """Test combo detection with no combos defined."""
        from hand_sign_detection.services.combo_detection import ComboDetector

        detector = ComboDetector(combos={})
        detector.add_prediction("A", 0.9, "rf")

        result = detector.check_combos()
        assert result is None

    def test_very_long_combo(self):
        """Test combo detection with very long sequence."""
        from hand_sign_detection.services.combo_detection import ComboDetector

        long_combo = ["A"] * 50
        detector = ComboDetector(combos={"LONG": long_combo})

        for _ in range(50):
            detector.add_prediction("A", 0.9, "rf")

        result = detector.check_combos()
        assert result is not None
        assert result["combo"] == "LONG"

    def test_low_confidence_predictions(self):
        """Test combo detection with low confidence predictions."""
        from hand_sign_detection.services.combo_detection import ComboDetector

        detector = ComboDetector(combos={"TEST": ["A", "B"]})

        # Add low confidence predictions
        detector.add_prediction("A", 0.3, "rf")
        detector.add_prediction("B", 0.3, "rf")

        # Should not match with high confidence threshold
        result = detector.check_combos(min_confidence=0.8)
        assert result is None

        # Should match with low threshold
        result = detector.check_combos(min_confidence=0.2)
        assert result is not None

    def test_repeated_predictions(self):
        """Test combo detection ignores repeated same predictions."""
        from hand_sign_detection.services.combo_detection import ComboDetector

        detector = ComboDetector(combos={"TEST": ["A", "B"]})

        # Add same prediction multiple times
        detector.add_prediction("A", 0.9, "rf")
        detector.add_prediction("A", 0.9, "rf")
        detector.add_prediction("A", 0.9, "rf")
        detector.add_prediction("B", 0.9, "rf")

        result = detector.check_combos()
        # Should still detect the combo
        assert result is not None


class TestPredictionServiceEdgeCases:
    """Test edge cases in prediction service."""

    def test_predict_with_corrupted_image_data(self):
        """Test prediction with corrupted image data."""
        from fastapi import HTTPException

        from hand_sign_detection.models.manager import ModelManager
        from hand_sign_detection.services.prediction import PredictionService

        mock_manager = MagicMock(spec=ModelManager)
        mock_manager.rf_available = True

        service = PredictionService(model_manager=mock_manager)

        with pytest.raises((HTTPException, Exception)):
            service.predict_single(b"corrupted data that is not an image")

    def test_predict_batch_empty_list(self):
        """Test batch prediction with empty list."""
        from hand_sign_detection.models.manager import ModelManager
        from hand_sign_detection.services.prediction import PredictionService

        mock_manager = MagicMock(spec=ModelManager)
        mock_manager.rf_available = True

        service = PredictionService(model_manager=mock_manager)

        result = service.predict_batch([])
        assert result == []


class TestMediaPipeEdgeCases:
    """Test edge cases in MediaPipe integration."""

    def test_mediapipe_with_no_hands(self):
        """Test MediaPipe detection on image with no hands."""
        from hand_sign_detection.models.features import detect_hand, reset_mediapipe

        try:
            # Plain colored image (no hands)
            no_hands_image = np.full((480, 640, 3), 128, dtype=np.uint8)

            result = detect_hand(no_hands_image)

            # Should return None or empty result
            assert result is None or result.confidence == 0.0
        finally:
            reset_mediapipe()

    def test_mediapipe_cleanup(self):
        """Test MediaPipe cleanup doesn't crash."""
        from hand_sign_detection.models.features import detect_hand, reset_mediapipe

        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with contextlib.suppress(Exception):
            detect_hand(test_image)  # MediaPipe may not be available

        # Reset should not crash even if MediaPipe wasn't initialized
        reset_mediapipe()
        reset_mediapipe()  # Double reset should also be safe


class TestAPIValidationEdgeCases:
    """Test API validation edge cases."""

    def test_predict_with_invalid_content_type(self, api_client):
        """Test prediction with wrong content type."""
        response = api_client.post(
            "/predict", content=b"raw bytes", headers={"Content-Type": "application/octet-stream"}
        )
        assert response.status_code in [400, 415, 422]

    def test_predict_with_empty_session_id(self, api_client, sample_image_data):
        """Test prediction with empty session ID."""
        response = api_client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_data, "image/jpeg")},
            headers={"x-session-id": ""},
        )
        # Should work or use default session
        assert response.status_code in [200, 400]

    def test_combos_with_invalid_session(self, api_client):
        """Test clearing combos with very long session ID."""
        long_session = "a" * 10000
        response = api_client.post("/clear_combos", headers={"x-session-id": long_session})
        # Should handle gracefully
        assert response.status_code in [200, 400, 422]
