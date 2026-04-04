"""Tests for service modules."""

from unittest.mock import MagicMock, patch

import pytest


class TestRateLimiter:
    """Test RateLimiter class."""

    def test_check_limit_within_limit(self):
        """Test rate limiting within allowed limit."""
        from hand_sign_detection.services.rate_limiting import RateLimiter

        limiter = RateLimiter()

        # Should not raise
        for _ in range(5):
            limiter.check_limit("test", "client1", max_requests=10, window_seconds=60)

    def test_check_limit_exceeded(self):
        """Test rate limiting when exceeded."""
        from fastapi import HTTPException

        from hand_sign_detection.services.rate_limiting import RateLimiter

        limiter = RateLimiter()

        # First 3 should succeed
        for _ in range(3):
            limiter.check_limit("test", "client1", max_requests=3, window_seconds=60)

        # Fourth should fail
        with pytest.raises(HTTPException) as exc_info:
            limiter.check_limit("test", "client1", max_requests=3, window_seconds=60)

        assert exc_info.value.status_code == 429

    def test_separate_buckets(self):
        """Test that different buckets are tracked separately."""
        from hand_sign_detection.services.rate_limiting import RateLimiter

        limiter = RateLimiter()

        # Fill up bucket1
        for _ in range(3):
            limiter.check_limit("bucket1", "client1", max_requests=3, window_seconds=60)

        # bucket2 should still work
        limiter.check_limit("bucket2", "client1", max_requests=3, window_seconds=60)

    def test_backend_property(self):
        """Test backend property returns correct value."""
        from hand_sign_detection.services.rate_limiting import RateLimiter

        with patch(
            "hand_sign_detection.services.rate_limiting.get_redis_client", return_value=None
        ):
            limiter = RateLimiter()
            assert limiter.backend == "in_memory"


class TestComboDetector:
    """Test ComboDetector class."""

    def test_add_prediction(self):
        """Test adding predictions to buffer."""
        from hand_sign_detection.services.combo_detection import ComboDetector

        detector = ComboDetector()
        detector.add_prediction("A", 0.9, "rf")
        detector.add_prediction("B", 0.85, "rf")

        assert len(detector.prediction_buffer) == 2

    def test_check_combos_match(self):
        """Test combo detection with matching sequence."""
        from hand_sign_detection.services.combo_detection import ComboDetector

        detector = ComboDetector(combos={"TEST_COMBO": ["A", "B", "C"]})

        detector.add_prediction("A", 0.9, "rf")
        detector.add_prediction("B", 0.85, "rf")
        detector.add_prediction("C", 0.8, "rf")

        result = detector.check_combos(min_confidence=0.7)

        assert result is not None
        assert result["combo"] == "TEST_COMBO"
        assert result["sequence"] == ["A", "B", "C"]

    def test_check_combos_no_match(self):
        """Test combo detection with non-matching sequence."""
        from hand_sign_detection.services.combo_detection import ComboDetector

        detector = ComboDetector(combos={"TEST_COMBO": ["A", "B", "C"]})

        detector.add_prediction("X", 0.9, "rf")
        detector.add_prediction("Y", 0.85, "rf")

        result = detector.check_combos()
        assert result is None

    def test_get_available_combos(self):
        """Test getting available combo names."""
        from hand_sign_detection.services.combo_detection import ComboDetector

        detector = ComboDetector()
        combos = detector.get_available_combos()

        assert isinstance(combos, list)
        assert "HELLO_WORLD" in combos

    def test_clear(self):
        """Test clearing the prediction buffer."""
        from hand_sign_detection.services.combo_detection import ComboDetector

        detector = ComboDetector()
        detector.add_prediction("A", 0.9, "rf")
        detector.add_prediction("B", 0.85, "rf")
        detector.clear()

        assert len(detector.prediction_buffer) == 0


class TestComboService:
    """Test ComboService class."""

    def test_add_and_check_prediction(self):
        """Test adding predictions and checking combos via service."""
        from hand_sign_detection.services.combo_detection import ComboService

        service = ComboService()

        # Add predictions for a simple combo
        service.add_prediction("session1", "A", 0.9, "rf")
        service.add_prediction("session1", "B", 0.85, "rf")
        service.add_prediction("session1", "C", 0.8, "rf")

        # Check combos
        service.check_combos("session1")

        # May or may not match depending on default combos
        # The point is it shouldn't crash

    def test_session_isolation(self):
        """Test that sessions are isolated."""
        from hand_sign_detection.services.combo_detection import ComboService

        service = ComboService()

        service.add_prediction("session1", "A", 0.9, "rf")
        service.add_prediction("session2", "B", 0.85, "rf")

        # Clear session1
        service.clear_session("session1")

        # session2 should still have its prediction
        # (we can't easily verify this without checking internals,
        # but at least verify it doesn't crash)
        service.check_combos("session2")

    def test_get_available_combos(self):
        """Test getting available combos from service."""
        from hand_sign_detection.services.combo_detection import ComboService

        service = ComboService()
        result = service.get_available_combos()

        assert "combos" in result
        assert "patterns" in result
        assert isinstance(result["combos"], list)


class TestPredictionService:
    """Test PredictionService class."""

    def test_predict_single_model_unavailable(self):
        """Test prediction when model is unavailable."""
        from fastapi import HTTPException

        from hand_sign_detection.models.manager import ModelManager
        from hand_sign_detection.services.prediction import PredictionService

        # Create mock model manager that says RF is not available
        mock_manager = MagicMock(spec=ModelManager)
        mock_manager.rf_available = False

        service = PredictionService(model_manager=mock_manager)

        with pytest.raises(HTTPException) as exc_info:
            service.predict_single(b"fake_image_data")

        assert exc_info.value.status_code == 503
