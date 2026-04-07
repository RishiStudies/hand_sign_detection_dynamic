"""Tests for the ModelManager class."""

from unittest.mock import patch

import numpy as np
import pytest


class TestModelManager:
    """Test ModelManager class."""

    def test_initial_state(self):
        """Test initial ModelManager state."""
        from hand_sign_detection.models.manager import ModelManager

        manager = ModelManager()

        assert not manager.rf_available
        assert not manager.lstm_available
        assert manager.rf_n_features == 8

    def test_load_rf_model_success(self, mock_rf_model, monkeypatch):
        """Test successful RF model loading."""
        from hand_sign_detection.models.manager import ModelManager

        model_path, labels_path = mock_rf_model

        # Mock resolve_shared_path to return our test paths
        def mock_resolve(section, key):
            if section == "random_forest":
                if key == "model_path":
                    return model_path
                if key == "labels_path":
                    return labels_path
            raise KeyError(f"{section}.{key}")

        with patch("hand_sign_detection.models.manager.resolve_shared_path", mock_resolve):
            with patch(
                "hand_sign_detection.models.manager.load_shared_state",
                return_value={
                    "random_forest": {
                        "feature_dimension": 8,
                        "feature_schema": "histogram",
                    }
                },
            ):
                manager = ModelManager()
                result = manager.load_rf_model()

        assert result is True
        assert manager.rf_available
        assert manager.rf_n_features == 8

    def test_load_rf_model_file_not_found(self):
        """Test RF model loading with missing file."""
        from hand_sign_detection.models.manager import ModelManager

        def mock_resolve(section, key):
            return "/nonexistent/path/model.pkl"

        with patch("hand_sign_detection.models.manager.resolve_shared_path", mock_resolve):
            manager = ModelManager()
            result = manager.load_rf_model()

        assert result is False
        assert not manager.rf_available

    def test_predict_rf(self, mock_rf_model, monkeypatch):
        """Test RF prediction."""
        from hand_sign_detection.models.manager import ModelManager

        model_path, labels_path = mock_rf_model

        def mock_resolve(section, key):
            if section == "random_forest":
                if key == "model_path":
                    return model_path
                if key == "labels_path":
                    return labels_path
            raise KeyError(f"{section}.{key}")

        with patch("hand_sign_detection.models.manager.resolve_shared_path", mock_resolve):
            with patch(
                "hand_sign_detection.models.manager.load_shared_state",
                return_value={"random_forest": {"feature_dimension": 8}},
            ):
                manager = ModelManager()
                manager.load_rf_model()

        # Create test features
        features = np.random.rand(1, 8).astype(np.float32)
        label, confidence = manager.predict_rf(features)

        assert isinstance(label, str)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_predict_rf_not_available(self):
        """Test RF prediction when model not loaded."""
        from hand_sign_detection.models.manager import ModelManager

        manager = ModelManager()
        features = np.random.rand(1, 8).astype(np.float32)

        with pytest.raises(RuntimeError, match="RandomForest model not available"):
            manager.predict_rf(features)

    def test_get_status(self):
        """Test get_status method."""
        from hand_sign_detection.models.manager import ModelManager

        manager = ModelManager()
        status = manager.get_status()

        assert "random_forest" in status
        assert "lstm" in status
        assert "tensorflow" in status
        assert "rf_feature_schema" in status

    def test_singleton_pattern(self):
        """Test get_model_manager returns same instance."""
        from hand_sign_detection.models.manager import (
            get_model_manager,
            reset_model_manager,
        )

        reset_model_manager()

        manager1 = get_model_manager()
        manager2 = get_model_manager()

        assert manager1 is manager2


class TestFeatureExtraction:
    """Test feature extraction functions."""

    def test_extract_histogram_features(self, sample_image_data):
        """Test histogram feature extraction."""
        from hand_sign_detection.models.features import extract_features_from_bytes

        with patch("hand_sign_detection.models.features.get_settings") as mock_settings:
            mock_settings.return_value.feature_schema = "histogram"
            mock_settings.return_value.feature_schema_dimensions = {"histogram": 8}

            features = extract_features_from_bytes(sample_image_data)

        assert features.shape == (8,)
        assert np.isclose(features.sum(), 1.0, atol=0.01)  # Should be normalized

    def test_extract_features_invalid_image(self):
        """Test feature extraction with invalid image data."""
        from hand_sign_detection.models.features import extract_features_from_bytes

        with pytest.raises(ValueError, match="Invalid image payload"):
            extract_features_from_bytes(b"not an image")

    def test_get_expected_feature_dimension(self):
        """Test get_expected_feature_dimension function."""
        from hand_sign_detection.models.features import get_expected_feature_dimension

        with patch("hand_sign_detection.models.features.get_settings") as mock_settings:
            mock_settings.return_value.feature_schema = "histogram"
            mock_settings.return_value.feature_schema_dimensions = {
                "histogram": 8,
                "mediapipe": 63,
            }

            assert get_expected_feature_dimension() == 8
            assert get_expected_feature_dimension("mediapipe") == 63

    def test_validate_feature_contract(self):
        """Test feature contract validation."""
        from hand_sign_detection.models.features import validate_feature_contract

        # Should not raise
        validate_feature_contract(8, 8, "Test")

        # Should raise
        with pytest.raises(ValueError, match="feature mismatch"):
            validate_feature_contract(8, 10, "Test")
