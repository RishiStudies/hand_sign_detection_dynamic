"""Tests for the core configuration module."""

import pytest


class TestSettings:
    """Test Settings class."""

    def test_default_settings(self):
        """Test default settings values."""
        # Clear cache to ensure fresh settings
        from hand_sign_detection.core.config import Settings, get_settings

        get_settings.cache_clear()

        settings = Settings()

        assert settings.log_level == "INFO"
        assert settings.log_to_file is False
        assert settings.max_upload_bytes == 2 * 1024 * 1024
        assert settings.max_sequence_frames == 30
        assert settings.feature_schema == "histogram"

    def test_log_level_validation(self):
        """Test log level validation."""
        from hand_sign_detection.core.config import Settings

        # Valid log levels should work
        settings = Settings(log_level="DEBUG")
        assert settings.log_level == "DEBUG"

        settings = Settings(log_level="warning")  # lowercase
        assert settings.log_level == "WARNING"

        # Invalid should raise
        with pytest.raises(ValueError):
            Settings(log_level="INVALID")

    def test_feature_schema_validation(self):
        """Test feature schema validation."""
        from hand_sign_detection.core.config import Settings

        settings = Settings(feature_schema="histogram")
        assert settings.feature_schema == "histogram"

        settings = Settings(feature_schema="MEDIAPIPE")  # uppercase
        assert settings.feature_schema == "mediapipe"

        with pytest.raises(ValueError):
            Settings(feature_schema="invalid")

    def test_cors_origins_parsing(self):
        """Test CORS origins list parsing."""
        from hand_sign_detection.core.config import Settings

        settings = Settings(cors_origins="http://localhost:3000,http://example.com")
        assert settings.cors_origins_list == ["http://localhost:3000", "http://example.com"]

        settings = Settings(cors_origins="")
        assert len(settings.cors_origins_list) == 2  # defaults

    def test_device_profiles(self):
        """Test device profiles property."""
        from hand_sign_detection.core.config import Settings

        settings = Settings()
        profiles = settings.device_profiles

        assert "pi_zero" in profiles
        assert "full" in profiles
        assert profiles["full"]["rf_estimators"] == 300
        assert profiles["pi_zero"]["rf_n_jobs"] == 1

    def test_feature_dimensions(self):
        """Test feature dimension mapping."""
        from hand_sign_detection.core.config import Settings

        settings = Settings(feature_schema="histogram")
        assert settings.expected_feature_dimension == 8

        settings = Settings(feature_schema="mediapipe")
        assert settings.expected_feature_dimension == 63

    def test_validate_environment(self):
        """Test environment validation."""
        from hand_sign_detection.core.config import Settings

        # Without API key - should warn
        settings = Settings(training_api_key=None)
        warnings, errors = settings.validate_environment()
        assert len(warnings) > 0
        assert len(errors) == 0

        # With short API key - should warn
        settings = Settings(training_api_key="short")
        warnings, errors = settings.validate_environment()
        assert any("weak" in w.lower() for w in warnings)

    def test_get_settings_caching(self):
        """Test that get_settings returns cached instance."""
        from hand_sign_detection.core.config import get_settings

        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2


class TestFeatureSchemaHelpers:
    """Test feature schema helper functions."""

    def test_get_feature_schema_dimensions(self):
        """Test get_feature_schema_dimensions function."""
        from hand_sign_detection.core.config import get_feature_schema_dimensions

        dims = get_feature_schema_dimensions()
        assert dims["histogram"] == 8
        assert dims["mediapipe"] == 63

    def test_get_device_profiles(self):
        """Test get_device_profiles function."""
        from hand_sign_detection.core.config import get_device_profiles

        profiles = get_device_profiles()
        assert "full" in profiles
        assert "pi_zero" in profiles
