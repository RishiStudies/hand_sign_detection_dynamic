"""Unit tests for API server validation and security functions."""

import os
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestCSVValidation:
    """Test CSV schema validation function."""

    def test_valid_csv(self, sample_csv_data, monkeypatch):
        """Test that valid CSV passes validation."""
        from api_server import validate_csv_schema

        # Should not raise
        validate_csv_schema(sample_csv_data)

    def test_invalid_csv_format(self, invalid_csv_data, monkeypatch):
        """Test that malformed CSV is rejected."""
        from api_server import validate_csv_schema

        with pytest.raises(HTTPException) as exc_info:
            validate_csv_schema(invalid_csv_data)

        assert exc_info.value.status_code == 400
        assert "Invalid CSV format" in exc_info.value.detail or "CSV" in exc_info.value.detail

    def test_empty_csv(self, temp_models_dir):
        """Test that empty CSV is rejected."""
        from api_server import validate_csv_schema

        csv_path = os.path.join(temp_models_dir, "empty.csv")
        with open(csv_path, "w") as f:
            f.write("")

        with pytest.raises(HTTPException) as exc_info:
            validate_csv_schema(csv_path)

        assert exc_info.value.status_code == 400


class TestAPIKeyValidation:
    """Test API key requirement function."""

    def test_valid_api_key(self, monkeypatch):
        """Test that valid API key passes."""

        # Mock TRAINING_API_KEY
        monkeypatch.setenv("TRAINING_API_KEY", "test-key-12345")

        # Reimport to get new env value
        import importlib

        import api_server

        importlib.reload(api_server)

        # Should not raise
        api_server.require_training_key("test-key-12345")

    def test_missing_api_key_with_requirement(self, monkeypatch):
        """Test that request without API key fails when required."""

        monkeypatch.setenv("TRAINING_API_KEY", "test-key-12345")

        # Reimport
        import importlib

        import api_server

        importlib.reload(api_server)

        # Should raise 403
        with pytest.raises(HTTPException) as exc_info:
            api_server.require_training_key(None)

        assert exc_info.value.status_code == 403

    def test_invalid_api_key(self, monkeypatch):
        """Test that wrong API key is rejected."""

        monkeypatch.setenv("TRAINING_API_KEY", "correct-key-12345")

        import importlib

        import api_server

        importlib.reload(api_server)

        with pytest.raises(HTTPException) as exc_info:
            api_server.require_training_key("wrong-key")

        assert exc_info.value.status_code == 403


class TestFeatureValidation:
    """Test feature dimension validation."""

    def test_feature_contract_mismatch(self):
        """Test that feature dimension mismatch raises error."""
        from api_server import validate_feature_contract

        with pytest.raises(ValueError) as exc_info:
            validate_feature_contract(8, 63, "Test context")

        assert "feature mismatch" in str(exc_info.value)
        assert "8" in str(exc_info.value)
        assert "63" in str(exc_info.value)

    def test_feature_contract_match(self):
        """Test that matching feature dimensions pass."""
        from api_server import validate_feature_contract

        # Should not raise
        validate_feature_contract(8, 8, "Test context")


class TestLSTMStatusTracking:
    """Test LSTM availability tracking."""

    def test_lstm_unavailable_flag_on_missing_model(self, mocker, monkeypatch):
        """Test that lstm_available flag is False when model is missing."""
        from api_server import load_lstm_model

        # Mock resolve_shared_path to return non-existent path
        mocker.patch("api_server.resolve_shared_path", return_value="/nonexistent/path/model.h5")

        result = load_lstm_model()

        assert result is False


class TestDataUploadValidation:
    """Test data upload validation."""

    def test_oversized_upload_rejected(self):
        """Test that oversized files are rejected."""
        from api_server import validate_upload

        # Create mock upload
        class MockUpload:
            def __init__(self):
                self.content_type = "image/jpeg"

        file = MockUpload()
        data = b"x" * (100 * 1024 * 1024)  # 100MB

        with pytest.raises(HTTPException) as exc_info:
            validate_upload(file, data)

        assert exc_info.value.status_code == 413  # Payload Too Large

    def test_invalid_content_type_rejected(self):
        """Test that invalid content types are rejected."""
        from api_server import validate_upload

        class MockUpload:
            def __init__(self):
                self.content_type = "application/pdf"

        file = MockUpload()
        data = b"some data"

        with pytest.raises(HTTPException) as exc_info:
            validate_upload(file, data)

        assert exc_info.value.status_code == 415  # Unsupported Media Type

    def test_empty_file_rejected(self):
        """Test that empty files are rejected."""
        from api_server import validate_upload

        class MockUpload:
            def __init__(self):
                self.content_type = "image/jpeg"

        file = MockUpload()
        data = b""

        with pytest.raises(HTTPException) as exc_info:
            validate_upload(file, data)

        assert exc_info.value.status_code == 400
