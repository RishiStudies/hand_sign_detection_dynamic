"""Centralized configuration using Pydantic Settings.

All environment variables and configuration are managed here.
Use `get_settings()` to get a cached Settings instance.
"""

import importlib.util
import os
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings have sensible defaults for development.
    Production deployments should set appropriate values via environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === Paths ===
    project_root: str = Field(
        default_factory=lambda: os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
    )

    @property
    def data_dir(self) -> str:
        return os.path.join(self.project_root, "data")

    @property
    def models_dir(self) -> str:
        return os.path.join(self.project_root, "models")

    @property
    def reports_dir(self) -> str:
        return os.path.join(self.project_root, "reports")

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.project_root, "logs")

    @property
    def shared_state_path(self) -> str:
        return os.path.join(self.models_dir, "shared_backend_state.json")

    # === Logging ===
    log_level: str = Field(default="INFO", description="Logging level")
    log_to_file: bool = Field(default=False, description="Enable file logging")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return upper_v

    # === API Server ===
    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        description="Comma-separated list of allowed CORS origins",
    )
    training_api_key: str | None = Field(
        default=None, description="API key for training endpoints (required in production)"
    )
    frontend_url: str = Field(
        default="http://127.0.0.1:3000", description="URL of the frontend application"
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into a list."""
        if not self.cors_origins:
            return ["http://localhost:3000", "http://127.0.0.1:3000"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    # === Upload Limits ===
    max_upload_bytes: int = Field(
        default=2 * 1024 * 1024,  # 2MB
        description="Maximum upload size in bytes for images",
    )
    max_csv_upload_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum upload size in bytes for CSV files",
    )
    max_sequence_frames: int = Field(
        default=30, description="Maximum number of frames in a sequence"
    )
    max_frame_width: int = Field(default=320, description="Maximum frame width for inference")
    max_frame_height: int = Field(default=240, description="Maximum frame height for inference")

    # === Rate Limiting ===
    rate_limit_window_seconds: int = Field(default=60, description="Rate limit window in seconds")
    max_predict_requests_per_window: int = Field(
        default=180, description="Max prediction requests per window"
    )
    max_sequence_requests_per_window: int = Field(
        default=30, description="Max sequence prediction requests per window"
    )
    max_train_requests_per_window: int = Field(
        default=6, description="Max training requests per window"
    )
    max_concurrent_sequence_requests: int = Field(
        default=2, description="Max concurrent sequence inference requests"
    )

    # === Redis ===
    redis_url: str = Field(default="", description="Redis URL for distributed state")
    redis_rate_limit_prefix: str = Field(
        default="hsd:ratelimit", description="Redis key prefix for rate limiting"
    )
    redis_combo_prefix: str = Field(
        default="hsd:combo", description="Redis key prefix for combo state"
    )
    combo_state_ttl_seconds: int = Field(default=300, description="TTL for combo state in Redis")

    # === Job Queue ===
    job_queue_name: str = Field(default="training", description="Name of the RQ job queue")
    job_timeout_seconds: int = Field(
        default=7200,  # 2 hours
        description="Timeout for background jobs",
    )

    # === Feature Extraction ===
    feature_schema: str = Field(
        default="histogram", description="Feature extraction schema (histogram or mediapipe)"
    )

    @field_validator("feature_schema")
    @classmethod
    def validate_feature_schema(cls, v: str) -> str:
        valid_schemas = {"histogram", "mediapipe"}
        lower_v = v.lower().strip()
        if lower_v not in valid_schemas:
            raise ValueError(f"Invalid feature schema: {v}. Must be one of {valid_schemas}")
        return lower_v

    # === Device Profiles ===
    @property
    def device_profiles(self) -> dict:
        """Training device profiles with hyperparameters."""
        return {
            "pi_zero": {
                "rf_estimators": 64,
                "rf_max_depth": 16,
                "rf_n_jobs": 1,
                "max_classes": 8,
                "max_videos_per_class": 3,
                "sequence_length": 20,
                "frame_stride": 3,
                "lstm_low_end": True,
            },
            "full": {
                "rf_estimators": 300,
                "rf_max_depth": 20,
                "rf_n_jobs": -1,
                "max_classes": 50,
                "max_videos_per_class": 8,
                "sequence_length": 30,
                "frame_stride": 1,
                "lstm_low_end": False,
            },
        }

    # === Feature Schema Dimensions ===
    @property
    def feature_schema_dimensions(self) -> dict:
        """Mapping from schema name to feature dimension."""
        return {
            "histogram": 8,
            "mediapipe": 63,
        }

    @property
    def feature_schema_by_dimension(self) -> dict:
        """Reverse mapping from dimension to schema name."""
        return {v: k for k, v in self.feature_schema_dimensions.items()}

    @property
    def feature_schema_version(self) -> str:
        """Current feature schema version string."""
        return f"{self.feature_schema}_v1"

    @property
    def expected_feature_dimension(self) -> int:
        """Expected feature dimension for current schema."""
        return self.feature_schema_dimensions.get(self.feature_schema, 8)

    # === Runtime Capabilities ===
    @property
    def tensorflow_available(self) -> bool:
        """Check if TensorFlow is available."""
        return importlib.util.find_spec("tensorflow") is not None

    @property
    def mediapipe_available(self) -> bool:
        """Check if MediaPipe is available."""
        return importlib.util.find_spec("mediapipe") is not None

    # === Allowed Content Types ===
    @property
    def allowed_image_content_types(self) -> set:
        """Allowed MIME types for image uploads."""
        return {
            "image/jpeg",
            "image/jpg",
            "image/png",
            "application/octet-stream",
        }

    def validate_environment(self) -> tuple:
        """Validate environment and return (warnings, errors)."""
        warnings_list = []
        errors_list = []

        # Check training API key
        if not self.training_api_key:
            warnings_list.append(
                "TRAINING_API_KEY not set - training endpoints are OPEN. "
                "Required for production deployments."
            )
        elif len(self.training_api_key) < 16:
            warnings_list.append("TRAINING_API_KEY is weak (less than 16 characters)")

        # Check Redis
        if not self.redis_url:
            warnings_list.append("Redis not configured, using in-memory fallback")

        return warnings_list, errors_list


@lru_cache
def get_settings() -> Settings:
    """Get cached Settings instance.

    Returns:
        Settings instance loaded from environment
    """
    return Settings()


# Convenience exports for backwards compatibility
def get_feature_schema_dimensions() -> dict:
    """Get feature schema dimensions mapping."""
    return get_settings().feature_schema_dimensions


def get_device_profiles() -> dict:
    """Get device profiles for training."""
    return get_settings().device_profiles
