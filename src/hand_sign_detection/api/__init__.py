"""API module for Hand Sign Detection.

Provides FastAPI application factory and route handlers.
"""

from hand_sign_detection.api.app import create_app

__all__ = ["create_app"]
