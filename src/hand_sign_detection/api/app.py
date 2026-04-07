"""FastAPI application factory.

Provides create_app() function for creating configured FastAPI instances.
Includes production-ready features like graceful shutdown and structured logging.
"""

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from hand_sign_detection.api.routes import combos, health, predict, training
from hand_sign_detection.core.config import get_settings
from hand_sign_detection.core.logging import setup_logging
from hand_sign_detection.models.features import reset_mediapipe
from hand_sign_detection.models.manager import get_model_manager

logger = logging.getLogger("hand_sign_detection.api")


class ApplicationState:
    """Shared application state for lifecycle management."""

    def __init__(self):
        self.shutting_down = False
        self.ready = False


_app_state = ApplicationState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager for startup/shutdown.

    Handles:
    - Model loading on startup
    - Resource cleanup on shutdown
    - Graceful shutdown signaling
    """
    _ = get_settings()  # Validate settings on startup

    # Startup
    logger.info("Starting up Hand Sign Detection API...")

    # Load models
    model_manager = get_model_manager()
    load_results = model_manager.load_all()

    if not load_results["random_forest"]:
        logger.error("CRITICAL: RandomForest model failed to load at startup")
    else:
        logger.info("RandomForest model loaded successfully")

    if not load_results["lstm"]:
        logger.warning("LSTM model not available - sequence inference disabled")
    else:
        logger.info("LSTM model loaded successfully")

    _app_state.ready = True
    logger.info("API Server initialized successfully")

    yield

    # Shutdown
    logger.info("Shutting down Hand Sign Detection API...")
    _app_state.shutting_down = True
    _app_state.ready = False

    # Cleanup resources
    reset_mediapipe()

    # Allow in-flight requests to complete
    await asyncio.sleep(0.5)

    logger.info("Shutdown complete")


def create_app(
    auto_load_models: bool = True,
    validate_env: bool = True,
) -> FastAPI:
    """Create and configure FastAPI application.

    Args:
        auto_load_models: Whether to load models on startup
        validate_env: Whether to validate environment on startup

    Returns:
        Configured FastAPI application instance
    """
    # Setup logging first
    setup_logging()
    logger.info("Creating FastAPI application...")

    settings = get_settings()

    # Validate environment
    if validate_env:
        warnings_list, errors_list = settings.validate_environment()
        for warning in warnings_list:
            logger.warning("⚠ %s", warning)
        for error in errors_list:
            logger.error("✗ %s", error)
        if errors_list:
            raise RuntimeError(f"Environment validation failed: {'; '.join(errors_list)}")

    # Create FastAPI app with lifespan if auto_load_models
    app_kwargs = {
        "title": "Hand Sign Detection API",
        "description": "Real-time hand sign/gesture recognition with ML backend",
        "version": "2.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }

    if auto_load_models:
        app_kwargs["lifespan"] = lifespan

    app = FastAPI(**app_kwargs)

    # Add middleware
    app.add_middleware(GZipMiddleware, minimum_size=500)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception on %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.log_level == "DEBUG" else None,
            },
        )

    # Include routers
    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(training.router)
    app.include_router(combos.router)

    # Root endpoint serving HTML UI
    @app.get("/", include_in_schema=False)
    def index():
        """Serve the web UI HTML file."""
        try:
            html_path = os.path.join(settings.project_root, "index.html")
            if os.path.exists(html_path):
                with open(html_path, encoding="utf-8") as f:
                    return HTMLResponse(content=f.read())
            else:
                return JSONResponse(
                    content={
                        "message": "Hand Sign Detection API",
                        "docs": "/docs",
                        "health": "/health",
                    }
                )
        except Exception as e:
            logger.error(f"Error serving index.html: {e}")
            return JSONResponse(
                content={
                    "error": str(e),
                    "docs": "/docs",
                }
            )

    return app


# Default app instance for uvicorn
app = create_app()
