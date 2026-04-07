"""Entry point for the Hand Sign Detection application."""

import uvicorn

from hand_sign_detection.api.app import create_app
from hand_sign_detection.core.config import get_settings


def main():
    """Run the API server."""
    settings = get_settings()
    app = create_app()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
