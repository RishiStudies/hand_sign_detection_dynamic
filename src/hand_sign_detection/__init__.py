"""Hand Sign Detection - A real-time hand gesture recognition system.

This package provides:
- API server for inference and training
- Model management (RandomForest, LSTM)
- Feature extraction pipelines
- Training services
- Combo detection for gesture sequences
"""

__version__ = "1.0.0"
__author__ = "Development Team"

from hand_sign_detection.core.config import get_settings

__all__ = ["get_settings", "__version__"]
