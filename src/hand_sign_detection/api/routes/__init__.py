"""API route modules.

Contains:
- health: Health check endpoints
- predict: Inference endpoints
- training: Model training endpoints
- combos: Combo detection endpoints
"""

from hand_sign_detection.api.routes import combos, health, predict, training

__all__ = ["health", "predict", "training", "combos"]
