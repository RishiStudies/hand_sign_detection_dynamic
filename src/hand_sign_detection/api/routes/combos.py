"""Combo detection endpoints."""

from fastapi import APIRouter, Depends, Header

from hand_sign_detection.api.dependencies import combo_service_dependency
from hand_sign_detection.services.combo_detection import ComboService

router = APIRouter(tags=["Combos"])


@router.get("/combos")
def get_combos(
    x_session_id: str | None = Header(default=None),
    combo_service: ComboService = Depends(combo_service_dependency),
):
    """Get available combo patterns.

    Returns:
        List of combo names and their patterns
    """
    return combo_service.get_available_combos()


@router.post("/clear_combos")
def clear_combo_history(
    x_session_id: str | None = Header(default=None),
    combo_service: ComboService = Depends(combo_service_dependency),
):
    """Clear combo history for the current session.

    Args:
        x_session_id: Session identifier

    Returns:
        {"status": "cleared"}
    """
    combo_service.clear_session(x_session_id)
    return {"status": "cleared"}
