"""Shared artifacts state management.

Manages the shared_backend_state.json file that tracks model artifacts
and their locations across components.
"""

import json
import os
from datetime import datetime
from typing import Any

from hand_sign_detection.core.config import get_settings

DEFAULT_SHARED_STATE = {
    "random_forest": {
        "model_path": "models/hand_alphabet_model.pkl",
        "labels_path": "models/class_labels.npy",
    },
    "dynamic_data": {
        "x_path": "data/X_data.npy",
        "y_path": "data/y_data.npy",
        "labels_path": "models/wlasl_labels.npy",
    },
    "lstm": {
        "model_path": "models/gesture_model.h5",
        "labels_path": "models/wlasl_labels.npy",
    },
    "last_updated": None,
    "publisher": None,
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_relative(path: str, project_root: str) -> str:
    """Convert absolute path to relative."""
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.relpath(path, project_root).replace("\\", "/")
    return path.replace("\\", "/")


def load_shared_state() -> dict[str, Any]:
    """Load shared state from disk.

    Returns:
        Merged state with defaults
    """
    settings = get_settings()
    state_path = settings.shared_state_path

    if not os.path.exists(state_path):
        return dict(DEFAULT_SHARED_STATE)

    try:
        with open(state_path, encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        return _deep_merge(DEFAULT_SHARED_STATE, data)
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_SHARED_STATE)


def save_shared_state(state: dict, publisher: str | None = None) -> str:
    """Save shared state to disk.

    Args:
        state: State dictionary to save
        publisher: Name of the component saving state

    Returns:
        Path to saved state file
    """
    settings = get_settings()
    state_path = settings.shared_state_path

    os.makedirs(settings.models_dir, exist_ok=True)

    merged = _deep_merge(DEFAULT_SHARED_STATE, state)
    merged["last_updated"] = datetime.now().isoformat()
    if publisher:
        merged["publisher"] = publisher

    with open(state_path, "w", encoding="utf-8") as file_obj:
        json.dump(merged, file_obj, indent=2)

    return state_path


def update_shared_state(
    section: str,
    values: dict[str, Any],
    publisher: str | None = None,
) -> str:
    """Update a section of shared state.

    Args:
        section: Section name (e.g., "random_forest", "lstm")
        values: Values to update in the section
        publisher: Name of the component making the update

    Returns:
        Path to saved state file
    """
    settings = get_settings()
    current = load_shared_state()

    # Normalize paths to relative
    normalized = {}
    for key, value in values.items():
        if key.endswith("_path") and value:
            normalized[key] = _to_relative(value, settings.project_root)
        else:
            normalized[key] = value

    existing_section = current.get(section, {})
    if not isinstance(existing_section, dict):
        existing_section = {}

    current[section] = _deep_merge(existing_section, normalized)
    return save_shared_state(current, publisher=publisher)


def resolve_shared_path(section: str, key: str) -> str:
    """Resolve a shared artifact path to absolute path.

    Args:
        section: Section name
        key: Key within section (e.g., "model_path")

    Returns:
        Absolute path to the artifact

    Raises:
        KeyError: If path not found in state
    """
    settings = get_settings()
    state = load_shared_state()

    value = state.get(section, {}).get(key)
    if not value:
        value = DEFAULT_SHARED_STATE.get(section, {}).get(key)
    if not value:
        raise KeyError(f"Shared artifact path not found for {section}.{key}")

    return os.path.join(settings.project_root, value.replace("/", os.sep))


def get_model_paths(model_type: str) -> dict[str, str]:
    """Get all paths for a model type.

    Args:
        model_type: "random_forest" or "lstm"

    Returns:
        Dictionary with resolved paths
    """
    state = load_shared_state()
    section = state.get(model_type, {})
    settings = get_settings()

    paths = {}
    for key, value in section.items():
        if key.endswith("_path") and value:
            paths[key] = os.path.join(settings.project_root, value.replace("/", os.sep))

    return paths
