import json
import os
from datetime import datetime


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
SHARED_STATE_PATH = os.path.join(MODELS_DIR, "shared_backend_state.json")


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
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_relative(path: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")
    return path.replace("\\", "/")


def load_shared_state() -> dict:
    if not os.path.exists(SHARED_STATE_PATH):
        return dict(DEFAULT_SHARED_STATE)

    with open(SHARED_STATE_PATH, "r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    return _deep_merge(DEFAULT_SHARED_STATE, data)


def save_shared_state(state: dict, publisher: str = None) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    merged = _deep_merge(DEFAULT_SHARED_STATE, state)
    merged["last_updated"] = datetime.now().isoformat()
    if publisher:
        merged["publisher"] = publisher

    with open(SHARED_STATE_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(merged, file_obj, indent=2)
    return SHARED_STATE_PATH


def update_shared_state(section: str, values: dict, publisher: str = None) -> str:
    current = load_shared_state()
    normalized = {}
    for key, value in values.items():
        if key.endswith("_path") and value:
            normalized[key] = _to_relative(value)
        else:
            normalized[key] = value

    existing_section = current.get(section, {})
    if not isinstance(existing_section, dict):
        existing_section = {}
    current[section] = _deep_merge(existing_section, normalized)
    return save_shared_state(current, publisher=publisher)


def resolve_shared_path(section: str, key: str) -> str:
    state = load_shared_state()
    value = state.get(section, {}).get(key)
    if not value:
        value = DEFAULT_SHARED_STATE.get(section, {}).get(key)
    if not value:
        raise KeyError(f"Shared artifact path not found for {section}.{key}")
    return os.path.join(PROJECT_ROOT, value.replace("/", os.sep))