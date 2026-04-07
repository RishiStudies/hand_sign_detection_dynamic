# Requirements Directory

This folder contains pip requirements files for different use cases.

## Files

| File | Purpose |
|------|---------|
| `requirements-runtime.txt` | Production API server dependencies |
| `requirements-training.txt` | ML training dependencies (TensorFlow, etc.) |
| `requirements-device.txt` | Edge device deployment (MediaPipe, etc.) |

## Installation

```bash
# For production API
pip install -r requirements/requirements-runtime.txt

# For training
pip install -r requirements/requirements-training.txt

# For edge devices
pip install -r requirements/requirements-device.txt

# Or use pyproject.toml extras (recommended)
pip install -e ".[training]"
pip install -e ".[device]"
pip install -e ".[dev]"
```

## Note

The canonical dependencies are defined in `pyproject.toml`. These requirements files
provide thin wrappers for compatibility with tools that don't support pyproject.toml.
