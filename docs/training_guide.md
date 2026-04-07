# Local Training Guide

This guide covers the device-local training workflow and artifact publishing contract used by the backend.

## 1. Core Concepts

The training tool is intentionally decoupled from serving, but it publishes artifacts that the backend consumes.

Shared runtime contract:
- Active RF model: `models/hand_alphabet_model.pkl`
- Active RF labels: `models/class_labels.npy`
- Active sequence data: `data/X_data.npy`, `data/y_data.npy`
- Active dynamic labels: `models/wlasl_labels.npy`
- Active registry: `models/shared_backend_state.json`

Feature contract:
- `FEATURE_SCHEMA=histogram` for 8-dimensional histogram features.
- `FEATURE_SCHEMA=mediapipe` for 63-dimensional landmark features.
- Use the same value in both training and inference.

Primary entry point:

```bash
python train_model.py
```

## 2. Install Profiles

Choose dependencies based on your execution target.

### Runtime only

```bash
pip install -r requirements-runtime.txt
```

### Device training (Pi-friendly)

```bash
pip install -r requirements-device.txt
```

### Full training (workstation)

```bash
pip install -r requirements-training.txt
```

### Using pyproject.toml (editable install)

```bash
# Runtime only
pip install -e .

# With device training deps
pip install -e ".[device]"

# With full training deps (TensorFlow)
pip install -e ".[training]"

# Development (includes testing tools)
pip install -e ".[dev]"
```

## 3. Local Training Commands

### Train Random Forest from CSV

```bash
python train_model.py
```

### Using the training module directly

```bash
# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Import and use training components
python -c "from hand_sign_detection.training import rf_trainer; print('Ready')"
```

## 4. Hardware Profiles

Profiles are configured in `src/hand_sign_detection/core/config.py`:

### `pi_zero`

- Random Forest defaults tuned for constrained hardware.
- Lower preprocessing limits by default.
- Best for frequent on-device retraining.

### `full`

- Higher-capacity defaults for workstation hardware.
- Better for broad dataset runs and LSTM-heavy workflows.

## 5. Artifact and Metadata Outputs

Training writes:
- Model files under `models/`
- Run metadata in `models/shared_backend_state.json`

Metadata includes:
- Profile used
- Training metrics
- Preprocessing summary
- File manifest

Because the backend loads from the shared registry, training can remain a separate workflow without changing serving code.

## 6. LSTM Guidance on Low-End Devices

For Raspberry Pi Zero 2 W, full LSTM training is generally not practical for regular use due to runtime and thermal limits.

Recommended strategy:
1. Retrain RF locally on the device.
2. Train LSTM on a stronger machine or cloud (Colab).
3. Deploy optimized artifacts back to the device.

## 7. Environment Variables

All training configuration respects environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `FEATURE_SCHEMA` | Feature extraction mode | `histogram` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

See `.env.example` for the full list.