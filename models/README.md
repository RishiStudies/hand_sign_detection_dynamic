# Models Directory

This directory contains trained model files for the Hand Sign Detection system.

> **Note:** Model binaries are NOT committed to Git. Train or download them locally.

## Available Models

### 1. Hand Alphabet Model (Static Signs)

Random Forest classifier for A-Z fingerspelling:

```bash
# Train the model
python -m hand_sign_detection.training.pipeline train-alphabet

# Or using the CLI
python train_model.py --model alphabet
```

Output: `hand_alphabet_model.pkl`

### 2. Gesture Model (Dynamic Signs)

Neural network for dynamic gesture recognition:

```bash
# Train the model
python -m hand_sign_detection.training.pipeline train-gesture

# Or using the CLI
python train_model.py --model gesture
```

Output: `gesture_model.h5`

### 3. Class Labels

Label mappings for trained models:

- `class_labels.npy` - Gesture class names
- `wlasl_labels.npy` - WLASL vocabulary

## Pre-trained Models

Download pre-trained models (if available):

```bash
# Using the download script
python scripts/download_models.py

# Or manually from releases
# https://github.com/YOUR_ORG/hand_sign_detection/releases
```

## Directory Structure

```
models/
├── README.md              # This file
├── hand_alphabet_model.pkl  # Static sign classifier (gitignored)
├── gesture_model.h5       # Dynamic gesture model (gitignored)
├── class_labels.npy       # Class mappings (gitignored)
└── wlasl_labels.npy       # WLASL vocabulary (gitignored)
```

## Model Performance

| Model | Accuracy | Size | Inference Time |
|-------|----------|------|----------------|
| Alphabet (RF) | ~95% | ~5 MB | <10ms |
| Gesture (CNN) | ~85% | ~50 MB | ~50ms |

## Training Requirements

- **Alphabet model:** ~5 minutes on CPU
- **Gesture model:** ~30 minutes on GPU (or 2+ hours on CPU)

Ensure data files exist in `data/` before training. See `data/README.md`.

## API Usage

Models are loaded automatically by the API server:

```python
from hand_sign_detection.models import get_predictor

# Get the appropriate predictor
predictor = get_predictor("alphabet")
result = predictor.predict(frame)
```

## Troubleshooting

**Model not found?**
1. Train the model: `python train_model.py`
2. Check file permissions
3. Verify `MODELS_DIR` environment variable

**Out of memory during training?**
- Reduce batch size in config
- Use CPU training for smaller models
- Enable gradient checkpointing for large models
