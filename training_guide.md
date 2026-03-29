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
python src/training_pipeline.py
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

## 3. Local Training Commands

### Preprocess WLASL data

```bash
python src/training_pipeline.py --command preprocess --profile pi_zero
```

### Train Random Forest from CSV

```bash
python src/training_pipeline.py --command train-rf --profile pi_zero
```

### Evaluate active RF model

```bash
python src/training_pipeline.py --command evaluate --profile pi_zero
```

### Package artifacts and metadata

```bash
python src/training_pipeline.py --command package --profile pi_zero --note "local retrain"
```

### Export training data bundle

```bash
python src/training_pipeline.py --command export-data --profile full --archive-prefix training_data_full
```

Optional export flags:
- `--exclude-videos`
- `--include-hashes`
- `--output-dir reports/exports`

### Run end-to-end local workflow

```bash
python src/training_pipeline.py --command device-all --profile pi_zero --note "nightly device run"
```

## 4. Hardware Profiles

### `pi_zero`

- Random Forest defaults tuned for constrained hardware.
- Lower preprocessing limits by default.
- Best for frequent on-device retraining.

### `full`

- Higher-capacity defaults for workstation hardware.
- Better for broad dataset runs and LSTM-heavy workflows.

## 5. Useful Overrides

Example override set:

```bash
python src/training_pipeline.py --command preprocess --profile pi_zero \
	--max-classes 12 --max-videos-per-class 4 --sequence-length 24 --frame-stride 2
```

Additional path overrides:
- `--json-file`
- `--video-folder`
- `--csv-path`

## 6. Artifact and Metadata Outputs

Packaging writes:
- Packaged artifacts under `models/packages/`
- Run metadata under `models/packages/` and `reports/`
- Updated active registry at `models/shared_backend_state.json`

Metadata includes:
- Profile used
- Training metrics
- Preprocessing summary
- Packaged file manifest

Because the backend loads from the shared registry, training can remain a separate workflow without changing serving code.

## 7. LSTM Guidance on Low-End Devices

For Raspberry Pi Zero 2 W, full LSTM training is generally not practical for regular use due to runtime and thermal limits.

Recommended strategy:
1. Retrain RF locally on the device.
2. Train LSTM on a stronger machine.
3. Deploy optimized artifacts back to the device.

## 8. Legacy Compatibility

Legacy command mode remains available:

```bash
python src/training_pipeline.py --model all
python src/training_pipeline.py --model random_forest --low-end
```