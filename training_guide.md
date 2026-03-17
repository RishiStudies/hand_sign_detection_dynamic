# Hand Sign Detection Local Training Guide

This project now includes a device-local training software path focused on low-end hardware.

The local trainer is intentionally separate from the serving backend, but it publishes shared artifacts for the backend to consume.

Shared backend contract:
- Active RF model: `models/hand_alphabet_model.pkl`
- Active RF labels: `models/class_labels.npy`
- Active sequence data: `data/X_data.npy`, `data/y_data.npy`
- Active dynamic labels: `models/wlasl_labels.npy`
- Active registry: `models/shared_backend_state.json`

Feature contract:
- Set `FEATURE_SCHEMA=histogram` for 8-value grayscale histogram features.
- Set `FEATURE_SCHEMA=mediapipe` for 63-value hand landmark features.
- Use the same `FEATURE_SCHEMA` value for both training and inference. The backend now validates model/input compatibility explicitly.

Primary entry point:

```bash
python src/training_pipeline.py
```

## 1. Install Profiles

Use profile-specific dependencies based on where the training runs.

### Runtime only (inference server)

```bash
pip install -r requirements-runtime.txt
```

### Device local training (Pi-friendly)

```bash
pip install -r requirements-device.txt
```

### Full training workstation (includes TensorFlow)

```bash
pip install -r requirements-training.txt
```

## 2. Device-Local Commands

The local trainer supports command mode and hardware profiles.

### Preprocess WLASL data for local workflows

```bash
python src/training_pipeline.py --command preprocess --profile pi_zero
```

### Train Random Forest locally on CSV

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

### Extract all training data to one ZIP

```bash
python src/training_pipeline.py --command export-data --profile full --archive-prefix training_data_full
```

Optional flags:
- `--exclude-videos` to skip `data/videos`.
- `--include-hashes` to include SHA256 checksums in the manifest.
- `--output-dir reports/exports` to choose export destination.

### Run full device local workflow

```bash
python src/training_pipeline.py --command device-all --profile pi_zero --note "nightly device run"
```

## 3. Profile Behavior

### `pi_zero`
- RF defaults tuned for constrained hardware.
- WLASL preprocessing uses reduced data limits by default.
- Designed for reliable on-device retraining and packaging.

### `full`
- Higher-capacity defaults for workstation-class training.
- Larger dataset processing limits.

## 4. Useful Overrides

You can override profile defaults when needed:

```bash
python src/training_pipeline.py --command preprocess --profile pi_zero --max-classes 12 --max-videos-per-class 4 --sequence-length 24 --frame-stride 2
```

Additional path overrides:
- `--json-file`
- `--video-folder`
- `--csv-path`

## 5. Artifacts and Metadata

Local packaging writes:
- Packaged copies under `models/packages/`
- Run metadata JSON under `models/packages/` and `reports/`
- Shared active registry under `models/shared_backend_state.json`

These metadata files include:
- Device profile used
- Last training metrics
- Preprocessing summary
- File list included in package

The backend reads active model and dataset paths from the shared registry, so the local trainer can stay a separate tool while still feeding the same backend runtime.

## 6. Notes About LSTM on Device

For Raspberry Pi Zero 2 W, full LSTM training is not recommended for regular operation due to runtime and thermal constraints.

Recommended strategy:
1. Do RF retraining locally on the device.
2. Run full LSTM training on a stronger machine.
3. Deploy optimized artifacts back to the device.

## 7. Backward Compatibility

Legacy mode still works:

```bash
python src/training_pipeline.py --model all
python src/training_pipeline.py --model random_forest --low-end
```