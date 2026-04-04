# 🚀 Quick Start Guide

Get up and running with the Hand Sign Detection system in minutes.

## Prerequisites

- Python 3.11+
- Virtual environment (recommended)
- Git (for cloning)

## 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/yourorg/hand-sign-detection.git
cd hand-sign-detection

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements-runtime.txt
```

## 2. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env with your settings (optional for local dev)
```

Key settings:
- `FEATURE_SCHEMA=histogram` - Feature extraction mode
- `LOG_LEVEL=INFO` - Logging verbosity
- `TRAINING_API_KEY=` - Leave empty for local dev

## 3. Start the API Server

### Option A: Using launch script (Recommended)

```bash
python launch.py
```

This auto-opens your browser to the UI.

### Option B: Direct uvicorn

```bash
# Windows
$env:PYTHONPATH="src"
uvicorn hand_sign_detection.api.app:app --host 127.0.0.1 --port 8000 --reload

# macOS/Linux
PYTHONPATH=src uvicorn hand_sign_detection.api.app:app --host 127.0.0.1 --port 8000 --reload
```

### Option C: Using startup scripts

```bash
# Windows
.\run.ps1

# macOS/Linux
./run.sh
```

## 4. Access the Application

| URL | Purpose |
|-----|---------|
| http://localhost:8000 | Web UI |
| http://localhost:8000/docs | API documentation (Swagger) |
| http://localhost:8000/health/ready | Health check |

## 5. Test a Prediction

### Via Web UI
1. Open http://localhost:8000
2. Upload an image or use webcam
3. See prediction with confidence score

### Via API
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@hand_gesture.jpg"
```

Response:
```json
{
  "prediction": "A",
  "confidence": 0.92,
  "inference_time_ms": 15.3
}
```

## Docker Quick Start

```bash
# Backend + Redis
docker compose up --build

# Full stack (with frontend)
docker compose --profile full up --build
```

## Web UI Features

- **Image Upload**: Drag-and-drop or click to select
- **Webcam Capture**: Real-time detection from camera
- **Prediction Display**: Label + confidence score
- **Processing Info**: Inference time metrics

## Troubleshooting

### "Cannot reach API"
- Ensure server is running: `curl http://localhost:8000/health/live`
- Check port 8000 is not in use

### "Models not found"
- Train models first: `python train_model.py`
- Or download pre-trained models

### "Import errors"
- Ensure PYTHONPATH includes `src/`:
  ```bash
  export PYTHONPATH=$PYTHONPATH:$(pwd)/src
  ```

### Webcam not working
- Check browser camera permissions
- Use HTTPS for production (required for camera access)

## Next Steps

- **Train models**: See [training_guide.md](training_guide.md)
- **Deploy to production**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Pre-deploy checklist**: See [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md)
- **Architecture overview**: See [architecture_and_workflows.md](architecture_and_workflows.md)
