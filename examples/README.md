# Examples Directory

Sample scripts demonstrating Hand Sign Detection API usage.

## Available Examples

| Script | Description |
|--------|-------------|
| `basic_usage.py` | Simple API calls for image prediction |

## Running Examples

1. Start the API server:
   ```bash
   ./run.sh  # or run.ps1 on Windows
   ```

2. Run an example:
   ```bash
   python examples/basic_usage.py
   ```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check server status |
| `/predict` | POST | Predict from uploaded file |
| `/predict/base64` | POST | Predict from base64 image |
| `/predict/webcam` | POST | Process webcam frame |

See full API documentation at `http://localhost:8000/docs` when server is running.
