# Deployment Guide

This guide covers deploying the Hand Sign Detection API Server to production.

## Quick Start

### Prerequisites
- Python 3.11+
- Redis (for distributed deployments; optional for single-server)
- Docker & Docker Compose (for containerized deployment)

### Local Installation

1. **Clone and setup environment:**
   ```bash
   cd hand_sign_detection_dynamic
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   # For runtime (inference + API server):
   pip install -r requirements-runtime.txt
   
   # OR for development/training:
   pip install -r requirements-training.txt
   ```

3. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   nano .env
   ```

4. **Start the API server:**
   ```bash
   PYTHONPATH=src uvicorn hand_sign_detection.api.app:app \
     --host 0.0.0.0 \
     --port 8000 \
     --workers 4
   ```

### Docker Deployment

1. **Build the image:**
   ```bash
   docker build -f Dockerfile.backend -t hsd-api:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     -p 8000:8000 \
     -e LOG_LEVEL=INFO \
     -e TRAINING_API_KEY='your-secure-key-here' \
     -e FRONTEND_URL='https://your-frontend-domain.com' \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/logs:/app/logs \
     --name hsd-api \
     hsd-api:latest
   ```

3. **Or use Docker Compose:**
   ```bash
   docker-compose up -d
   ```

## Configuration

### Required Environment Variables

| Variable | Description | Example |
|----------|---|---|
| `TRAINING_API_KEY` | API key for training endpoints (CHANGE IN PRODUCTION) | `secret-key-min-32-chars` |
| `FRONTEND_URL` | Frontend application URL (for CORS and redirects) | `https://app.example.com` |

### Recommended Environment Variables

| Variable | Description | Default |
|----------|---|---|
| `LOG_LEVEL` | Logging verbosity | `INFO` |
| `LOG_TO_FILE` | Enable file logging | `false` |
| `CORS_ORIGINS` | Allowed CORS origins (comma-separated) | `localhost:3000` |
| `REDIS_URL` | Redis connection URL | empty (in-memory fallback) |
| `MAX_UPLOAD_BYTES` | Max single frame upload size | `2097152` (2MB) |
| `MAX_CSV_UPLOAD_BYTES` | Max CSV file size | `10485760` (10MB) |

## Production Deployment Checklist

Before deploying to production, complete the items in [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md).

## Monitoring

### Health Endpoints

- `GET /health/live` - Liveness check (always returns 200)
- `GET /health/ready` - Readiness check (models loaded?)
- `GET /health/details` - Detailed system status

### Logs

Logs are written to:
- stderr (always)
- `logs/api_server.log` (if `LOG_TO_FILE=true`)

Log files rotate at 50MB with 5 backups retained.

### Metrics

Monitor:
- Response time: Target <200ms for `/predict`
- Error rate: Target <0.1%
- LSTM availability: Should be available if TensorFlow installed
- Job queue: Check Redis connection and job backlog

## Troubleshooting

### API won't start
1. Check `FRONTEND_URL` is set
2. Verify `TRAINING_API_KEY` length (min 16 chars recommended)
3. Check logs: `tail -f logs/api_server.log`

### Training requests fail
1. Verify Redis is running: `redis-cli ping`
2. Check permission on `data/job_inputs/` directory
3. Verify `TRAINING_API_KEY` in request matches server

### Inference returns 501
- LSTM model unavailable (TensorFlow not installed or model file missing)
- This is normal if LSTM not needed; use `/predict` instead

### Models won't load
1. Verify `models/shared_backend_state.json` exists
2. Check file paths in `shared_backend_state.json` are valid
3. Ensure model file formats are correct (.pkl for RF, .h5 for LSTM)

## Production Performance Tuning

### API Server
```bash
# Increase workers for CPU-bound inference
python -m uvicorn src.api_server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 8  # Match CPU cores
```

### Rate Limiting
Adjust environment variables:
```bash
MAX_PREDICT_REQUESTS_PER_WINDOW=300  # Higher for high-traffic
MAX_TRAIN_REQUESTS_PER_WINDOW=10     # Lower for safety
```

### Redis
For distributed deployments, configure Redis with:
```bash
# In redis.conf or docker run:
--maxmemory 512mb
--maxmemory-policy allkeys-lru  # Evict LRU keys when full
```

## Backup & Recovery

### Model Backup
```bash
# Backup trained models
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/

# Restore
tar -xzf models-backup-YYYYMMDD.tar.gz
```

### Data Backup
```bash
# Backup training data
tar -czf data-backup-$(date +%Y%m%d).tar.gz data/*.npy data/*.csv

# Note: Videos are large; archive separately if needed
tar -czf data-videos-backup-$(date +%Y%m%d).tar.gz data/videos/
```

## Scaling

### Horizontal Scaling (Multiple API Servers)
1. Deploy multiple API server containers behind a load balancer
2. Configure Redis for distributed rate limiting and sessions
3. Share model files via NFS or S3
4. Use separate Redis instance for job queue (Redis RQ)

### Example Docker Compose (3 API servers + Redis):
```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  api-1:
    build: .
    ports:
      - "8001:8000"
    environment:
      REDIS_URL: redis://redis:6379
    depends_on:
      - redis
  
  api-2:
    build: .
    ports:
      - "8002:8000"
    environment:
      REDIS_URL: redis://redis:6379
    depends_on:
      - redis
  
  api-3:
    build: .
    ports:
      - "8003:8000"
    environment:
      REDIS_URL: redis://redis:6379
    depends_on:
      - redis
```

## Rollback

If new deployment is failing:

1. **Check logs:**
   ```bash
   docker logs hsd-api
   ```

2. **Switch to previous version:**
   ```bash
   docker stop hsd-api
   docker run -d ... hsd-api:previous-tag
   ```

3. **Restore models if needed:**
   ```bash
   tar -xzf models-backup.tar.gz
   docker restart hsd-api
   ```

## Support

For issues:
1. Check [PRODUCTION_CHECKLIST.md](PRODUCTION_CHECKLIST.md)
2. Review logs
3. Test with `/health/details` endpoint
4. Consult the main [README.md](README.md)
