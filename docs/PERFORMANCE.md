# Performance Optimization Guide

This guide covers profiling, benchmarking, and optimization techniques for the Hand Sign Detection system.

## Quick Start

```bash
# Run all benchmarks
python scripts/benchmark.py --all

# Run specific benchmark
python scripts/benchmark.py --feature-extraction
python scripts/benchmark.py --inference
python scripts/benchmark.py --api --url http://localhost:8000

# Profile specific components
python scripts/profile.py --target feature_extraction
python scripts/profile.py --target inference --output report.txt
```

## Benchmarking

### Feature Extraction

| Method | Dimensions | Typical Latency | Use Case |
|--------|------------|-----------------|----------|
| Histogram | 8 | ~0.5ms | Fast inference, embedded |
| MediaPipe | 63 | ~15-30ms | Accurate detection |

### Model Inference

| Model | Input Shape | Typical Latency | Accuracy |
|-------|-------------|-----------------|----------|
| Random Forest | (8,) or (63,) | ~0.1ms | Good |
| LSTM | (30, features) | ~5-10ms | Better for sequences |

## Performance Tuning

### 1. Feature Schema Selection

```bash
# Fast mode (histogram features)
FEATURE_SCHEMA=histogram

# Accurate mode (MediaPipe landmarks)
FEATURE_SCHEMA=mediapipe
```

**Trade-off**: Histogram is ~30x faster but captures less spatial information.

### 2. Device-Specific Optimization

```bash
# Raspberry Pi / embedded
DEVICE_PROFILE=raspberry_pi
FEATURE_SCHEMA=histogram
MAX_PREDICT_REQUESTS_PER_WINDOW=60

# Desktop with GPU
DEVICE_PROFILE=default
FEATURE_SCHEMA=mediapipe
MAX_PREDICT_REQUESTS_PER_WINDOW=180
```

### 3. Batch Processing

For video processing, use batch endpoints:

```python
# Single frame (high latency per frame)
POST /predict

# Sequence (optimized for video)
POST /predict_sequence
```

### 4. Caching Strategy

The prediction service uses LRU caching for repeated frames:

```python
# In services/prediction.py
CACHE_SIZE = 128  # Adjust based on memory
```

### 5. Model Optimization

#### Random Forest
- Reduce `n_estimators` for faster inference (trade-off: accuracy)
- Use `n_jobs=-1` during training for parallel processing

#### LSTM
- Enable TensorFlow Lite conversion for 2-3x speedup
- Use `model.predict(..., verbose=0)` to disable logging overhead

```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

### 6. API Optimization

#### Rate Limiting
Adjust limits based on server capacity:

```bash
MAX_PREDICT_REQUESTS_PER_WINDOW=180  # Predictions per minute
MAX_SEQUENCE_REQUESTS_PER_WINDOW=30  # Sequences per minute
MAX_CONCURRENT_SEQUENCE_REQUESTS=2   # Parallel LSTM calls
```

#### Connection Pooling
For high-throughput scenarios:

```python
import httpx

# Use connection pooling
with httpx.Client(limits=httpx.Limits(max_connections=20)) as client:
    responses = [client.post("/predict", ...) for _ in range(100)]
```

### 7. Docker Optimization

```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
```

## Profiling Deep Dive

### CPU Profiling

```bash
# Detailed function-level profile
python scripts/profile.py --target feature_extraction --output cpu_profile.txt
```

### Memory Profiling

```bash
pip install memory_profiler

# Profile memory usage
python -m memory_profiler scripts/benchmark.py --feature-extraction
```

### GPU Profiling (TensorFlow)

```python
import tensorflow as tf

# Enable profiler
tf.profiler.experimental.start('logdir')
# ... run inference ...
tf.profiler.experimental.stop()
```

## Monitoring in Production

### Metrics to Track

1. **Latency Percentiles** (p50, p95, p99)
2. **Throughput** (requests/second)
3. **Error Rate** (4xx, 5xx responses)
4. **Memory Usage** (especially for LSTM)
5. **Cache Hit Rate** (prediction caching)

### Health Endpoints

```bash
# Liveness (is server running?)
curl http://localhost:8000/health/live

# Readiness (is server ready for traffic?)
curl http://localhost:8000/health/ready
```

### Prometheus Metrics (Optional)

Add to requirements:
```
prometheus-fastapi-instrumentator
```

```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

## Benchmark Results Template

```json
{
  "timestamp": "2024-01-15 10:30:00",
  "environment": {
    "python": "3.11.5",
    "cpu": "Intel i7-10700",
    "memory": "16GB",
    "os": "Ubuntu 22.04"
  },
  "results": [
    {
      "name": "Histogram Feature Extraction",
      "mean_time_ms": 0.45,
      "throughput_per_sec": 2222
    },
    {
      "name": "Random Forest Inference",
      "mean_time_ms": 0.08,
      "throughput_per_sec": 12500
    }
  ]
}
```

## Common Bottlenecks

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| High CPU on predict | MediaPipe processing | Switch to histogram |
| Memory growing | Model not released | Check singleton lifecycle |
| Slow first request | Cold start | Add warmup in lifespan |
| Timeouts on sequence | LSTM inference | Reduce sequence length |
| Rate limit errors | Too many requests | Increase limits or add caching |

## Best Practices

1. **Profile before optimizing** - Measure, don't guess
2. **Start with feature schema** - Biggest impact on latency
3. **Use appropriate model** - RF for speed, LSTM for accuracy
4. **Enable caching** - For repeated/similar frames
5. **Set realistic limits** - Match rate limits to hardware
6. **Monitor in production** - Track latency percentiles
