#!/usr/bin/env python3
"""
Performance Benchmarking Script for Hand Sign Detection

Benchmarks:
- Feature extraction (histogram vs mediapipe)
- Model inference (Random Forest vs LSTM)
- API endpoint latency
- Batch processing throughput

Usage:
    python scripts/benchmark.py --all
    python scripts/benchmark.py --feature-extraction
    python scripts/benchmark.py --inference
    python scripts/benchmark.py --api --url http://localhost:8000
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",  # Simple format for benchmark output
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class BenchmarkResult:
    """Benchmark result container."""
    name: str
    iterations: int
    total_time: float
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    throughput: float
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_s": round(self.total_time, 4),
            "mean_time_ms": round(self.mean_time * 1000, 3),
            "std_time_ms": round(self.std_time * 1000, 3),
            "min_time_ms": round(self.min_time * 1000, 3),
            "max_time_ms": round(self.max_time * 1000, 3),
            "throughput_per_sec": round(self.throughput, 2),
            **self.extra
        }

    def __str__(self) -> str:
        return (
            f"\n{self.name}\n"
            f"{'=' * len(self.name)}\n"
            f"  Iterations: {self.iterations}\n"
            f"  Total time: {self.total_time:.3f}s\n"
            f"  Mean: {self.mean_time * 1000:.3f}ms\n"
            f"  Std:  {self.std_time * 1000:.3f}ms\n"
            f"  Min:  {self.min_time * 1000:.3f}ms\n"
            f"  Max:  {self.max_time * 1000:.3f}ms\n"
            f"  Throughput: {self.throughput:.2f}/sec\n"
        )


def run_benchmark(func, iterations: int = 100, warmup: int = 5, name: str = "Benchmark") -> BenchmarkResult:
    """Run a benchmark function multiple times and collect statistics."""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times = np.array(times)
    total = np.sum(times)

    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_time=total,
        mean_time=np.mean(times),
        std_time=np.std(times),
        min_time=np.min(times),
        max_time=np.max(times),
        throughput=iterations / total
    )


def benchmark_feature_extraction(iterations: int = 100) -> list[BenchmarkResult]:
    """Benchmark feature extraction methods."""
    from hand_sign_detection.models.features import (
        detect_hand,
        extract_histogram_features,
        reset_mediapipe,
    )

    results = []

    # Create test image (640x480 RGB)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Benchmark histogram extraction
    def histogram_benchmark():
        extract_histogram_features(test_image)

    results.append(run_benchmark(
        histogram_benchmark,
        iterations=iterations,
        name="Histogram Feature Extraction"
    ))

    # Benchmark MediaPipe detection
    try:
        def mediapipe_benchmark():
            detect_hand(test_image)

        results.append(run_benchmark(
            mediapipe_benchmark,
            iterations=iterations,
            name="MediaPipe Hand Detection"
        ))
    except (ImportError, RuntimeError, OSError) as e:
        logger.warning("MediaPipe benchmark skipped: %s", e)
    finally:
        reset_mediapipe()

    return results


def benchmark_model_inference(iterations: int = 100) -> list[BenchmarkResult]:
    """Benchmark model inference."""
    from hand_sign_detection.models.manager import get_model_manager

    results = []
    manager = get_model_manager()

    # Check if models are loaded
    if manager.rf_model is None:
        logger.info("No Random Forest model found - skipping RF benchmark")
    else:
        # Create test features matching the model's expected input
        feature_dim = 8  # histogram features
        test_features = np.random.rand(feature_dim).astype(np.float32)

        def rf_benchmark():
            manager.rf_model.predict([test_features])

        results.append(run_benchmark(
            rf_benchmark,
            iterations=iterations,
            name="Random Forest Inference"
        ))

    if manager.lstm_model is None:
        logger.info("No LSTM model found - skipping LSTM benchmark")
    else:
        # LSTM expects sequence of features
        sequence_length = 30
        feature_dim = 8
        test_sequence = np.random.rand(1, sequence_length, feature_dim).astype(np.float32)

        def lstm_benchmark():
            manager.lstm_model.predict(test_sequence, verbose=0)

        results.append(run_benchmark(
            lstm_benchmark,
            iterations=iterations,
            name="LSTM Sequence Inference"
        ))

    return results


def benchmark_api_endpoints(base_url: str, iterations: int = 50) -> list[BenchmarkResult]:
    """Benchmark API endpoint latency."""
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed - skipping API benchmark")
        logger.info("Install with: pip install httpx")
        return []

    results = []
    client = httpx.Client(base_url=base_url, timeout=30.0)

    # Health check endpoint
    def health_benchmark():
        client.get("/health/live")

    try:
        results.append(run_benchmark(
            health_benchmark,
            iterations=iterations,
            name="Health Check Endpoint"
        ))
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logger.warning("Health endpoint benchmark failed: %s", e)

    # Predict endpoint with test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    try:
        import cv2
        _, img_encoded = cv2.imencode('.jpg', test_image)
        img_bytes = img_encoded.tobytes()

        def predict_benchmark():
            client.post(
                "/predict",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")}
            )

        results.append(run_benchmark(
            predict_benchmark,
            iterations=iterations,
            name="Predict Endpoint"
        ))
    except (OSError, ValueError) as e:
        logger.warning("Predict endpoint benchmark failed: %s", e)

    client.close()
    return results


def benchmark_batch_processing(batch_sizes: list[int] = None) -> list[BenchmarkResult]:
    """Benchmark batch processing at different sizes."""
    from hand_sign_detection.models.features import extract_histogram_features

    if batch_sizes is None:
        batch_sizes = [1, 10, 50, 100]

    results = []

    for batch_size in batch_sizes:
        # Create batch of test images
        batch_test_images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(batch_size)
        ]

        def batch_benchmark(images=batch_test_images):
            for img in images:
                extract_histogram_features(img)

        result = run_benchmark(
            batch_benchmark,
            iterations=max(10, 100 // batch_size),
            name=f"Batch Processing (size={batch_size})"
        )
        result.extra["batch_size"] = batch_size
        result.extra["images_per_sec"] = round(batch_size * result.throughput, 2)
        results.append(result)

    return results


def save_results(results: list[BenchmarkResult], output_path: str):
    """Save benchmark results to JSON file."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [r.to_dict() for r in results]
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("\nResults saved to: %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Benchmark hand sign detection system")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--feature-extraction", action="store_true", help="Benchmark feature extraction")
    parser.add_argument("--inference", action="store_true", help="Benchmark model inference")
    parser.add_argument("--api", action="store_true", help="Benchmark API endpoints")
    parser.add_argument("--batch", action="store_true", help="Benchmark batch processing")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")

    args = parser.parse_args()

    if not any([args.all, args.feature_extraction, args.inference, args.api, args.batch]):
        args.all = True

    all_results = []

    logger.info("=" * 60)
    logger.info("Hand Sign Detection - Performance Benchmark")
    logger.info("=" * 60)

    if args.all or args.feature_extraction:
        logger.info("\n[1/4] Feature Extraction Benchmarks...")
        results = benchmark_feature_extraction(args.iterations)
        for r in results:
            logger.info(r)
        all_results.extend(results)

    if args.all or args.inference:
        logger.info("\n[2/4] Model Inference Benchmarks...")
        results = benchmark_model_inference(args.iterations)
        for r in results:
            logger.info(r)
        all_results.extend(results)

    if args.all or args.batch:
        logger.info("\n[3/4] Batch Processing Benchmarks...")
        results = benchmark_batch_processing()
        for r in results:
            logger.info(r)
        all_results.extend(results)

    if args.all or args.api:
        logger.info("\n[4/4] API Endpoint Benchmarks...")
        results = benchmark_api_endpoints(args.url, args.iterations // 2)
        for r in results:
            logger.info(r)
        all_results.extend(results)

    if all_results:
        save_results(all_results, args.output)

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
