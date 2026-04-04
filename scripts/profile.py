#!/usr/bin/env python3
"""
Profiling Script for Hand Sign Detection

Uses cProfile and line_profiler for detailed performance analysis.

Usage:
    python scripts/profile.py --target feature_extraction
    python scripts/profile.py --target inference --output profile_report.txt
    python scripts/profile.py --target api_request --url http://localhost:8000
"""

import argparse
import cProfile
import io
import pstats
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def profile_feature_extraction(iterations: int = 100):
    """Profile feature extraction pipeline."""
    from hand_sign_detection.models.features import (
        extract_histogram_features,
        detect_hand,
        reset_mediapipe,
    )
    
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def run_histogram():
        for _ in range(iterations):
            extract_histogram_features(test_image)
    
    def run_mediapipe():
        for _ in range(iterations):
            detect_hand(test_image)
        reset_mediapipe()
    
    print(f"\n{'='*60}")
    print(f"HISTOGRAM FEATURE EXTRACTION ({iterations} iterations)")
    print(f"{'='*60}")
    
    profiler = cProfile.Profile()
    profiler.enable()
    run_histogram()
    profiler.disable()
    
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(20)
    print(stream.getvalue())
    
    print(f"\n{'='*60}")
    print(f"MEDIAPIPE HAND DETECTION ({iterations} iterations)")
    print(f"{'='*60}")
    
    try:
        profiler = cProfile.Profile()
        profiler.enable()
        run_mediapipe()
        profiler.disable()
        
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
        stats.print_stats(20)
        print(stream.getvalue())
    except Exception as e:
        print(f"MediaPipe profiling skipped: {e}")


def profile_model_inference(iterations: int = 100):
    """Profile model inference."""
    from hand_sign_detection.models.manager import get_model_manager
    
    manager = get_model_manager()
    
    if manager.rf_model is not None:
        test_features = np.random.rand(8).astype(np.float32)
        
        def run_rf():
            for _ in range(iterations):
                manager.rf_model.predict([test_features])
        
        print(f"\n{'='*60}")
        print(f"RANDOM FOREST INFERENCE ({iterations} iterations)")
        print(f"{'='*60}")
        
        profiler = cProfile.Profile()
        profiler.enable()
        run_rf()
        profiler.disable()
        
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
        stats.print_stats(20)
        print(stream.getvalue())
    
    if manager.lstm_model is not None:
        test_sequence = np.random.rand(1, 30, 8).astype(np.float32)
        
        def run_lstm():
            for _ in range(iterations):
                manager.lstm_model.predict(test_sequence, verbose=0)
        
        print(f"\n{'='*60}")
        print(f"LSTM SEQUENCE INFERENCE ({iterations} iterations)")
        print(f"{'='*60}")
        
        profiler = cProfile.Profile()
        profiler.enable()
        run_lstm()
        profiler.disable()
        
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
        stats.print_stats(20)
        print(stream.getvalue())


def profile_api_request(base_url: str, iterations: int = 20):
    """Profile API request handling."""
    try:
        import httpx
        import cv2
    except ImportError:
        print("Required packages not installed: pip install httpx opencv-python")
        return
    
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', test_image)
    img_bytes = img_encoded.tobytes()
    
    client = httpx.Client(base_url=base_url, timeout=30.0)
    
    def run_requests():
        for _ in range(iterations):
            client.post(
                "/predict",
                files={"file": ("test.jpg", img_bytes, "image/jpeg")}
            )
    
    print(f"\n{'='*60}")
    print(f"API REQUEST PROFILING ({iterations} requests to {base_url})")
    print(f"{'='*60}")
    
    profiler = cProfile.Profile()
    profiler.enable()
    run_requests()
    profiler.disable()
    
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(30)
    print(stream.getvalue())
    
    client.close()


def main():
    parser = argparse.ArgumentParser(description="Profile hand sign detection components")
    parser.add_argument("--target", choices=["feature_extraction", "inference", "api_request", "all"],
                       default="all", help="Component to profile")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", help="Output file for profile report")
    
    args = parser.parse_args()
    
    # Redirect output if file specified
    original_stdout = sys.stdout
    if args.output:
        sys.stdout = open(args.output, "w")
    
    try:
        print("Hand Sign Detection - Performance Profiling")
        print("=" * 60)
        
        if args.target in ("feature_extraction", "all"):
            profile_feature_extraction(args.iterations)
        
        if args.target in ("inference", "all"):
            profile_model_inference(args.iterations)
        
        if args.target in ("api_request", "all"):
            profile_api_request(args.url, args.iterations // 5)
        
        print("\n" + "=" * 60)
        print("Profiling Complete!")
        
    finally:
        if args.output:
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"Profile report saved to: {args.output}")


if __name__ == "__main__":
    main()
