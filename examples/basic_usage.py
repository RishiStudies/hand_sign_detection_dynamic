#!/usr/bin/env python3
"""
Example: Basic Hand Sign Detection

Demonstrates how to use the Hand Sign Detection API for image prediction.
"""

import requests
import base64
from pathlib import Path


def predict_from_file(image_path: str, api_url: str = "http://localhost:8000") -> dict:
    """
    Send an image file to the API for prediction.
    
    Args:
        image_path: Path to the image file
        api_url: Base URL of the API server
        
    Returns:
        Prediction result dictionary
    """
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        response = requests.post(f"{api_url}/predict", files=files)
    
    response.raise_for_status()
    return response.json()


def predict_from_base64(image_data: bytes, api_url: str = "http://localhost:8000") -> dict:
    """
    Send base64-encoded image data to the API.
    
    Args:
        image_data: Raw image bytes
        api_url: Base URL of the API server
        
    Returns:
        Prediction result dictionary
    """
    encoded = base64.b64encode(image_data).decode("utf-8")
    response = requests.post(
        f"{api_url}/predict/base64",
        json={"image": encoded}
    )
    
    response.raise_for_status()
    return response.json()


def check_health(api_url: str = "http://localhost:8000") -> dict:
    """Check if the API server is running."""
    response = requests.get(f"{api_url}/health")
    response.raise_for_status()
    return response.json()


def main():
    api_url = "http://localhost:8000"
    
    print("Hand Sign Detection - Example Usage")
    print("=" * 40)
    
    # Check health
    try:
        health = check_health(api_url)
        print(f"✓ API Status: {health.get('status', 'unknown')}")
    except requests.exceptions.ConnectionError:
        print("✗ API not running. Start with: python -m uvicorn hand_sign_detection.api.app:app")
        return
    
    # Example prediction (uncomment and provide an image path)
    # result = predict_from_file("path/to/hand_sign.jpg")
    # print(f"Prediction: {result}")
    
    print("\nTo test predictions:")
    print("  1. Start the server: ./run.sh or ./run.ps1")
    print("  2. Open http://localhost:8000 in your browser")
    print("  3. Use the web UI to upload images or use webcam")
    print("\nOr use this script:")
    print("  result = predict_from_file('your_image.jpg')")


if __name__ == "__main__":
    main()
