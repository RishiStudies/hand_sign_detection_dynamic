#!/usr/bin/env python3
"""
Hand Sign Detection - Desktop Launcher
Starts the API server and optionally opens the web UI in browser
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def main():
    print("\n" + "="*60)
    print("🚀 Hand Sign Detection - Desktop Launcher")
    print("="*60 + "\n")
    
    # Get project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check if virtual environment exists
    venv_path = project_root / ".venv"
    if not venv_path.exists():
        print("❌ Virtual environment not found!")
        print(f"   Expected at: {venv_path}")
        print("\n   To create: python -m venv .venv")
        return 1
    
    print("✅ Virtual environment found\n")
    
    # Check if key files exist
    print("📋 Checking project files...")
    required_files = [
        "src/api_server.py",
        "index.html",
        "models/hand_alphabet_model.pkl"
    ]
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   ✓ {file_path}")
        else:
            print(f"   ✗ {file_path} - NOT FOUND")
            if "model" in file_path:
                print(f"     ⚠️  Train models first: python src/training_pipeline.py --command train-rf")
    
    print("\n" + "-"*60)
    print("🔧 Starting API Server...")
    print("-"*60 + "\n")
    
    # Get Python executable from venv
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        python_exe = venv_path / "bin" / "python"
    
    if not python_exe.exists():
        print(f"❌ Python executable not found at {python_exe}")
        return 1
    
    # Start API server
    api_cmd = [
        str(python_exe),
        "-m", "uvicorn",
        "src.api_server:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    
    print(f"Command: {' '.join(api_cmd)}\n")
    
    try:
        # Wait a moment, then optionally open browser
        print("⏱️  Server starting...")
        print("\n✅ API running at: http://localhost:8000")
        print("📖 Docs at: http://localhost:8000/docs")
        print("\n🌐 Opening web UI in browser in 3 seconds...")
        
        # Start server process
        proc = subprocess.Popen(api_cmd)
        
        # Wait for server to start
        time.sleep(3)
        
        # Try to open browser
        try:
            webbrowser.open("http://localhost:8000")
            print("✅ Browser opened\n")
        except:
            print("⚠️  Could not open browser automatically")
            print("   Visit: http://localhost:8000\n")
        
        print("="*60)
        print("🎯 Web UI is ready!")
        print("   • Drag & drop images to predict")
        print("   • Use webcam for live recognition")
        print("   • Press Ctrl+C to stop server")
        print("="*60 + "\n")
        
        # Keep server running
        proc.wait()
        
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
