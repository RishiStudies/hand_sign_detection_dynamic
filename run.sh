#!/bin/bash
# Hand Sign Detection - Quick Start Script (macOS/Linux)
# Starts the FastAPI backend server for the web UI

echo "🚀 Hand Sign Detection - Starting Backend"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated. Activating now..."
    source .venv/bin/activate
fi

echo "📦 Checking dependencies..."

# Check if required packages are installed
for pkg in fastapi uvicorn numpy scikit-learn opencv-python; do
    python -c "import $pkg" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  ✓ $pkg installed"
    else
        echo "  ⚠️  $pkg not found"
    fi
done

echo ""
echo "🔧 Starting FastAPI Server..."
echo ""
echo "The API will be available at: http://localhost:8000"
echo "Swagger docs at: http://localhost:8000/docs"
echo ""
echo "⏱️  Wait for 'Uvicorn running on' message before opening the UI..."
echo ""

# Start the API server (use new module path)
python -m uvicorn hand_sign_detection.api.app:app --host 0.0.0.0 --port 8000 --reload
