# Hand Sign Detection - Quick Start Script
# Starts the FastAPI backend server for the web UI

Write-Host "🚀 Hand Sign Detection - Starting Backend" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment is activated
if ($null -eq $env:VIRTUAL_ENV) {
    Write-Host "⚠️  Virtual environment not activated. Activating now..." -ForegroundColor Yellow
    & .venv\Scripts\Activate.ps1
}

Write-Host "📦 Checking dependencies..." -ForegroundColor Yellow

# Check if required packages are installed
$requiredPackages = @('fastapi', 'uvicorn', 'numpy', 'scikit-learn', 'opencv-python')
foreach ($pkg in $requiredPackages) {
    $installed = python -c "import $pkg" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ $pkg installed" -ForegroundColor Green
    } else {
        Write-Host "  ⚠️  $pkg not found" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "🔧 Starting FastAPI Server..." -ForegroundColor Green
Write-Host ""
Write-Host "The API will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Swagger docs at: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "⏱️  Wait for 'Uvicorn running on' message before opening the UI..." -ForegroundColor Yellow
Write-Host ""

# Start the API server
python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
