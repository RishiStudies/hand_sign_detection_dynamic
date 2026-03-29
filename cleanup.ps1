# Hand Sign Detection - Cleanup Script
# Removes unnecessary legacy files and cache directories
# This is safe to run - all functionality is available in the canonical training_module/

Write-Host "🧹 Hand Sign Detection Project Cleanup" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Files to delete with reasons
$filesToDelete = @(
    @{
        Path = "model_training_orchestrator.py"
        Reason = "Wrapper for orchestrator_main - functionality in training_module/cli.py"
    },
    @{
        Path = "run_lstm_notebook_cells.py"
        Reason = "Legacy notebook runner - superseded by training_module CLI"
    },
    @{
        Path = "lstm_model_training.ipynb"
        Reason = "Duplicate reference notebook - training workflow now in training_module/"
    },
    @{
        Path = "node-v24.14.1-x64.msi"
        Reason = "Node.js installer - not needed for Python backend"
    }
)

$directoriesToDelete = @(
    @{
        Path = ".pytest_cache"
        Reason = "Auto-generated pytest cache (already in .gitignore, rebuilds automatically)"
    }
)

# Count what we're deleting
$totalSize = 0
$fileCount = 0

Write-Host "📋 Files to delete:" -ForegroundColor Yellow
foreach ($file in $filesToDelete) {
    $fullPath = $file.Path
    if (Test-Path $fullPath) {
        $item = Get-Item $fullPath
        $size = if ($item.PSIsContainer) { (Get-ChildItem -Path $fullPath -Recurse | Measure-Object -Property Length -Sum).Sum } else { $item.Length }
        $totalSize += $size
        $fileCount++
        $sizeStr = if ($size -lt 1MB) { "$([math]::Round($size/1KB, 2)) KB" } else { "$([math]::Round($size/1MB, 2)) MB" }
        Write-Host "  ✓ $($file.Path) ($sizeStr) - $($file.Reason)" -ForegroundColor Green
    } else {
        Write-Host "  ⊘ $($file.Path) - Not found" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "📁 Directories to delete:" -ForegroundColor Yellow
foreach ($dir in $directoriesToDelete) {
    $fullPath = $dir.Path
    if (Test-Path $fullPath) {
        $item = Get-Item $fullPath
        $size = if ($item.PSIsContainer) { (Get-ChildItem -Path $fullPath -Recurse | Measure-Object -Property Length -Sum).Sum } else { $item.Length }
        $totalSize += $size
        $fileCount++
        $sizeStr = if ($size -lt 1MB) { "$([math]::Round($size/1KB, 2)) KB" } else { "$([math]::Round($size/1MB, 2)) MB" }
        Write-Host "  ✓ $($dir.Path) ($sizeStr) - $($dir.Reason)" -ForegroundColor Green
    } else {
        Write-Host "  ⊘ $($dir.Path) - Not found" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "📊 Total space to reclaim: $fileCount items, ~$([math]::Round($totalSize/1MB, 2)) MB" -ForegroundColor Cyan
Write-Host ""

# Get user confirmation
$confirm = Read-Host "🔒 Confirm deletion? (yes/no)"
if ($confirm -ne "yes") {
    Write-Host "❌ Cleanup cancelled." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🗑️  Deleting files..." -ForegroundColor Yellow

# Delete files
foreach ($file in $filesToDelete) {
    $fullPath = $file.Path
    if (Test-Path $fullPath) {
        try {
            Remove-Item -Path $fullPath -Force -ErrorAction Stop
            Write-Host "  ✓ Deleted: $fullPath" -ForegroundColor Green
        } catch {
            Write-Host "  ✗ Failed to delete $fullPath : $_" -ForegroundColor Red
        }
    }
}

# Delete directories
foreach ($dir in $directoriesToDelete) {
    $fullPath = $dir.Path
    if (Test-Path $fullPath) {
        try {
            Remove-Item -Path $fullPath -Recurse -Force -ErrorAction Stop
            Write-Host "  ✓ Deleted: $fullPath" -ForegroundColor Green
        } catch {
            Write-Host "  ✗ Failed to delete $fullPath : $_" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "✅ Cleanup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📚 Canonical entry points:" -ForegroundColor Cyan
Write-Host "  • Main training CLI: python src/training_pipeline.py" -ForegroundColor White
Write-Host "  • Full orchestrator: python -m src.training_module.cli orchestrator_main" -ForegroundColor White
Write-Host "  • Random Forest: python -m src.training_module.cli random_forest_main" -ForegroundColor White
Write-Host "  • LSTM trainer: python -m src.training_module.cli lstm_main" -ForegroundColor White
