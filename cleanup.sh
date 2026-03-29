#!/bin/bash
# Hand Sign Detection - Cleanup Script
# Removes unnecessary legacy files and cache directories
# This is safe to run - all functionality is available in the canonical training_module/

echo "🧹 Hand Sign Detection Project Cleanup"
echo "====================================="
echo ""

# Files to delete
FILES=(
    "model_training_orchestrator.py:Wrapper for orchestrator_main - functionality in training_module/cli.py"
    "run_lstm_notebook_cells.py:Legacy notebook runner - superseded by training_module CLI"
    "lstm_model_training.ipynb:Duplicate reference notebook - training workflow now in training_module/"
    "node-v24.14.1-x64.msi:Node.js installer - not needed for Python backend"
)

DIRS=(
    ".pytest_cache:Auto-generated pytest cache (already in .gitignore, rebuilds automatically)"
)

# Count what we're deleting
total_size=0
file_count=0

echo "📋 Files to delete:"
for entry in "${FILES[@]}"; do
    file="${entry%%:*}"
    reason="${entry##*:}"
    if [ -f "$file" ]; then
        size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
        total_size=$((total_size + size))
        file_count=$((file_count + 1))
        size_mb=$((size / 1048576))
        echo "  ✓ $file ($size_mb MB) - $reason"
    else
        echo "  ⊘ $file - Not found"
    fi
done

echo ""
echo "📁 Directories to delete:"
for entry in "${DIRS[@]}"; do
    dir="${entry%%:*}"
    reason="${entry##*:}"
    if [ -d "$dir" ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            size=$(du -sh "$dir" | cut -f1)
        else
            size=$(du -sh "$dir" | cut -f1)
        fi
        file_count=$((file_count + 1))
        echo "  ✓ $dir ($size) - $reason"
    else
        echo "  ⊘ $dir - Not found"
    fi
done

echo ""
echo "📊 Total items to delete: $file_count"
echo ""

# Get user confirmation
read -p "🔒 Confirm deletion? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Cleanup cancelled."
    exit 1
fi

echo ""
echo "🗑️  Deleting files..."

# Delete files
for entry in "${FILES[@]}"; do
    file="${entry%%:*}"
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "  ✓ Deleted: $file"
    fi
done

# Delete directories
for entry in "${DIRS[@]}"; do
    dir="${entry%%:*}"
    if [ -d "$dir" ]; then
        rm -rf "$dir"
        echo "  ✓ Deleted: $dir"
    fi
done

echo ""
echo "✅ Cleanup completed successfully!"
echo ""
echo "📚 Canonical entry points:"
echo "  • Main training CLI: python src/training_pipeline.py"
echo "  • Full orchestrator: python -m src.training_module.cli orchestrator_main"
echo "  • Random Forest: python -m src.training_module.cli random_forest_main"
echo "  • LSTM trainer: python -m src.training_module.cli lstm_main"
