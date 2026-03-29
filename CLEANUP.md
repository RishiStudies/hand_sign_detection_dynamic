# Cleanup Instructions

## Files to Delete

This project contains unnecessary legacy files that have been superseded by the canonical `training_module/` package. The following files can be safely deleted:

### 1. **model_training_orchestrator.py** (13 lines)
- Simple wrapper importing `orchestrator_main` from canonical `training_module/cli.py`
- No longer needed - use canonical entry point instead

### 2. **run_lstm_notebook_cells.py** (120+ lines)
- Legacy notebook cell runner script
- Functionality superseded by training_module CLI commands
- Not referenced anywhere in active codebase

### 3. **lstm_model_training.ipynb** (14000+ lines)
- Duplicate/legacy reference notebook
- Training workflow now standardized in `training_module/`
- Can be safely archived or deleted

### 4. **node-v24.14.1-x64.msi**
- Node.js installer file
- Not needed for Python-only backend
- Can be safely deleted

### 5. **.pytest_cache/** (directory)
- Auto-generated pytest cache
- Already excluded in .gitignore
- Rebuilds automatically on next `pytest` run

---

## How to Delete

### Option 1: Automated Cleanup (Windows)
```powershell
cd path\to\hand_sign_detection_dynamic
.\cleanup.ps1
```

### Option 2: Automated Cleanup (macOS/Linux)
```bash
cd path/to/hand_sign_detection_dynamic
chmod +x cleanup.sh
./cleanup.sh
```

### Option 3: Manual Deletion (Windows)
```powershell
Remove-Item model_training_orchestrator.py -Force
Remove-Item run_lstm_notebook_cells.py -Force
Remove-Item lstm_model_training.ipynb -Force
Remove-Item node-v24.14.1-x64.msi -Force
Remove-Item .pytest_cache -Recurse -Force
```

### Option 4: Manual Deletion (macOS/Linux)
```bash
rm model_training_orchestrator.py
rm run_lstm_notebook_cells.py
rm lstm_model_training.ipynb
rm node-v24.14.1-x64.msi
rm -rf .pytest_cache
```

---

## Impact Analysis

**Functionality Impact:** ✅ ZERO
- All training functionality available through canonical entry points
- No active code imports these deleted files
- All models, data, and tests remain unaffected

**Canonical Entry Points (use these instead):**
```bash
# Main training CLI
python src/training_pipeline.py --help

# Full device pipeline
python src/training_pipeline.py --command device-all --profile pi_zero

# Random Forest only
python src/training_pipeline.py --command train-rf --profile full

# LSTM preprocessing & training
python src/training_pipeline.py --command preprocess --profile pi_zero
python src/training_pipeline.py --command train-lstm --profile full

# Programmatic usage
python -m src.training_module.cli orchestrator_main --help
```

---

## Space Reclaimed

- **model_training_orchestrator.py**: ~1 KB
- **run_lstm_notebook_cells.py**: ~4 KB  
- **lstm_model_training.ipynb**: ~14 MB
- **node-v24.14.1-x64.msi**: ~200+ MB
- **.pytest_cache/**: Variable (typically 2-10 MB)

**Total: ~200+ MB of disk space freed**

---

## Files Kept Intentionally

The following files are kept because they provide important project context:

- ✅ **model_training_legacy_backup.py** - Historical reference (monolithic trainer before refactoring)
- ✅ **training_guide.md** - User documentation for training commands
- ✅ **architecture_and_workflows.md** - System architecture documentation
- ✅ **DEPLOYMENT.md** - Production deployment guide
- ✅ **PRODUCTION_CHECKLIST.md** - Pre-deployment checklist
- ✅ All **src/** code - Active production code
- ✅ All **tests/** - Active test suite
- ✅ All data in **data/**, **models/** - Training datasets and artifacts

---

## Verification After Cleanup

After deletion, verify the cleanup with:

```powershell
# Windows
ls -Name | grep -E "model_training_orchestrator|run_lstm|lstm_model_training|node-v24|\.pytest_cache"
# Should return nothing

# macOS/Linux  
ls | grep -E "model_training_orchestrator|run_lstm|lstm_model_training|node-v24|\.pytest_cache"
# Should return nothing
```

If the verification returns nothing (empty), all files were successfully deleted. ✅
