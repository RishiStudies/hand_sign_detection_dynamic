#!/usr/bin/env python3
"""
Complete training script for dynamic hand sign detection.
This script:
1. Cleans up useless files
2. Preprocesses WLASL video data
3. Trains the LSTM model

Uses the new modular architecture in hand_sign_detection package.
"""
import os
import shutil
import sys

# Set project root and add src to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

print("=" * 70)
print("HAND SIGN DETECTION - TRAINING PIPELINE")
print("=" * 70)

# Step 1: Cleanup useless files
print("\n[STEP 1] Cleaning up useless files...")
files_to_delete = [
    "node-v24.14.1-x64.msi",
    "model_training_orchestrator.py",
    "run_lstm_notebook_cells.py",
    "lstm_model_training.ipynb",
    "ANALYSIS_SUMMARY.md",
    "CLEANUP_ANALYSIS.txt",
    "ANALYSIS_FINDINGS.md",
    "VISUAL_SUMMARY.txt",
    "ANALYSIS_REPORT.py",
    "analyze.bat",
    "analyze_project.py",
    "cleanup_files.py",
    "cleanup_script.py",
    "file_size_analysis.py",
    "run_analysis.bat",
    "run_analysis.py",
    "run_cleanup.bat",
    os.path.join("models", "random_forest_20260310_225242.pkl"),
]

dirs_to_delete = [".pytest_cache"]

deleted = 0
for f in files_to_delete:
    path = os.path.join(PROJECT_ROOT, f)
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"  [OK] Deleted: {f}")
            deleted += 1
        except Exception as e:
            print(f"  [ERROR] Error deleting {f}: {e}")

for d in dirs_to_delete:
    path = os.path.join(PROJECT_ROOT, d)
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            print(f"  [OK] Deleted directory: {d}/")
            deleted += 1
        except Exception as e:
            print(f"  [ERROR] Error deleting {d}: {e}")

print(f"  Cleanup complete: {deleted} items deleted")

# Step 2: Import training modules from new architecture
print("\n[STEP 2] Initializing training modules...")
try:
    from hand_sign_detection.core.config import get_settings
    from hand_sign_detection.training import WlaslPreprocessor, LstmTrainer
    
    settings = get_settings()
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    
    # Check for TensorFlow
    try:
        import tensorflow as tf
        TENSORFLOW_AVAILABLE = True
        print(f"  [OK] TensorFlow version: {tf.__version__}")
    except ImportError:
        TENSORFLOW_AVAILABLE = False
    
    print(f"  [OK] Training modules imported")
    print(f"  [OK] TensorFlow available: {TENSORFLOW_AVAILABLE}")
    print(f"  [OK] Data directory: {DATA_DIR}")
    print(f"  [OK] Models directory: {MODELS_DIR}")
except ImportError as e:
    print(f"  [ERROR] Failed to import training module: {e}")
    sys.exit(1)

if not TENSORFLOW_AVAILABLE:
    print("\n  [ERROR] TensorFlow is required for LSTM training!")
    print("  Install with: pip install tensorflow")
    sys.exit(1)

# Step 3: Check existing data
print("\n[STEP 3] Checking existing training data...")
x_path = os.path.join(DATA_DIR, "X_data.npy")
y_path = os.path.join(DATA_DIR, "y_data.npy")
wlasl_json = os.path.join(DATA_DIR, "WLASL_v0.3.json")
video_folder = os.path.join(DATA_DIR, "videos")

print(f"  X_data.npy exists: {os.path.exists(x_path)}")
print(f"  y_data.npy exists: {os.path.exists(y_path)}")
print(f"  WLASL_v0.3.json exists: {os.path.exists(wlasl_json)}")
print(f"  videos folder exists: {os.path.exists(video_folder)}")

if os.path.exists(video_folder):
    video_count = len([f for f in os.listdir(video_folder) if f.endswith('.mp4')])
    print(f"  Video files: {video_count}")

# Step 4: Initialize preprocessor and trainer
print("\n[STEP 4] Creating training components...")
preprocessor = WlaslPreprocessor(
    data_dir=DATA_DIR,
    models_dir=MODELS_DIR,
)
lstm_trainer = LstmTrainer(models_dir=MODELS_DIR)
print("  [OK] Preprocessor and trainer created")

# Step 5: Preprocess videos (create fresh training data)
print("\n[STEP 5] Preprocessing WLASL videos...")
print("  Configuration:")
print("    - Max classes: 20")
print("    - Max videos per class: 10")
print("    - Sequence length: 30 frames")
print("    - Frame stride: 2")
print()

try:
    x_data, y_data, labels = preprocessor.process_videos(
        json_file=wlasl_json,
        video_folder=video_folder,
        max_classes=20,
        max_videos_per_class=10,
        sequence_length=30,
        frame_stride=2,
        save_data=True,
    )
    print(f"\n  [OK] Preprocessing complete!")
    print(f"    X_data shape: {x_data.shape}")
    print(f"    y_data shape: {y_data.shape}")
    print(f"    Total sequences: {len(x_data)}")
    print(f"    Classes: {len(labels)}")
except Exception as e:
    print(f"  [ERROR] Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Train LSTM model
print("\n[STEP 6] Training LSTM model...")
print("  This may take several minutes...")
print()

try:
    metrics = lstm_trainer.train(
        x_data=x_data,
        y_data=y_data,
        labels=labels,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        early_stopping_patience=10,
        save_model=True,
    )
    
    print(f"\n  [OK] LSTM training complete!")
    print(f"    Test accuracy: {metrics.get('test_accuracy', 0):.2%}")
    print(f"    Test loss: {metrics.get('test_loss', 0):.4f}")
    print(f"    Epochs ran: {metrics.get('epochs_ran', 0)}")
except Exception as e:
    print(f"  [ERROR] LSTM training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 7: Verify model
print("\n[STEP 7] Verifying saved model...")
model_path = os.path.join(MODELS_DIR, "gesture_model.h5")
labels_path = os.path.join(MODELS_DIR, "wlasl_labels.npy")

if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"  [OK] Model saved: {model_path} ({size_mb:.2f} MB)")
else:
    print(f"  [ERROR] Model not found: {model_path}")

if os.path.exists(labels_path):
    import numpy as np
    labels = np.load(labels_path, allow_pickle=True)
    print(f"  [OK] Labels saved: {len(labels)} classes")
    print(f"    Classes: {', '.join(labels[:10])}...")
else:
    print(f"  [ERROR] Labels not found: {labels_path}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"\nModel can detect {len(labels) if os.path.exists(labels_path) else 'N/A'} dynamic hand signs:")
if os.path.exists(labels_path):
    for i, label in enumerate(labels):
        print(f"  {i+1}. {label}")
print("\nTo use the model, run:")
print("  uvicorn hand_sign_detection.api.app:app --host 127.0.0.1 --port 8000")
