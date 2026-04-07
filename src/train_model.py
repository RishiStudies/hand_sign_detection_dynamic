#!/usr/bin/env python3
"""
Complete training script for dynamic hand sign detection.

This script:
1. Cleans up useless files
2. Preprocesses WLASL video data
3. Trains the LSTM model

Uses the new modular architecture in hand_sign_detection package.
"""
import logging
import os
import shutil
import sys
from typing import NoReturn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Set project root and add src to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


def cleanup_legacy_files() -> int:
    """Remove legacy and temporary files.

    Returns:
        Number of items deleted.
    """
    logger.info("Cleaning up legacy files...")

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
                logger.debug("Deleted: %s", f)
                deleted += 1
            except OSError as e:
                logger.warning("Failed to delete %s: %s", f, e)

    for d in dirs_to_delete:
        path = os.path.join(PROJECT_ROOT, d)
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                logger.debug("Deleted directory: %s/", d)
                deleted += 1
            except OSError as e:
                logger.warning("Failed to delete directory %s: %s", d, e)

    logger.info("Cleanup complete: %d items deleted", deleted)
    return deleted


def check_tensorflow() -> bool:
    """Check if TensorFlow is available.

    Returns:
        True if TensorFlow is available, False otherwise.
    """
    try:
        import tensorflow as tf
        logger.info("TensorFlow version: %s", tf.__version__)
        return True
    except ImportError:
        logger.error("TensorFlow not available. Install with: pip install tensorflow")
        return False


def check_training_data(data_dir: str) -> dict[str, bool]:
    """Check existence of training data files.

    Args:
        data_dir: Path to data directory.

    Returns:
        Dictionary of file paths and their existence status.
    """
    paths = {
        "X_data.npy": os.path.join(data_dir, "X_data.npy"),
        "y_data.npy": os.path.join(data_dir, "y_data.npy"),
        "WLASL_v0.3.json": os.path.join(data_dir, "WLASL_v0.3.json"),
        "videos": os.path.join(data_dir, "videos"),
    }

    status = {}
    for name, path in paths.items():
        exists = os.path.exists(path)
        status[name] = exists
        logger.info("%s exists: %s", name, exists)

        # Count videos if folder exists
        if name == "videos" and exists:
            video_count = len([f for f in os.listdir(path) if f.endswith('.mp4')])
            logger.info("Video files: %d", video_count)

    return status


def preprocess_data(preprocessor, data_dir: str) -> tuple:
    """Preprocess video data for training.

    Args:
        preprocessor: WlaslPreprocessor instance.
        data_dir: Path to data directory.

    Returns:
        Tuple of (x_data, y_data, labels).

    Raises:
        RuntimeError: If preprocessing fails.
    """
    wlasl_json = os.path.join(data_dir, "WLASL_v0.3.json")
    video_folder = os.path.join(data_dir, "videos")

    logger.info("Preprocessing WLASL videos...")
    logger.info("  Max classes: 20")
    logger.info("  Max videos per class: 10")
    logger.info("  Sequence length: 30 frames")
    logger.info("  Frame stride: 2")

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
        logger.info("Preprocessing complete!")
        logger.info("  X_data shape: %s", x_data.shape)
        logger.info("  y_data shape: %s", y_data.shape)
        logger.info("  Total sequences: %d", len(x_data))
        logger.info("  Classes: %d", len(labels))
        return x_data, y_data, labels
    except FileNotFoundError as e:
        logger.error("Data files not found: %s", e)
        raise RuntimeError(f"Preprocessing failed: {e}") from e
    except ValueError as e:
        logger.error("Invalid data format: %s", e)
        raise RuntimeError(f"Preprocessing failed: {e}") from e


def train_model(trainer, x_data, y_data, labels) -> dict:
    """Train the LSTM model.

    Args:
        trainer: LstmTrainer instance.
        x_data: Training features.
        y_data: Training labels.
        labels: Class label names.

    Returns:
        Training metrics dictionary.

    Raises:
        RuntimeError: If training fails.
    """
    logger.info("Training LSTM model (this may take several minutes)...")

    try:
        metrics = trainer.train(
            x_data=x_data,
            y_data=y_data,
            labels=labels,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            early_stopping_patience=10,
            save_model=True,
        )

        logger.info("LSTM training complete!")
        logger.info("  Test accuracy: %.2f%%", metrics.get('test_accuracy', 0) * 100)
        logger.info("  Test loss: %.4f", metrics.get('test_loss', 0))
        logger.info("  Epochs ran: %d", metrics.get('epochs_ran', 0))
        return metrics
    except ValueError as e:
        logger.error("Invalid training data: %s", e)
        raise RuntimeError(f"Training failed: {e}") from e


def verify_model(models_dir: str) -> bool:
    """Verify saved model files exist.

    Args:
        models_dir: Path to models directory.

    Returns:
        True if all model files exist, False otherwise.
    """
    import numpy as np

    logger.info("Verifying saved model...")
    model_path = os.path.join(models_dir, "gesture_model.h5")
    labels_path = os.path.join(models_dir, "wlasl_labels.npy")

    success = True

    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        logger.info("Model saved: %s (%.2f MB)", model_path, size_mb)
    else:
        logger.error("Model not found: %s", model_path)
        success = False

    if os.path.exists(labels_path):
        labels = np.load(labels_path, allow_pickle=True)
        logger.info("Labels saved: %d classes", len(labels))
        logger.info("  Classes: %s...", ", ".join(labels[:10]))
    else:
        logger.error("Labels not found: %s", labels_path)
        success = False

    return success


def main() -> NoReturn:
    """Main training pipeline entry point."""
    logger.info("=" * 70)
    logger.info("HAND SIGN DETECTION - TRAINING PIPELINE")
    logger.info("=" * 70)

    # Step 1: Cleanup
    cleanup_legacy_files()

    # Step 2: Check TensorFlow
    logger.info("\nInitializing training modules...")
    if not check_tensorflow():
        sys.exit(1)

    try:
        from hand_sign_detection.training import LstmTrainer, WlaslPreprocessor

        DATA_DIR = os.path.join(PROJECT_ROOT, "data")
        MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

        logger.info("Training modules imported successfully")
        logger.info("  Data directory: %s", DATA_DIR)
        logger.info("  Models directory: %s", MODELS_DIR)
    except ImportError as e:
        logger.error("Failed to import training module: %s", e)
        sys.exit(1)

    # Step 3: Check data
    logger.info("\nChecking existing training data...")
    check_training_data(DATA_DIR)

    # Step 4: Initialize components
    logger.info("\nCreating training components...")
    preprocessor = WlaslPreprocessor(
        data_dir=DATA_DIR,
        models_dir=MODELS_DIR,
    )
    lstm_trainer = LstmTrainer(models_dir=MODELS_DIR)
    logger.info("Preprocessor and trainer created")

    # Step 5: Preprocess data
    try:
        x_data, y_data, labels = preprocess_data(preprocessor, DATA_DIR)
    except RuntimeError:
        sys.exit(1)

    # Step 6: Train model
    try:
        train_model(lstm_trainer, x_data, y_data, labels)
    except RuntimeError:
        sys.exit(1)

    # Step 7: Verify model
    if not verify_model(MODELS_DIR):
        logger.warning("Some model files may be missing")

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info("\nTo use the model, run:")
    logger.info("  uvicorn hand_sign_detection.api.app:app --host 127.0.0.1 --port 8000")
    sys.exit(0)


if __name__ == "__main__":
    main()
