"""Background job handlers for training tasks.

Provides job handlers for use with RQ (Redis Queue).
"""

import logging

from hand_sign_detection.training.lstm_trainer import LstmTrainer
from hand_sign_detection.training.preprocessor import WlaslPreprocessor
from hand_sign_detection.training.rf_trainer import RandomForestTrainer

logger = logging.getLogger("hand_sign_detection.training.jobs")


def handle_train_rf_samples(manifest_path: str) -> dict:
    """Handle RF training from image samples job.

    Args:
        manifest_path: Path to training manifest JSON

    Returns:
        Training metrics
    """
    logger.info("Job: train_rf_samples started with manifest=%s", manifest_path)

    trainer = RandomForestTrainer()
    metrics = trainer.train_from_manifest(manifest_path=manifest_path)

    return {
        "job_name": "train_rf_samples",
        "success": True,
        "metrics": metrics,
    }


def handle_train_rf_csv(csv_path: str) -> dict:
    """Handle RF training from CSV job.

    Args:
        csv_path: Path to training CSV file

    Returns:
        Training metrics
    """
    logger.info("Job: train_rf_csv started with csv_path=%s", csv_path)

    trainer = RandomForestTrainer()
    metrics = trainer.train_from_csv(data_path=csv_path)

    return {
        "job_name": "train_rf_csv",
        "success": True,
        "metrics": metrics,
    }


def handle_process_wlasl() -> dict:
    """Handle WLASL video preprocessing job.

    Returns:
        Preprocessing summary
    """
    logger.info("Job: process_wlasl started")

    preprocessor = WlaslPreprocessor()
    x_data, y_data = preprocessor.process_videos()

    return {
        "job_name": "process_wlasl",
        "success": True,
        "summary": preprocessor.last_summary,
        "x_shape": list(x_data.shape),
        "y_shape": list(y_data.shape),
    }


def handle_train_lstm() -> dict:
    """Handle LSTM training job.

    Returns:
        Training metrics
    """
    logger.info("Job: train_lstm started")

    trainer = LstmTrainer()
    trainer.train()

    return {
        "job_name": "train_lstm",
        "success": True,
        "metrics": trainer.last_metrics,
    }


# Job handler mapping for job queue
JOB_HANDLERS = {
    "train_rf_samples": handle_train_rf_samples,
    "train_rf_csv": handle_train_rf_csv,
    "process_wlasl": handle_process_wlasl,
    "train_lstm": handle_train_lstm,
}
