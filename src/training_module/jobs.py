import os
import shutil
from typing import Any, Callable, Dict


def _get_training_service_class():
    # Lazy import prevents TensorFlow-heavy module loading during API startup.
    from .service import TrainingService

    return TrainingService


def process_wlasl_job() -> Dict[str, Any]:
    trainer = _get_training_service_class()()
    x_values, _ = trainer.process_wlasl_videos()
    return {
        "job_name": "process_wlasl",
        "status": "success",
        "sequences": int(x_values.shape[0]) if hasattr(x_values, "shape") else 0,
        "summary": trainer.last_preprocess_summary,
    }


def train_lstm_job() -> Dict[str, Any]:
    trainer = _get_training_service_class()()
    trainer.train_lstm(source="job_train_lstm")
    return {
        "job_name": "train_lstm",
        "status": "success",
        **trainer.last_metrics.get("lstm", {}),
    }


def train_rf_samples_job(manifest_path: str) -> Dict[str, Any]:
    try:
        trainer = _get_training_service_class()()
        metrics = trainer.train_random_forest_from_manifest(
            manifest_path=manifest_path,
            source="job_train_rf_samples",
        )
        return {
            "job_name": "train_rf_samples",
            "status": "success",
            **metrics,
        }
    finally:
        _cleanup_path(os.path.dirname(manifest_path))


def train_rf_csv_job(csv_path: str) -> Dict[str, Any]:
    try:
        trainer = _get_training_service_class()()
        metrics = trainer.train_random_forest_from_csv(
            data_path=csv_path,
            source="job_train_rf_csv",
        )
        return {
            "job_name": "train_rf_csv",
            "status": "success",
            **metrics,
        }
    finally:
        _cleanup_path(csv_path)


JOB_HANDLERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "process_wlasl": process_wlasl_job,
    "train_lstm": train_lstm_job,
    "train_rf_samples": train_rf_samples_job,
    "train_rf_csv": train_rf_csv_job,
}


def _cleanup_path(path: str) -> None:
    if not path:
        return
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)