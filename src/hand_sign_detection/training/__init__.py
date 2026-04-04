"""Training module for model development.

Contains:
- WlaslPreprocessor: Video preprocessing for WLASL dataset
- RandomForestTrainer: RF model training
- LstmTrainer: LSTM model training
- ArtifactManager: Model artifact packaging
- GestureClusterer: Unsupervised K-Means clustering
- FeatureAutoencoder: Unsupervised feature learning
- UnsupervisedTrainer: High-level unsupervised training
- Job queue utilities for background training
"""

from hand_sign_detection.training.artifact_manager import ArtifactManager
from hand_sign_detection.training.job_queue import (
    JOB_QUEUE_NAME,
    _get_redis_connection,
    enqueue_named_job,
    get_job_status,
    is_job_queue_available,
)
from hand_sign_detection.training.lstm_trainer import LstmTrainer
from hand_sign_detection.training.preprocessor import WlaslPreprocessor
from hand_sign_detection.training.rf_trainer import RandomForestTrainer
from hand_sign_detection.training.unsupervised import (
    AutoencoderResult,
    ClusteringResult,
    FeatureAutoencoder,
    GestureClusterer,
    UnsupervisedTrainer,
)

__all__ = [
    "WlaslPreprocessor",
    "RandomForestTrainer",
    "LstmTrainer",
    "ArtifactManager",
    "GestureClusterer",
    "FeatureAutoencoder",
    "UnsupervisedTrainer",
    "ClusteringResult",
    "AutoencoderResult",
    "JOB_QUEUE_NAME",
    "_get_redis_connection",
    "enqueue_named_job",
    "get_job_status",
    "is_job_queue_available",
]
