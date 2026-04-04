"""RandomForest model trainer.

Handles training RandomForest models from various data sources.
"""

import json
import logging
import os
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from hand_sign_detection.core.config import get_settings
from hand_sign_detection.core.shared_state import update_shared_state
from hand_sign_detection.models.features import extract_features_from_bytes

logger = logging.getLogger("hand_sign_detection.training.rf_trainer")


class RandomForestTrainer:
    """Trainer for RandomForest gesture classification models.

    Supports training from:
    - CSV files with precomputed features
    - Image sample manifests

    Usage:
        trainer = RandomForestTrainer()
        metrics = trainer.train_from_csv("path/to/data.csv")
    """

    def __init__(self):
        self._settings = get_settings()
        self.model: RandomForestClassifier | None = None
        self.labels: np.ndarray | None = None
        self.last_metrics: dict = {}

    def train_from_csv(
        self,
        data_path: str | None = None,
        save_model: bool = True,
        low_end: bool = False,
        profile_name: str = "full",
        source: str = "rf_trainer_csv",
    ) -> dict:
        """Train RandomForest from CSV data.

        Args:
            data_path: Path to CSV file (uses default if None)
            save_model: Whether to save model to disk
            low_end: Use low-end device configuration
            profile_name: Device profile name
            source: Source identifier for tracking

        Returns:
            Training metrics dictionary
        """
        settings = self._settings

        if data_path is None:
            data_path = os.path.join(settings.data_dir, "hand_alphabet_data.csv")

        logger.info(
            "Starting RandomForest training from CSV: %s (low_end=%s, profile=%s)",
            data_path,
            low_end,
            profile_name,
        )

        if not os.path.exists(data_path):
            logger.error("CSV data file not found: %s", data_path)
            raise FileNotFoundError(f"CSV data file not found: {data_path}")

        data = pd.read_csv(data_path, header=None, on_bad_lines="skip")
        data = data.dropna().reset_index(drop=True)
        logger.debug("CSV loaded: shape=%s", data.shape)

        x_values = data.iloc[:, :-1].values
        y_values = data.iloc[:, -1].values
        logger.info("Data extracted: X shape=%s, y shape=%s", x_values.shape, y_values.shape)

        return self._train(
            x_values=x_values,
            y_values=y_values,
            save_model=save_model,
            low_end=low_end,
            profile_name=profile_name,
            source=source,
        )

    def train_from_manifest(
        self,
        manifest_path: str,
        save_model: bool = True,
        profile_name: str = "full",
        source: str = "rf_trainer_manifest",
    ) -> dict:
        """Train RandomForest from image sample manifest.

        Args:
            manifest_path: Path to manifest JSON file
            save_model: Whether to save model to disk
            profile_name: Device profile name
            source: Source identifier for tracking

        Returns:
            Training metrics dictionary
        """
        logger.info("Training from manifest: %s", manifest_path)

        with open(manifest_path, encoding="utf-8") as file_obj:
            manifest = json.load(file_obj)

        samples = manifest.get("samples", [])
        if not samples:
            raise ValueError("No samples found in training manifest")

        x_values = []
        y_values = []
        label_to_idx: dict[str, int] = {}

        for sample in samples:
            label = str(sample["label"])
            if label not in label_to_idx:
                label_to_idx[label] = len(label_to_idx)

            with open(sample["file_path"], "rb") as image_file:
                features = extract_features_from_bytes(image_file.read())

            x_values.append(features.flatten())
            y_values.append(label_to_idx[label])

        metrics = self._train(
            x_values=np.array(x_values),
            y_values=np.array(y_values),
            class_labels=np.array(list(label_to_idx.keys())),
            save_model=save_model,
            profile_name=profile_name,
            source=source,
        )

        # Cleanup manifest directory
        shutil.rmtree(os.path.dirname(manifest_path), ignore_errors=True)

        return metrics

    def _train(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        save_model: bool = True,
        low_end: bool = False,
        profile_name: str = "full",
        source: str = "rf_trainer",
        class_labels: np.ndarray | None = None,
    ) -> dict:
        """Internal training implementation.

        Args:
            x_values: Feature matrix
            y_values: Label vector
            save_model: Whether to save model
            low_end: Use low-end configuration
            profile_name: Device profile name
            source: Source identifier
            class_labels: Optional class label names

        Returns:
            Training metrics
        """
        settings = self._settings

        logger.info(
            "_train: X shape=%s, y shape=%s, save_model=%s",
            x_values.shape,
            y_values.shape,
            save_model,
        )

        # Get hyperparameters from profile
        profile = settings.device_profiles.get(profile_name)
        if profile is None:
            logger.error("Unknown profile: %s", profile_name)
            raise ValueError(f"Unknown profile: {profile_name}")

        if low_end and profile_name == "full":
            n_estimators = 100
            max_depth = 18
            n_jobs = 1
            logger.info("Using low_end configuration for RandomForest")
        else:
            n_estimators = int(profile["rf_estimators"])
            max_depth = int(profile["rf_max_depth"])
            n_jobs = int(profile["rf_n_jobs"])

        logger.debug(
            "RandomForest hyperparams: n_estimators=%d, max_depth=%d, n_jobs=%d",
            n_estimators,
            max_depth,
            n_jobs,
        )

        # Split data
        stratify = y_values if len(np.unique(y_values)) > 1 else None
        logger.info(
            "Splitting data: unique_classes=%d, stratify=%s",
            len(np.unique(y_values)),
            stratify is not None,
        )

        x_train, x_test, y_train, y_test = train_test_split(
            x_values,
            y_values,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
        logger.debug("Split completed: train_shape=%s, test_shape=%s", x_train.shape, x_test.shape)

        # Train model
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=n_jobs,
        )
        logger.info("Starting RandomForest model training...")
        clf.fit(x_train, y_train)
        logger.info("RandomForest model training completed")

        # Evaluate
        train_acc = clf.score(x_train, y_train)
        test_acc = clf.score(x_test, y_test)
        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred)

        logger.info(
            "RandomForest scores: train_acc=%.4f, test_acc=%.4f, accuracy=%.4f",
            train_acc,
            test_acc,
            acc,
        )

        # Determine labels to save
        labels_to_save = clf.classes_
        if class_labels is not None:
            labels_to_save = class_labels[clf.classes_.astype(int)]

        # Save model
        if save_model:
            model_path = os.path.join(settings.models_dir, "hand_alphabet_model.pkl")
            labels_path = os.path.join(settings.models_dir, "class_labels.npy")

            os.makedirs(settings.models_dir, exist_ok=True)

            logger.info("Saving model to %s", model_path)
            joblib.dump(clf, model_path)

            logger.info("Saving labels to %s", labels_path)
            np.save(labels_path, labels_to_save)

            update_shared_state(
                "random_forest",
                {
                    "model_path": model_path,
                    "labels_path": labels_path,
                    "profile": profile_name,
                    "source": source,
                    "feature_schema": settings.feature_schema,
                    "feature_schema_version": settings.feature_schema_version,
                    "feature_dimension": int(x_values.shape[1]),
                },
                publisher="RandomForestTrainer",
            )
            logger.info("Shared state updated with RandomForest model metadata")

        metrics = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "accuracy_score": float(acc),
            "samples": int(len(x_values)),
            "classes": int(len(labels_to_save)),
            "profile": profile_name,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }

        self.model = clf
        self.labels = labels_to_save
        self.last_metrics = metrics

        logger.info(
            "RandomForest training completed: %d classes, %d samples",
            len(labels_to_save),
            metrics["samples"],
        )

        return metrics

    def evaluate(
        self,
        data_path: str | None = None,
        model_path: str | None = None,
    ) -> float:
        """Evaluate model accuracy on a dataset.

        Args:
            data_path: Path to evaluation CSV
            model_path: Path to model file

        Returns:
            Accuracy score
        """
        settings = self._settings

        if data_path is None:
            data_path = os.path.join(settings.data_dir, "hand_alphabet_data.csv")
        if model_path is None:
            model_path = os.path.join(settings.models_dir, "hand_alphabet_model.pkl")

        logger.info("Evaluating RandomForest: data=%s, model=%s", data_path, model_path)

        data = pd.read_csv(data_path, header=None, on_bad_lines="skip").dropna()
        x_values = data.iloc[:, :-1].values
        y_values = data.iloc[:, -1].values

        clf = joblib.load(model_path)
        y_pred = clf.predict(x_values)
        acc = accuracy_score(y_values, y_pred)

        logger.info("RandomForest evaluation completed: accuracy=%.4f", acc)
        return acc


def train_rf_from_csv(**kwargs) -> dict:
    """Convenience function for training from CSV."""
    trainer = RandomForestTrainer()
    return trainer.train_from_csv(**kwargs)


def train_rf_from_manifest(**kwargs) -> dict:
    """Convenience function for training from manifest."""
    trainer = RandomForestTrainer()
    return trainer.train_from_manifest(**kwargs)
