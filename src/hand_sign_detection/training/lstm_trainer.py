"""LSTM model trainer.

Handles training LSTM models for gesture sequence recognition.
"""

import logging
import os

import numpy as np
from sklearn.model_selection import train_test_split

from hand_sign_detection.core.config import get_settings
from hand_sign_detection.core.shared_state import update_shared_state

logger = logging.getLogger("hand_sign_detection.training.lstm_trainer")


class LstmTrainer:
    """Trainer for LSTM gesture sequence classification models.

    Requires TensorFlow to be installed.

    Usage:
        trainer = LstmTrainer()
        model = trainer.train()  # Uses preprocessed data
    """

    def __init__(self):
        self._settings = get_settings()
        self.model = None
        self.last_metrics: dict = {}

        # Check TensorFlow availability
        if not self._settings.tensorflow_available:
            logger.warning("TensorFlow not available - LSTM training disabled")

    def train(
        self,
        x_values: np.ndarray | None = None,
        y_values: np.ndarray | None = None,
        save_model: bool = True,
        low_end: bool = False,
        source: str = "lstm_trainer",
    ):
        """Train LSTM model.

        Args:
            x_values: Sequence data (samples, sequence_length, features)
            y_values: Labels
            save_model: Whether to save model to disk
            low_end: Use low-end device configuration
            source: Source identifier for tracking

        Returns:
            Trained Keras model

        Raises:
            RuntimeError: If TensorFlow not available
            FileNotFoundError: If preprocessed data not found
        """
        settings = self._settings

        if not settings.tensorflow_available:
            logger.error("TensorFlow not available - cannot train LSTM")
            raise RuntimeError("TensorFlow not available. Cannot train LSTM model.")

        # Import TensorFlow components
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.utils import to_categorical

        logger.info(
            "Starting LSTM training: x_values=%s, y_values=%s, save_model=%s, low_end=%s",
            x_values is not None,
            y_values is not None,
            save_model,
            low_end,
        )

        # Load preprocessed data if not provided
        if x_values is None or y_values is None:
            x_path = os.path.join(settings.data_dir, "X_data.npy")
            y_path = os.path.join(settings.data_dir, "y_data.npy")
            logger.info("Loading preprocessed data: X from %s, y from %s", x_path, y_path)

            if not os.path.exists(x_path) or not os.path.exists(y_path):
                logger.error(
                    "Preprocessed data not found. X exists=%s, y exists=%s",
                    os.path.exists(x_path),
                    os.path.exists(y_path),
                )
                raise FileNotFoundError("Processed data not found. Run preprocessing first.")

            x_values = np.load(x_path)
            y_values = np.load(y_path)
            logger.info("Data loaded: X shape=%s, y shape=%s", x_values.shape, y_values.shape)

        # Encode labels
        _, y_encoded = np.unique(y_values, return_inverse=True)
        y_encoded = y_encoded.astype(np.int32)
        num_classes = int(np.max(y_encoded)) + 1
        y_cat = to_categorical(y_encoded, num_classes)

        min_samples_per_class = min(np.bincount(y_encoded))
        use_stratify = min_samples_per_class >= 2

        logger.info(
            "Data encoding: num_classes=%d, min_samples_per_class=%d, use_stratify=%s",
            num_classes,
            min_samples_per_class,
            use_stratify,
        )

        # Split data
        if use_stratify:
            x_train, x_test, y_train, y_test = train_test_split(
                x_values, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
            )
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x_values, y_cat, test_size=0.2, random_state=42
            )

        logger.debug("Train/test split: X_train=%s, X_test=%s", x_train.shape, x_test.shape)

        # Configure model architecture
        if low_end:
            lstm_units = [64, 32, 16]
            dense_units = 32
            epochs = 30
            logger.info("Using low_end configuration for LSTM")
        else:
            lstm_units = [128, 64, 32]
            dense_units = 64
            epochs = 50

        logger.info(
            "Building LSTM model: lstm_units=%s, dense_units=%d, epochs=%d",
            lstm_units,
            dense_units,
            epochs,
        )

        # Build model
        model = Sequential(
            [
                LSTM(
                    lstm_units[0],
                    return_sequences=True,
                    input_shape=(x_values.shape[1], x_values.shape[2]),
                ),
                Dropout(0.3),
                LSTM(lstm_units[1], return_sequences=True),
                Dropout(0.3),
                LSTM(lstm_units[2]),
                Dropout(0.3),
                Dense(dense_units, activation="relu"),
                Dropout(0.2),
                Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        logger.info("LSTM model compiled successfully")

        # Train model
        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            restore_best_weights=True,
        )

        logger.info("Starting LSTM model.fit()...")

        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=16,
            validation_data=(x_test, y_test),
            callbacks=[early_stopping],
            verbose=1,
        )

        epochs_ran = len(history.history.get("loss", []))
        logger.info("LSTM training completed: epochs_ran=%d", epochs_ran)

        # Evaluate
        logger.info("Evaluating LSTM on test set...")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        logger.info("LSTM evaluation: test_loss=%.4f, test_acc=%.4f", test_loss, test_acc)

        # Save model
        if save_model:
            model_path = os.path.join(settings.models_dir, "gesture_model.h5")
            os.makedirs(settings.models_dir, exist_ok=True)

            logger.info("Saving LSTM model to %s", model_path)
            model.save(model_path)
            logger.info("Model saved successfully (size=%d bytes)", os.path.getsize(model_path))

            update_shared_state(
                "lstm",
                {
                    "model_path": model_path,
                    "source": source,
                    "feature_schema": settings.feature_schema,
                    "feature_schema_version": settings.feature_schema_version,
                    "feature_dimension": int(x_values.shape[2]),
                },
                publisher="LstmTrainer",
            )
            logger.info("Shared state updated with LSTM model metadata")

        self.model = model
        self.last_metrics = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "epochs_ran": int(epochs_ran),
            "samples": int(x_values.shape[0]),
            "classes": int(num_classes),
        }

        logger.info("LSTM training completed: metrics=%s", self.last_metrics)

        return model


def train_lstm(**kwargs):
    """Convenience function for training LSTM model."""
    trainer = LstmTrainer()
    return trainer.train(**kwargs)
