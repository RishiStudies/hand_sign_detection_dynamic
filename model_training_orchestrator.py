"""
Hand Sign Detection Model Trainer
=================================

A comprehensive training script for hand sign detection models that supports:
- Random Forest training on CSV landmark data
- LSTM training on video sequences (WLASL dataset)
- Data preprocessing and validation
- Model evaluation and saving

Usage:
    python model_training_orchestrator.py

This script will automatically detect available data and train appropriate models.

Author: Hand Sign Detection System
Date: 2026
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import warnings
import json
import cv2
from datetime import datetime
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("⚠️  Seaborn not available (optional for plotting)")


try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )
    from tensorflow.keras.optimizers import Adam

    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow available for LSTM training")
except ImportError as e:
    print(f"⚠️  TensorFlow not available: {e}")
    print("   LSTM training will be disabled")
    TENSORFLOW_AVAILABLE = False


try:
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
    )
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe available for feature extraction")
except (ImportError, AttributeError) as e:
    print(f"⚠️  MediaPipe not available or incompatible: {e}")
    print("   Using histogram-based features")
    MEDIAPIPE_AVAILABLE = False
    mp = None

warnings.filterwarnings("ignore")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


for dir_path in [MODELS_DIR, REPORTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)


class HandSignModelTrainer:
    """
    Comprehensive trainer for hand sign detection models
    """

    def __init__(self):
        self.models = {}
        self.training_history = {}
        self.data_info = {}

    def print_header(self, title):
        """Print a formatted header"""
        print("\n" + "=" * 60)
        print(f" {title}")
        print("=" * 60)

    def check_data_availability(self):
        """Check what datasets are available"""
        self.print_header("🔍 CHECKING DATA AVAILABILITY")

        available_data = {}

        csv_path = os.path.join(DATA_DIR, "hand_alphabet_data.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, header=None, on_bad_lines="skip")
                available_data["csv"] = {
                    "path": csv_path,
                    "samples": len(df),
                    "features": df.shape[1] - 1,
                    "classes": len(df.iloc[:, -1].unique()),
                }
                print(
                    f"✅ CSV Dataset: {available_data['csv']['samples']} samples, {available_data['csv']['classes']} classes"
                )
            except Exception as e:
                print(f"❌ CSV Dataset error: {e}")

        x_path = os.path.join(DATA_DIR, "X_data.npy")
        y_path = os.path.join(DATA_DIR, "y_data.npy")
        if os.path.exists(x_path) and os.path.exists(y_path):
            try:
                X = np.load(x_path)
                y = np.load(y_path)
                available_data["processed"] = {
                    "X_path": x_path,
                    "y_path": y_path,
                    "sequences": X.shape[0],
                    "timesteps": X.shape[1],
                    "features": X.shape[2],
                    "classes": len(np.unique(y)),
                }
                print(
                    f"✅ Processed Sequences: {available_data['processed']['sequences']} sequences, {available_data['processed']['classes']} classes"
                )
            except Exception as e:
                print(f"❌ Processed data error: {e}")

        wlasl_path = os.path.join(DATA_DIR, "WLASL_v0.3.json")
        videos_path = os.path.join(DATA_DIR, "videos")
        if os.path.exists(wlasl_path):
            try:
                with open(wlasl_path) as f:
                    data = json.load(f)
                available_data["wlasl"] = {
                    "json_path": wlasl_path,
                    "videos_path": videos_path,
                    "classes": len(data),
                    "videos_available": (
                        len(os.listdir(videos_path))
                        if os.path.exists(videos_path)
                        else 0
                    ),
                }
                print(
                    f"✅ WLASL Dataset: {available_data['wlasl']['classes']} classes, {available_data['wlasl']['videos_available']} videos"
                )
            except Exception as e:
                print(f"❌ WLASL data error: {e}")

        self.data_info = available_data
        return available_data

    def extract_features_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """Extract features from a single frame"""
        if MEDIAPIPE_AVAILABLE:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    row = []
                    for lm in hand_landmarks.landmark:
                        row += [lm.x, lm.y, lm.z]
                    return np.array(row)
            except Exception as e:
                print(f"MediaPipe failed, using histogram: {e}")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-10)
        return hist

    def train_random_forest(self, data_info=None):
        """Train Random Forest on CSV data"""
        self.print_header("🌳 TRAINING RANDOM FOREST CLASSIFIER")

        if data_info is None:
            if "csv" not in self.data_info:
                print("❌ No CSV data available for Random Forest training")
                return None
            data_info = self.data_info["csv"]

        print(f"📂 Loading data from: {data_info['path']}")
        data = pd.read_csv(data_info["path"], header=None, on_bad_lines="skip")
        data = data.dropna().reset_index(drop=True)

        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        print(f"📊 Dataset: {len(X)} samples, {len(np.unique(y))} classes")
        print(f"🎯 Feature dimensions: {X.shape[1]}")

        unique, counts = np.unique(y, return_counts=True)
        print("📈 Class distribution:")
        for cls, count in zip(unique, counts):
            print(f"   {cls}: {count} samples")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"📈 Training set: {len(X_train)} samples")
        print(f"🧪 Test set: {len(X_test)} samples")

        print("🚀 Training Random Forest...")
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )

        clf.fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        print(f"🎯 Training Accuracy: {train_acc:.2f}")
        print(f"🎯 Test Accuracy: {test_acc:.2f}")

        y_pred = clf.predict(X_test)
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODELS_DIR, f"random_forest_{timestamp}.pkl")
        labels_path = os.path.join(MODELS_DIR, f"class_labels_{timestamp}.npy")

        joblib.dump(clf, model_path)
        np.save(labels_path, clf.classes_)

        print(f"💾 Model saved: {model_path}")
        print(f"💾 Labels saved: {labels_path}")

        joblib.dump(clf, os.path.join(MODELS_DIR, "hand_alphabet_model.pkl"))
        np.save(os.path.join(MODELS_DIR, "class_labels.npy"), clf.classes_)

        self.models["random_forest"] = clf
        self.training_history["random_forest"] = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "model_path": model_path,
            "labels_path": labels_path,
            "timestamp": timestamp,
        }

        return clf

    def process_wlasl_videos(
        self, data_info=None, max_classes=10, max_videos_per_class=5
    ):
        """Process WLASL videos for LSTM training"""
        self.print_header("🎬 PROCESSING WLASL VIDEOS")

        if data_info is None:
            if "wlasl" not in self.data_info:
                print("❌ WLASL data not available")
                return None, None
            data_info = self.data_info["wlasl"]

        json_file = data_info["json_path"]
        video_folder = data_info["videos_path"]

        with open(json_file) as f:
            data = json.load(f)

        X = []
        y = []
        labels = []
        sequence_length = 30

        print(
            f"📹 Processing up to {max_classes} classes, {max_videos_per_class} videos each"
        )
        print(f"🎬 Sequence length: {sequence_length} frames")

        processed_classes = 0
        total_sequences = 0

        for label_index, item in enumerate(data[:max_classes]):
            gloss = item["gloss"]
            labels.append(gloss)
            print(f"🎯 Processing class {label_index + 1}/{max_classes}: {gloss}")

            videos_processed = 0
            for instance in item["instances"][:max_videos_per_class]:
                video_id = instance["video_id"]
                video_path = os.path.join(video_folder, video_id + ".mp4")

                if not os.path.exists(video_path):
                    continue

                try:
                    cap = cv2.VideoCapture(video_path)
                    sequence = []

                    while len(sequence) < sequence_length:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        features = self.extract_features_from_frame(frame)
                        sequence.append(features)

                    cap.release()

                    if len(sequence) == sequence_length:
                        X.append(sequence)
                        y.append(label_index)
                        videos_processed += 1
                        total_sequences += 1

                except Exception as e:
                    print(f"   ❌ Error processing {video_id}: {e}")
                    continue

            print(f"   ✅ Processed {videos_processed} videos for {gloss}")
            if videos_processed > 0:
                processed_classes += 1

        if total_sequences == 0:
            print("❌ No sequences could be processed")
            return None, None

        X = np.array(X)
        y = np.array(y)

        print(
            f"✅ Successfully processed {total_sequences} sequences from {processed_classes} classes"
        )
        print(f"📏 Data shape: {X.shape}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        X_path = os.path.join(DATA_DIR, f"X_data_{timestamp}.npy")
        y_path = os.path.join(DATA_DIR, f"y_data_{timestamp}.npy")
        labels_path = os.path.join(DATA_DIR, f"wlasl_labels_{timestamp}.npy")

        np.save(X_path, X)
        np.save(y_path, y)
        np.save(labels_path, np.array(labels))

        np.save(os.path.join(DATA_DIR, "X_data.npy"), X)
        np.save(os.path.join(DATA_DIR, "y_data.npy"), y)
        np.save(os.path.join(DATA_DIR, "wlasl_labels.npy"), np.array(labels))

        print(f"💾 Data saved to: {DATA_DIR}")

        self.data_info["processed"] = {
            "sequences": len(X),
            "timesteps": X.shape[1],
            "features": X.shape[2],
            "classes": len(labels),
            "X_path": X_path,
            "y_path": y_path,
            "labels_path": labels_path,
        }

        return X, y

    def train_lstm(self, X=None, y=None, data_info=None):
        """Train LSTM model on video sequences"""
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available for LSTM training")
            return None

        self.print_header("🧠 TRAINING LSTM NEURAL NETWORK")

        if X is None or y is None:
            if "processed" not in self.data_info:
                print(
                    "❌ No processed data available. Run process_wlasl_videos() first."
                )
                return None

            data_info = self.data_info["processed"]
            X = np.load(data_info["X_path"])
            y = np.load(data_info["y_path"])

        print(
            f"📊 Dataset: {X.shape[0]} sequences, {X.shape[1]} timesteps, {X.shape[2]} features"
        )
        print(f"🎯 Classes: {len(np.unique(y))}")

        num_classes = len(np.unique(y))
        y_cat = to_categorical(y, num_classes)

        min_samples_per_class = min(np.bincount(y))
        use_stratify = min_samples_per_class >= 2

        if use_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_cat, test_size=0.2, random_state=42, stratify=y
            )
            print("✅ Using stratified split")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_cat, test_size=0.2, random_state=42
            )
            print(
                "⚠️  Using random split (insufficient samples per class for stratification)"
            )

        print(f"📈 Training set: {len(X_train)} sequences")
        print(f"🧪 Test set: {len(X_test)} sequences")

        model = Sequential(
            [
                LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                BatchNormalization(),
                Dropout(0.3),
                LSTM(64, return_sequences=True),
                BatchNormalization(),
                Dropout(0.3),
                LSTM(32),
                BatchNormalization(),
                Dropout(0.3),
                Dense(64, activation="relu"),
                BatchNormalization(),
                Dropout(0.2),
                Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        print("🏗️ Model Architecture:")
        model.summary()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(MODELS_DIR, f"lstm_checkpoint_{timestamp}.h5")

        callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                patience=15,
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-6, verbose=1
            ),
        ]

        print("🚀 Training LSTM model...")
        history = model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=16,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1,
        )

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"🎯 Test Loss: {test_loss:.4f}")
        print(f"🎯 Test Accuracy: {test_acc:.4f}")

        model_path = os.path.join(MODELS_DIR, f"lstm_model_{timestamp}.h5")
        model.save(model_path)

        latest_path = os.path.join(MODELS_DIR, "gesture_model.h5")
        model.save(latest_path)

        print(f"💾 Model saved: {model_path}")
        print(f"💾 Latest model: {latest_path}")

        self.models["lstm"] = model
        self.training_history["lstm"] = {
            "train_accuracy": history.history["accuracy"][-1],
            "val_accuracy": history.history["val_accuracy"][-1],
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "epochs_trained": len(history.history["accuracy"]),
            "model_path": model_path,
            "checkpoint_path": checkpoint_path,
            "timestamp": timestamp,
            "history": history.history,
        }

        return model

    def generate_report(self):
        """Generate a training report"""
        self.print_header("📊 GENERATING TRAINING REPORT")

        report_path = os.path.join(
            REPORTS_DIR,
            f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )

        with open(report_path, "w") as f:
            f.write("Hand Sign Detection Training Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("DATA SUMMARY\n")
            f.write("-" * 20 + "\n")
            for data_type, info in self.data_info.items():
                f.write(f"{data_type.upper()}: {info}\n")
            f.write("\n")

            f.write("TRAINING RESULTS\n")
            f.write("-" * 20 + "\n")
            for model_type, results in self.training_history.items():
                f.write(f"\n{model_type.upper()} MODEL:\n")
                for key, value in results.items():
                    if key != "history":
                        f.write(f"  {key}: {value}\n")

        print(f"📄 Report saved: {report_path}")

    def run_full_training_pipeline(self):
        """Run the complete training pipeline"""
        print("🚀 STARTING HAND SIGN DETECTION TRAINING PIPELINE")
        print("=" * 60)

        available_data = self.check_data_availability()

        if not available_data:
            print("❌ No training data available!")
            return

        trained_models = []

        if "csv" in available_data:
            print("\n" + "🌳" * 20 + " RANDOM FOREST TRAINING " + "🌳" * 20)
            rf_model = self.train_random_forest()
            if rf_model:
                trained_models.append("Random Forest")

        if "wlasl" in available_data and TENSORFLOW_AVAILABLE:
            print("\n" + "🎬" * 20 + " WLASL PROCESSING " + "🎬" * 20)
            X, y = self.process_wlasl_videos()

            if X is not None and y is not None:
                print("\n" + "🧠" * 20 + " LSTM TRAINING " + "🧠" * 20)
                lstm_model = self.train_lstm(X, y)
                if lstm_model:
                    trained_models.append("LSTM")

        if trained_models:
            self.generate_report()

            print("\n" + "=" * 60)
            print("✅ TRAINING COMPLETED SUCCESSFULLY!")
            print(f"📊 Models trained: {', '.join(trained_models)}")
            print("📁 Models saved in: models/")
            print("📄 Training report: reports/")
            print("=" * 60)
        else:
            print("❌ No models were successfully trained")


def main():
    """Main training function"""
    trainer = HandSignModelTrainer()
    trainer.run_full_training_pipeline()


if __name__ == "__main__":
    main()
