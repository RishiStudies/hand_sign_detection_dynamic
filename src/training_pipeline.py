"""
Comprehensive Machine Learning Training Script for Hand Sign Detection
========================================================================

This script provides a unified interface for training various machine learning models
for hand sign and gesture recognition, including:

1. Random Forest Classifier (static gestures from CSV data)
2. LSTM Neural Network (dynamic gestures from video sequences)
3. Data preprocessing and feature extraction utilities

Usage:
    python src/training_pipeline.py --command train-rf --profile pi_zero
    python src/training_pipeline.py --command preprocess --profile pi_zero
    python src/training_pipeline.py --command device-all --profile pi_zero
    python src/training_pipeline.py --model all

Author: Hand Sign Detection System
Date: 2026
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
from tqdm import tqdm
import json
import cv2
import shutil
from datetime import datetime

try:
    from .shared_artifacts import (
        DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        SHARED_STATE_PATH,
        update_shared_state,
    )
except ImportError:
    from shared_artifacts import (
        DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        SHARED_STATE_PATH,
        update_shared_state,
    )

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping

    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. LSTM training will be disabled.")
    TENSORFLOW_AVAILABLE = False


try:
    import mediapipe as mp

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
    )
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError) as e:
    print(
        f"Warning: MediaPipe not available or incompatible version ({e}). Using histogram-based features."
    )
    MEDIAPIPE_AVAILABLE = False
    mp = None

warnings.filterwarnings("ignore")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


DEVICE_PROFILES = {
    "pi_zero": {
        "rf_estimators": 64,
        "rf_max_depth": 16,
        "rf_n_jobs": 1,
        "max_classes": 8,
        "max_videos_per_class": 3,
        "sequence_length": 20,
        "frame_stride": 3,
        "lstm_low_end": True,
    },
    "full": {
        "rf_estimators": 300,
        "rf_max_depth": 20,
        "rf_n_jobs": -1,
        "max_classes": 50,
        "max_videos_per_class": 8,
        "sequence_length": 30,
        "frame_stride": 1,
        "lstm_low_end": False,
    },
}


class HandSignTrainer:
    """Comprehensive trainer for hand sign detection models"""

    def __init__(self):
        self.models = {}
        self.labels = {}
        self.last_metrics = {}
        self.last_preprocess_summary = {}

    def get_profile(self, profile_name: str):
        if profile_name not in DEVICE_PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}")
        return DEVICE_PROFILES[profile_name]

    def extract_features_from_frame(self, frame: np.ndarray) -> np.ndarray:
        """Extract features from a single frame using histogram or MediaPipe"""
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

    def train_random_forest(
        self,
        data_path=None,
        save_model=True,
        low_end=False,
        profile_name="full",
    ):
        """Train Random Forest classifier on CSV data"""
        print("🔍 Training Random Forest Classifier...")

        if data_path is None:
            data_path = os.path.join(DATA_DIR, "hand_alphabet_data.csv")

        if not os.path.exists(data_path):
            print(f"❌ CSV data file not found: {data_path}")
            return None

        data = pd.read_csv(data_path, header=None, on_bad_lines="skip")
        data = data.dropna().reset_index(drop=True)

        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"📊 Dataset: {len(X)} samples, {len(np.unique(y))} classes")
        print(f"📈 Training set: {len(X_train)} samples")
        print(f"🧪 Test set: {len(X_test)} samples")

        profile = self.get_profile(profile_name)
        # Keep low_end backward-compatible while giving profiles precedence.
        if low_end and profile_name == "full":
            n_estimators = 100
            max_depth = 18
            n_jobs = 1
        else:
            n_estimators = profile["rf_estimators"]
            max_depth = profile["rf_max_depth"]
            n_jobs = profile["rf_n_jobs"]

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=n_jobs,
        )

        print("🌳 Training Random Forest...")
        clf.fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        print(f"🎯 Training Accuracy: {train_acc:.2f}")
        print(f"🎯 Test Accuracy: {test_acc:.2f}")

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))

        if save_model:
            model_path = os.path.join(MODELS_DIR, "hand_alphabet_model.pkl")
            labels_path = os.path.join(MODELS_DIR, "class_labels.npy")

            joblib.dump(clf, model_path)
            np.save(labels_path, clf.classes_)
            update_shared_state(
                "random_forest",
                {
                    "model_path": model_path,
                    "labels_path": labels_path,
                    "profile": profile_name,
                },
                publisher="training_pipeline",
            )

            print(f"💾 Model saved to: {model_path}")
            print(f"💾 Labels saved to: {labels_path}")

        self.models["random_forest"] = clf
        self.labels["random_forest"] = clf.classes_
        self.last_metrics["random_forest"] = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "accuracy_score": float(acc),
            "samples": int(len(X)),
            "classes": int(len(np.unique(y))),
            "profile": profile_name,
            "n_estimators": int(n_estimators),
            "max_depth": int(max_depth),
        }

        return clf

    def process_wlasl_videos(
        self,
        json_file=None,
        video_folder=None,
        save_data=True,
        max_classes=5,
        max_videos_per_class=3,
        sequence_length=30,
        frame_stride=1,
    ):
        """Process WLASL videos and extract features for LSTM training"""
        print("🎬 Processing WLASL Videos...")

        if json_file is None:
            json_file = os.path.join(DATA_DIR, "WLASL_v0.3.json")
        if video_folder is None:
            video_folder = os.path.join(DATA_DIR, "videos")

        if not os.path.exists(json_file):
            print(f"❌ JSON file not found: {json_file}")
            return None, None

        if not os.path.exists(video_folder):
            print(f"❌ Video folder not found: {video_folder}")
            return None, None

        with open(json_file) as f:
            data = json.load(f)

        X = []
        y = []
        labels = []

        print(f"📹 Processing videos from: {video_folder}")
        print(
            f"⚙️ Limits: max_classes={max_classes}, max_videos_per_class={max_videos_per_class}, sequence_length={sequence_length}, frame_stride={frame_stride}"
        )

        processed_videos = 0
        missing_videos = 0

        for label_index, item in enumerate(tqdm(data[:max_classes])):
            gloss = item["gloss"]
            labels.append(gloss)

            for instance in item["instances"][:max_videos_per_class]:
                video_id = instance["video_id"]
                video_path = os.path.join(video_folder, video_id + ".mp4")

                if not os.path.exists(video_path):
                    missing_videos += 1
                    continue

                cap = cv2.VideoCapture(video_path)
                sequence = []
                frame_index = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_index += 1
                    if frame_stride > 1 and frame_index % frame_stride != 0:
                        continue

                    features = self.extract_features_from_frame(frame)
                    sequence.append(features)

                    if len(sequence) >= sequence_length:
                        break

                cap.release()

                if len(sequence) >= sequence_length:
                    sequence = sequence[:sequence_length]

                    while len(sequence) < sequence_length:
                        sequence.append(np.zeros_like(sequence[0]))

                    X.append(sequence)
                    y.append(label_index)
                    processed_videos += 1

        X = np.array(X)
        y = np.array(y)

        print(f"✅ Processed {len(X)} video sequences")
        print(f"📏 Sequence length: {sequence_length}")
        print(f"🎯 Feature dimensions: {X.shape[2] if len(X.shape) > 2 else 'N/A'}")
        print(f"📼 Source videos used: {processed_videos}, missing: {missing_videos}")

        if save_data:
            x_path = os.path.join(DATA_DIR, "X_data.npy")
            y_path = os.path.join(DATA_DIR, "y_data.npy")
            labels_path = os.path.join(MODELS_DIR, "wlasl_labels.npy")
            np.save(x_path, X)
            np.save(y_path, y)
            np.save(labels_path, np.array(labels))
            update_shared_state(
                "dynamic_data",
                {
                    "x_path": x_path,
                    "y_path": y_path,
                    "labels_path": labels_path,
                    "sequence_length": sequence_length,
                    "frame_stride": frame_stride,
                },
                publisher="training_pipeline",
            )
            update_shared_state(
                "lstm",
                {"labels_path": labels_path},
                publisher="training_pipeline",
            )

            print("💾 Processed data saved to data/ directory")

        self.last_preprocess_summary = {
            "sequences": int(len(X)),
            "sequence_length": int(sequence_length),
            "feature_dims": int(X.shape[2]) if len(X.shape) > 2 else 0,
            "processed_videos": int(processed_videos),
            "missing_videos": int(missing_videos),
            "max_classes": int(max_classes),
            "max_videos_per_class": int(max_videos_per_class),
            "frame_stride": int(frame_stride),
            "json_file": json_file,
            "video_folder": video_folder,
        }

        return X, y

    def evaluate_random_forest(self, data_path=None, model_path=None):
        """Evaluate currently active or loaded Random Forest model on CSV data."""
        if data_path is None:
            data_path = os.path.join(DATA_DIR, "hand_alphabet_data.csv")
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "hand_alphabet_model.pkl")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CSV data file not found: {data_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        data = pd.read_csv(data_path, header=None, on_bad_lines="skip").dropna()
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        clf = joblib.load(model_path)
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"🎯 Evaluation accuracy on full CSV: {acc:.4f}")

        self.last_metrics["random_forest_eval"] = {
            "accuracy": float(acc),
            "samples": int(len(X)),
            "model_path": model_path,
        }
        return acc

    def package_artifacts(self, profile_name="pi_zero", note=""):
        """Create a metadata package for local device training artifacts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_id = f"local_{profile_name}_{timestamp}"
        package_dir = os.path.join(MODELS_DIR, "packages")
        os.makedirs(package_dir, exist_ok=True)

        files_to_copy = [
            os.path.join(MODELS_DIR, "hand_alphabet_model.pkl"),
            os.path.join(MODELS_DIR, "class_labels.npy"),
            os.path.join(DATA_DIR, "X_data.npy"),
            os.path.join(DATA_DIR, "y_data.npy"),
            os.path.join(MODELS_DIR, "wlasl_labels.npy"),
        ]

        copied_files = []
        for src in files_to_copy:
            if os.path.exists(src):
                dst = os.path.join(package_dir, f"{package_id}__{os.path.basename(src)}")
                shutil.copy2(src, dst)
                copied_files.append(dst)

        metadata = {
            "package_id": package_id,
            "created_at": datetime.now().isoformat(),
            "profile": profile_name,
            "note": note,
            "metrics": self.last_metrics,
            "preprocess_summary": self.last_preprocess_summary,
            "shared_state_path": SHARED_STATE_PATH,
            "files": copied_files,
        }

        metadata_path = os.path.join(package_dir, f"{package_id}.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        report_path = os.path.join(REPORTS_DIR, f"local_training_{timestamp}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"📦 Package created: {metadata_path}")
        return metadata_path

    def train_lstm(self, X=None, y=None, save_model=True, low_end=False):
        """Train LSTM model on processed video sequences"""
        if not TENSORFLOW_AVAILABLE:
            print("❌ TensorFlow not available. Cannot train LSTM model.")
            return None

        print("🧠 Training LSTM Neural Network...")

        if X is None or y is None:
            X_path = os.path.join(DATA_DIR, "X_data.npy")
            y_path = os.path.join(DATA_DIR, "y_data.npy")

            if not os.path.exists(X_path) or not os.path.exists(y_path):
                print("❌ Processed data not found. Run process_wlasl_videos() first.")
                return None

            X = np.load(X_path)
            y = np.load(y_path)

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
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_cat, test_size=0.2, random_state=42
            )
            print(
                "⚠️  Using random split (no stratification - insufficient samples per class)"
            )

        # Scale down LSTM units in low-end mode to reduce memory and inference cost
        if low_end:
            lstm_units = [64, 32, 16]
            dense_units = 32
            epochs = 30
        else:
            lstm_units = [128, 64, 32]
            dense_units = 64
            epochs = 50

        model = Sequential(
            [
                LSTM(lstm_units[0], return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
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

        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

        print("🏗️ Model Architecture:")
        model.summary()

        early_stopping = EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True
        )

        print("🚀 Training LSTM model...")
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=16,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1,
        )

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"🎯 Test Loss: {test_loss:.4f}")
        print(f"🎯 Test Accuracy: {test_acc:.4f}")

        if save_model:
            model_path = os.path.join(MODELS_DIR, "gesture_model.h5")
            model.save(model_path)
            update_shared_state(
                "lstm",
                {"model_path": model_path},
                publisher="training_pipeline",
            )
            print(f"💾 Model saved to: {model_path}")

        self.models["lstm"] = model

        return model

    def train_all_models(self, low_end=False):
        """Train all available models"""
        print("🚀 Training All Available Models")
        print("=" * 50)

        print("\n" + "=" * 30 + " RANDOM FOREST " + "=" * 30)
        rf_model = self.train_random_forest(low_end=low_end)

        print("\n" + "=" * 30 + " WLASL PROCESSING " + "=" * 30)
        X, y = self.process_wlasl_videos()

        if X is not None and y is not None:
            print("\n" + "=" * 30 + " LSTM TRAINING " + "=" * 30)
            lstm_model = self.train_lstm(X, y, low_end=low_end)

        print("\n" + "=" * 50)
        print("✅ All training completed!")
        print("📊 Models trained:")
        if "random_forest" in self.models:
            print("  • Random Forest (static gestures)")
        if "lstm" in self.models:
            print("  • LSTM (dynamic gestures)")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Local training software for hand sign detection"
    )
    parser.add_argument(
        "--command",
        choices=[
            "legacy",
            "preprocess",
            "train-rf",
            "evaluate",
            "package",
            "device-all",
        ],
        default="legacy",
        help="Device-local command. 'legacy' preserves older --model behavior.",
    )
    parser.add_argument(
        "--profile",
        choices=["pi_zero", "full"],
        default="pi_zero",
        help="Hardware profile for local training defaults.",
    )
    parser.add_argument(
        "--max-classes",
        type=int,
        default=None,
        help="Optional override for WLASL preprocessing class cap.",
    )
    parser.add_argument(
        "--max-videos-per-class",
        type=int,
        default=None,
        help="Optional override for WLASL preprocessing video cap per class.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Optional override for generated sequence length.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help="Optional override for frame subsampling (>=1).",
    )
    parser.add_argument(
        "--json-file",
        default=None,
        help="WLASL metadata JSON path override.",
    )
    parser.add_argument(
        "--video-folder",
        default=None,
        help="Video folder path override.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="CSV path override for RF training/evaluation.",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Optional metadata note when packaging artifacts.",
    )
    parser.add_argument(
        "--model",
        choices=["random_forest", "lstm", "all"],
        default="all",
        help="Model to train",
    )
    parser.add_argument(
        "--data",
        choices=["csv", "wlasl"],
        help="Data source (for specific model training)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save trained models"
    )
    parser.add_argument(
        "--low-end",
        action="store_true",
        help="Use lighter hyperparameters optimized for low-end devices and microcomputers",
    )

    args = parser.parse_args()

    trainer = HandSignTrainer()
    save_models = not args.no_save
    low_end = args.low_end
    profile = trainer.get_profile(args.profile)

    max_classes = args.max_classes if args.max_classes is not None else profile["max_classes"]
    max_videos = (
        args.max_videos_per_class
        if args.max_videos_per_class is not None
        else profile["max_videos_per_class"]
    )
    sequence_length = (
        args.sequence_length if args.sequence_length is not None else profile["sequence_length"]
    )
    frame_stride = args.frame_stride if args.frame_stride is not None else profile["frame_stride"]

    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")

    if args.command != "legacy":
        print(f"🧩 Command mode: {args.command} | profile={args.profile}")

        if args.command == "preprocess":
            trainer.process_wlasl_videos(
                json_file=args.json_file,
                video_folder=args.video_folder,
                save_data=save_models,
                max_classes=max_classes,
                max_videos_per_class=max_videos,
                sequence_length=sequence_length,
                frame_stride=frame_stride,
            )
            return

        if args.command == "train-rf":
            trainer.train_random_forest(
                data_path=args.csv_path,
                save_model=save_models,
                low_end=low_end,
                profile_name=args.profile,
            )
            return

        if args.command == "evaluate":
            trainer.evaluate_random_forest(data_path=args.csv_path)
            return

        if args.command == "package":
            trainer.package_artifacts(profile_name=args.profile, note=args.note)
            return

        if args.command == "device-all":
            trainer.process_wlasl_videos(
                json_file=args.json_file,
                video_folder=args.video_folder,
                save_data=save_models,
                max_classes=max_classes,
                max_videos_per_class=max_videos,
                sequence_length=sequence_length,
                frame_stride=frame_stride,
            )
            trainer.train_random_forest(
                data_path=args.csv_path,
                save_model=save_models,
                low_end=low_end,
                profile_name=args.profile,
            )
            trainer.evaluate_random_forest(data_path=args.csv_path)
            trainer.package_artifacts(profile_name=args.profile, note=args.note)
            return

    if low_end:
        print("⚡ Low-end mode enabled: using lighter model hyperparameters")

    if args.model == "random_forest" or args.model == "all":
        trainer.train_random_forest(
            save_model=save_models,
            low_end=low_end,
            profile_name=args.profile,
            data_path=args.csv_path,
        )

    if args.model == "lstm" or args.model == "all":
        if args.data == "wlasl" or args.model == "all":
            X, y = trainer.process_wlasl_videos(
                save_data=save_models,
                json_file=args.json_file,
                video_folder=args.video_folder,
                max_classes=max_classes,
                max_videos_per_class=max_videos,
                sequence_length=sequence_length,
                frame_stride=frame_stride,
            )
            if X is not None and y is not None:
                trainer.train_lstm(X, y, save_model=save_models, low_end=low_end)
        else:
            trainer.train_lstm(save_model=save_models, low_end=low_end)

    if args.model == "all":
        trainer.train_all_models(low_end=low_end)


if __name__ == "__main__":
    main()
