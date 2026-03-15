"""
Comprehensive Machine Learning Training Script for Hand Sign Detection
========================================================================

This script provides a unified interface for training various machine learning models
for hand sign and gesture recognition, including:

1. Random Forest Classifier (static gestures from CSV data)
2. LSTM Neural Network (dynamic gestures from video sequences)
3. Data preprocessing and feature extraction utilities

Usage:
    python src/training_pipeline.py --model random_forest --data csv
    python src/training_pipeline.py --model lstm --data wlasl
    python src/training_pipeline.py --model all

Author: Hand Sign Detection System
Date: 2026
"""

import argparse
import os
import sys
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
DATA_DIR = os.path.join(SCRIPT_DIR, "../data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "../models")


os.makedirs(MODELS_DIR, exist_ok=True)


class HandSignTrainer:
    """Comprehensive trainer for hand sign detection models"""

    def __init__(self):
        self.models = {}
        self.labels = {}

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

    def train_random_forest(self, data_path=None, save_model=True):
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

        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        print("🌳 Training Random Forest...")
        clf.fit(X_train, y_train)

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        print(f"🎯 Training Accuracy: {train_acc:.2f}")
        print(f"🎯 Test Accuracy: {test_acc:.2f}")

        y_pred = clf.predict(X_test)
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred))

        if save_model:
            model_path = os.path.join(MODELS_DIR, "hand_alphabet_model.pkl")
            labels_path = os.path.join(MODELS_DIR, "class_labels.npy")

            joblib.dump(clf, model_path)
            np.save(labels_path, clf.classes_)

            print(f"💾 Model saved to: {model_path}")
            print(f"💾 Labels saved to: {labels_path}")

        self.models["random_forest"] = clf
        self.labels["random_forest"] = clf.classes_

        return clf

    def process_wlasl_videos(self, json_file=None, video_folder=None, save_data=True):
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
        sequence_length = 30

        print(f"📹 Processing videos from: {video_folder}")

        for label_index, item in enumerate(tqdm(data[:5])):
            gloss = item["gloss"]
            labels.append(gloss)

            for instance in item["instances"][:3]:
                video_id = instance["video_id"]
                video_path = os.path.join(video_folder, video_id + ".mp4")

                if not os.path.exists(video_path):
                    continue

                cap = cv2.VideoCapture(video_path)
                sequence = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    features = self.extract_features_from_frame(frame)
                    sequence.append(features)

                cap.release()

                if len(sequence) >= sequence_length:
                    sequence = sequence[:sequence_length]

                    while len(sequence) < sequence_length:
                        sequence.append(np.zeros_like(sequence[0]))

                    X.append(sequence)
                    y.append(label_index)

        X = np.array(X)
        y = np.array(y)

        print(f"✅ Processed {len(X)} video sequences")
        print(f"📏 Sequence length: {sequence_length}")
        print(f"🎯 Feature dimensions: {X.shape[2] if len(X.shape) > 2 else 'N/A'}")

        if save_data:
            np.save(os.path.join(DATA_DIR, "X_data.npy"), X)
            np.save(os.path.join(DATA_DIR, "y_data.npy"), y)
            np.save(os.path.join(MODELS_DIR, "wlasl_labels.npy"), np.array(labels))

            print("💾 Processed data saved to data/ directory")

        return X, y

    def train_lstm(self, X=None, y=None, save_model=True):
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

        model = Sequential(
            [
                LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.3),
                LSTM(64, return_sequences=True),
                Dropout(0.3),
                LSTM(32),
                Dropout(0.3),
                Dense(64, activation="relu"),
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
            epochs=50,
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
            print(f"💾 Model saved to: {model_path}")

        self.models["lstm"] = model

        return model

    def train_all_models(self):
        """Train all available models"""
        print("🚀 Training All Available Models")
        print("=" * 50)

        print("\n" + "=" * 30 + " RANDOM FOREST " + "=" * 30)
        rf_model = self.train_random_forest()

        print("\n" + "=" * 30 + " WLASL PROCESSING " + "=" * 30)
        X, y = self.process_wlasl_videos()

        if X is not None and y is not None:
            print("\n" + "=" * 30 + " LSTM TRAINING " + "=" * 30)
            lstm_model = self.train_lstm(X, y)

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
        description="Train ML models for hand sign detection"
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

    args = parser.parse_args()

    trainer = HandSignTrainer()
    save_models = not args.no_save

    if args.model == "random_forest" or args.model == "all":
        trainer.train_random_forest(save_model=save_models)

    if args.model == "lstm" or args.model == "all":
        if args.data == "wlasl" or args.model == "all":
            X, y = trainer.process_wlasl_videos(save_data=save_models)
            if X is not None and y is not None:
                trainer.train_lstm(X, y, save_model=save_models)
        else:
            trainer.train_lstm(save_model=save_models)

    if args.model == "all":
        trainer.train_all_models()


if __name__ == "__main__":
    main()
