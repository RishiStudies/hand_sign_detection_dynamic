import json
import os
import shutil
from importlib import import_module
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

try:
    from ..shared_artifacts import update_shared_state
except ImportError:
    from shared_artifacts import update_shared_state

from .config import (
    DATA_DIR,
    DEVICE_PROFILES,
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_VERSION,
    MODELS_DIR,
    REPORTS_DIR,
    SHARED_STATE_PATH,
    TENSORFLOW_AVAILABLE,
)
from .features import extract_features_from_bytes, extract_features_from_frame

if TENSORFLOW_AVAILABLE:
    EarlyStopping = import_module("tensorflow.keras.callbacks").EarlyStopping
    keras_layers = import_module("tensorflow.keras.layers")
    LSTM = keras_layers.LSTM
    Dense = keras_layers.Dense
    Dropout = keras_layers.Dropout
    Sequential = import_module("tensorflow.keras.models").Sequential
    to_categorical = import_module("tensorflow.keras.utils").to_categorical
else:
    EarlyStopping = None
    LSTM = None
    Dense = None
    Dropout = None
    Sequential = None
    to_categorical = None


class TrainingService:
    def __init__(self):
        self.models: Dict[str, object] = {}
        self.labels: Dict[str, np.ndarray] = {}
        self.last_metrics: Dict[str, Dict[str, object]] = {}
        self.last_preprocess_summary: Dict[str, object] = {}
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)

    def get_profile(self, profile_name: str) -> Dict[str, object]:
        if profile_name not in DEVICE_PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}")
        return DEVICE_PROFILES[profile_name]

    def train_random_forest_from_csv(
        self,
        data_path: Optional[str] = None,
        save_model: bool = True,
        low_end: bool = False,
        profile_name: str = "full",
        source: str = "training_module_csv",
    ) -> Dict[str, object]:
        if data_path is None:
            data_path = os.path.join(DATA_DIR, "hand_alphabet_data.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"CSV data file not found: {data_path}")

        data = pd.read_csv(data_path, header=None, on_bad_lines="skip")
        data = data.dropna().reset_index(drop=True)
        x_values = data.iloc[:, :-1].values
        y_values = data.iloc[:, -1].values

        return self._train_random_forest(
            x_values=x_values,
            y_values=y_values,
            save_model=save_model,
            low_end=low_end,
            profile_name=profile_name,
            source=source,
        )

    def train_random_forest_from_manifest(
        self,
        manifest_path: str,
        save_model: bool = True,
        profile_name: str = "full",
        source: str = "training_module_manifest",
    ) -> Dict[str, object]:
        with open(manifest_path, "r", encoding="utf-8") as file_obj:
            manifest = json.load(file_obj)

        samples = manifest.get("samples", [])
        if not samples:
            raise ValueError("No samples found in training manifest")

        x_values = []
        y_values = []
        label_to_idx: Dict[str, int] = {}

        for sample in samples:
            label = str(sample["label"])
            if label not in label_to_idx:
                label_to_idx[label] = len(label_to_idx)
            with open(sample["file_path"], "rb") as image_file:
                features = extract_features_from_bytes(image_file.read())
            x_values.append(features.flatten())
            y_values.append(label_to_idx[label])

        metrics = self._train_random_forest(
            x_values=np.array(x_values),
            y_values=np.array(y_values),
            class_labels=np.array(list(label_to_idx.keys())),
            save_model=save_model,
            profile_name=profile_name,
            source=source,
        )
        shutil.rmtree(os.path.dirname(manifest_path), ignore_errors=True)
        return metrics

    def _train_random_forest(
        self,
        x_values: np.ndarray,
        y_values: np.ndarray,
        save_model: bool = True,
        low_end: bool = False,
        profile_name: str = "full",
        source: str = "training_module_rf",
        class_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        profile = self.get_profile(profile_name)
        if low_end and profile_name == "full":
            n_estimators = 100
            max_depth = 18
            n_jobs = 1
        else:
            n_estimators = int(profile["rf_estimators"])
            max_depth = int(profile["rf_max_depth"])
            n_jobs = int(profile["rf_n_jobs"])

        stratify = y_values if len(np.unique(y_values)) > 1 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x_values,
            y_values,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=n_jobs,
        )
        clf.fit(x_train, y_train)

        train_acc = clf.score(x_train, y_train)
        test_acc = clf.score(x_test, y_test)
        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred)

        labels_to_save = clf.classes_
        if class_labels is not None:
            labels_to_save = class_labels[clf.classes_.astype(int)]

        if save_model:
            model_path = os.path.join(MODELS_DIR, "hand_alphabet_model.pkl")
            labels_path = os.path.join(MODELS_DIR, "class_labels.npy")
            joblib.dump(clf, model_path)
            np.save(labels_path, labels_to_save)
            update_shared_state(
                "random_forest",
                {
                    "model_path": model_path,
                    "labels_path": labels_path,
                    "profile": profile_name,
                    "source": source,
                    "feature_schema": FEATURE_SCHEMA,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "feature_dimension": int(x_values.shape[1]),
                },
                publisher="training_module",
            )

        metrics = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "accuracy_score": float(acc),
            "samples": int(len(x_values)),
            "classes": int(len(labels_to_save)),
            "profile": profile_name,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }
        self.models["random_forest"] = clf
        self.labels["random_forest"] = labels_to_save
        self.last_metrics["random_forest"] = metrics
        return metrics

    def process_wlasl_videos(
        self,
        json_file: Optional[str] = None,
        video_folder: Optional[str] = None,
        save_data: bool = True,
        max_classes: int = 5,
        max_videos_per_class: int = 3,
        sequence_length: int = 30,
        frame_stride: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if json_file is None:
            json_file = os.path.join(DATA_DIR, "WLASL_v0.3.json")
        if video_folder is None:
            video_folder = os.path.join(DATA_DIR, "videos")
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        if not os.path.exists(video_folder):
            raise FileNotFoundError(f"Video folder not found: {video_folder}")

        with open(json_file, encoding="utf-8") as file_obj:
            data = json.load(file_obj)

        x_values = []
        y_values = []
        labels = []
        processed_videos = 0
        missing_videos = 0

        for label_index, item in enumerate(data[:max_classes]):
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
                    sequence.append(extract_features_from_frame(frame))
                    if len(sequence) >= sequence_length:
                        break
                cap.release()

                if len(sequence) >= sequence_length:
                    sequence = sequence[:sequence_length]
                    while len(sequence) < sequence_length:
                        sequence.append(np.zeros_like(sequence[0]))
                    x_values.append(sequence)
                    y_values.append(label_index)
                    processed_videos += 1

        x_array = np.array(x_values)
        y_array = np.array(y_values)
        if save_data:
            x_path = os.path.join(DATA_DIR, "X_data.npy")
            y_path = os.path.join(DATA_DIR, "y_data.npy")
            labels_path = os.path.join(MODELS_DIR, "wlasl_labels.npy")
            np.save(x_path, x_array)
            np.save(y_path, y_array)
            np.save(labels_path, np.array(labels))
            update_shared_state(
                "dynamic_data",
                {
                    "x_path": x_path,
                    "y_path": y_path,
                    "labels_path": labels_path,
                    "sequence_length": sequence_length,
                    "frame_stride": frame_stride,
                    "feature_schema": FEATURE_SCHEMA,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "feature_dimension": int(x_array.shape[2]) if len(x_array.shape) > 2 else 0,
                },
                publisher="training_module",
            )
            update_shared_state(
                "lstm",
                {
                    "labels_path": labels_path,
                    "feature_schema": FEATURE_SCHEMA,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "feature_dimension": int(x_array.shape[2]) if len(x_array.shape) > 2 else 0,
                },
                publisher="training_module",
            )

        self.last_preprocess_summary = {
            "sequences": int(len(x_array)),
            "sequence_length": int(sequence_length),
            "feature_dims": int(x_array.shape[2]) if len(x_array.shape) > 2 else 0,
            "processed_videos": int(processed_videos),
            "missing_videos": int(missing_videos),
            "max_classes": int(max_classes),
            "max_videos_per_class": int(max_videos_per_class),
            "frame_stride": int(frame_stride),
        }
        return x_array, y_array

    def train_lstm(
        self,
        x_values: Optional[np.ndarray] = None,
        y_values: Optional[np.ndarray] = None,
        save_model: bool = True,
        low_end: bool = False,
        source: str = "training_module_lstm",
    ):
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow not available. Cannot train LSTM model.")
        if x_values is None or y_values is None:
            x_path = os.path.join(DATA_DIR, "X_data.npy")
            y_path = os.path.join(DATA_DIR, "y_data.npy")
            if not os.path.exists(x_path) or not os.path.exists(y_path):
                raise FileNotFoundError("Processed data not found. Run preprocessing first.")
            x_values = np.load(x_path)
            y_values = np.load(y_path)

        num_classes = len(np.unique(y_values))
        y_cat = to_categorical(y_values, num_classes)
        min_samples_per_class = min(np.bincount(y_values))
        use_stratify = min_samples_per_class >= 2

        if use_stratify:
            x_train, x_test, y_train, y_test = train_test_split(
                x_values, y_cat, test_size=0.2, random_state=42, stratify=y_values
            )
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x_values, y_cat, test_size=0.2, random_state=42
            )

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
                LSTM(lstm_units[0], return_sequences=True, input_shape=(x_values.shape[1], x_values.shape[2])),
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

        early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=16,
            validation_data=(x_test, y_test),
            callbacks=[early_stopping],
            verbose=1,
        )
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

        if save_model:
            model_path = os.path.join(MODELS_DIR, "gesture_model.h5")
            model.save(model_path)
            update_shared_state(
                "lstm",
                {
                    "model_path": model_path,
                    "source": source,
                    "feature_schema": FEATURE_SCHEMA,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "feature_dimension": int(x_values.shape[2]),
                },
                publisher="training_module",
            )

        self.models["lstm"] = model
        self.last_metrics["lstm"] = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "epochs_ran": int(len(history.history.get("loss", []))),
            "samples": int(x_values.shape[0]),
            "classes": int(num_classes),
        }
        return model

    def evaluate_random_forest(self, data_path: Optional[str] = None, model_path: Optional[str] = None) -> float:
        if data_path is None:
            data_path = os.path.join(DATA_DIR, "hand_alphabet_data.csv")
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "hand_alphabet_model.pkl")
        data = pd.read_csv(data_path, header=None, on_bad_lines="skip").dropna()
        x_values = data.iloc[:, :-1].values
        y_values = data.iloc[:, -1].values
        clf = joblib.load(model_path)
        y_pred = clf.predict(x_values)
        acc = accuracy_score(y_values, y_pred)
        self.last_metrics["random_forest_eval"] = {
            "accuracy": float(acc),
            "samples": int(len(x_values)),
            "model_path": model_path,
        }
        return acc

    def package_artifacts(self, profile_name: str = "pi_zero", note: str = "") -> str:
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
        with open(metadata_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, indent=2)

        report_path = os.path.join(REPORTS_DIR, f"local_training_{timestamp}.json")
        with open(report_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, indent=2)

        return metadata_path

    def run_device_pipeline(
        self,
        profile_name: str = "pi_zero",
        note: str = "",
        csv_path: Optional[str] = None,
        json_file: Optional[str] = None,
        video_folder: Optional[str] = None,
        max_classes: Optional[int] = None,
        max_videos_per_class: Optional[int] = None,
        sequence_length: Optional[int] = None,
        frame_stride: Optional[int] = None,
    ) -> Dict[str, object]:
        profile = self.get_profile(profile_name)
        self.process_wlasl_videos(
            json_file=json_file,
            video_folder=video_folder,
            save_data=True,
            max_classes=max_classes if max_classes is not None else int(profile["max_classes"]),
            max_videos_per_class=max_videos_per_class if max_videos_per_class is not None else int(profile["max_videos_per_class"]),
            sequence_length=sequence_length if sequence_length is not None else int(profile["sequence_length"]),
            frame_stride=frame_stride if frame_stride is not None else int(profile["frame_stride"]),
        )
        rf_metrics = self.train_random_forest_from_csv(
            data_path=csv_path,
            save_model=True,
            low_end=profile_name == "pi_zero",
            profile_name=profile_name,
            source="training_module_device_pipeline",
        )
        eval_acc = self.evaluate_random_forest(data_path=csv_path)
        package_path = self.package_artifacts(profile_name=profile_name, note=note)
        return {
            "random_forest": rf_metrics,
            "evaluation_accuracy": eval_acc,
            "package_path": package_path,
            "preprocess_summary": self.last_preprocess_summary,
        }

    def train_all_models(self, profile_name: str = "full") -> Dict[str, object]:
        profile = self.get_profile(profile_name)
        rf_metrics = self.train_random_forest_from_csv(
            save_model=True,
            low_end=profile_name == "pi_zero",
            profile_name=profile_name,
            source="training_module_train_all",
        )
        x_values, y_values = self.process_wlasl_videos(
            save_data=True,
            max_classes=int(profile["max_classes"]),
            max_videos_per_class=int(profile["max_videos_per_class"]),
            sequence_length=int(profile["sequence_length"]),
            frame_stride=int(profile["frame_stride"]),
        )
        self.train_lstm(
            x_values=x_values,
            y_values=y_values,
            save_model=True,
            low_end=bool(profile.get("lstm_low_end", False)),
            source="training_module_train_all",
        )
        return {
            "random_forest": rf_metrics,
            "lstm": self.last_metrics.get("lstm", {}),
            "preprocess_summary": self.last_preprocess_summary,
        }
