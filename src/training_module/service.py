import json
import logging
import os
import shutil
import tempfile
import zipfile
import hashlib
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

logger = logging.getLogger("training_service")

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
        logger.info(f"TrainingService initialized (models_dir={MODELS_DIR}, reports_dir={REPORTS_DIR})")

    def get_profile(self, profile_name: str) -> Dict[str, object]:
        if profile_name not in DEVICE_PROFILES:
            logger.error(f"Unknown profile requested: {profile_name}. Available: {list(DEVICE_PROFILES.keys())}")
            raise ValueError(f"Unknown profile: {profile_name}")
        logger.debug(f"Profile loaded: {profile_name}")
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
        
        logger.info(f"Starting RandomForest training from CSV: {data_path} (low_end={low_end}, profile={profile_name})")
        
        if not os.path.exists(data_path):
            logger.error(f"CSV data file not found: {data_path}")
            raise FileNotFoundError(f"CSV data file not found: {data_path}")

        try:
            data = pd.read_csv(data_path, header=None, on_bad_lines="skip")
            data = data.dropna().reset_index(drop=True)
            logger.debug(f"CSV loaded: shape={data.shape}")
            
            x_values = data.iloc[:, :-1].values
            y_values = data.iloc[:, -1].values
            logger.info(f"Data extracted: X shape={x_values.shape}, y shape={y_values.shape}")

            return self._train_random_forest(
                x_values=x_values,
                y_values=y_values,
                save_model=save_model,
                low_end=low_end,
                profile_name=profile_name,
                source=source,
            )
        except Exception as e:
            logger.error(f"Failed to train RandomForest from CSV: {e}", exc_info=True)
            raise

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
        logger.info(f"_train_random_forest: X shape={x_values.shape}, y shape={y_values.shape}, save_model={save_model}")
        
        try:
            profile = self.get_profile(profile_name)
            if low_end and profile_name == "full":
                n_estimators = 100
                max_depth = 18
                n_jobs = 1
                logger.info("Using low_end configuration for RandomForest")
            else:
                n_estimators = int(profile["rf_estimators"])
                max_depth = int(profile["rf_max_depth"])
                n_jobs = int(profile["rf_n_jobs"])
            
            logger.debug(f"RandomForest hyperparams: n_estimators={n_estimators}, max_depth={max_depth}, n_jobs={n_jobs}")

            stratify = y_values if len(np.unique(y_values)) > 1 else None
            logger.info(f"Splitting data: unique_classes={len(np.unique(y_values))}, stratify={stratify is not None}")
            
            x_train, x_test, y_train, y_test = train_test_split(
                x_values,
                y_values,
                test_size=0.2,
                random_state=42,
                stratify=stratify,
            )
            logger.debug(f"Split completed: train_shape={x_train.shape}, test_shape={x_test.shape}")

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

            train_acc = clf.score(x_train, y_train)
            test_acc = clf.score(x_test, y_test)
            y_pred = clf.predict(x_test)
            acc = accuracy_score(y_test, y_pred)
            
            logger.info(f"RandomForest scores: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}, accuracy={acc:.4f}")

            labels_to_save = clf.classes_
            if class_labels is not None:
                labels_to_save = class_labels[clf.classes_.astype(int)]

            if save_model:
                model_path = os.path.join(MODELS_DIR, "hand_alphabet_model.pkl")
                labels_path = os.path.join(MODELS_DIR, "class_labels.npy")
                logger.info(f"Saving model to {model_path}")
                joblib.dump(clf, model_path)
                logger.info(f"Saving labels to {labels_path}")
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
            self.models["random_forest"] = clf
            self.labels["random_forest"] = labels_to_save
            self.last_metrics["random_forest"] = metrics
            logger.info(f"RandomForest training completed: {len(labels_to_save)} classes, {metrics['samples']} samples")
            return metrics
        except Exception as e:
            logger.error(f"RandomForest training failed: {e}", exc_info=True)
            raise

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
        
        logger.info(f"Starting WLASL video processing: json={json_file}, video_folder={video_folder}")
        logger.info(f"Config: max_classes={max_classes}, max_videos_per_class={max_videos_per_class}, seq_len={sequence_length}, stride={frame_stride}")
        
        if not os.path.exists(json_file):
            logger.error(f"JSON file not found: {json_file}")
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        if not os.path.exists(video_folder):
            logger.error(f"Video folder not found: {video_folder}")
            raise FileNotFoundError(f"Video folder not found: {video_folder}")

        try:
            with open(json_file, encoding="utf-8") as file_obj:
                data = json.load(file_obj)
            logger.info(f"Loaded WLASL JSON with {len(data)} classes")

            x_values = []
            y_values = []
            labels = []
            processed_videos = 0
            missing_videos = 0

            for label_index, item in enumerate(data[:max_classes]):
                gloss = item["gloss"]
                labels.append(gloss)
                logger.debug(f"Processing class {label_index+1}/{min(max_classes, len(data))}: {gloss}")
                
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

            logger.info(f"Video processing completed: {processed_videos} processed, {missing_videos} missing")
            
            x_array = np.array(x_values)
            y_array = np.array(y_values)
            logger.info(f"Data arrays created: X shape={x_array.shape}, y shape={y_array.shape}")
            
            if save_data:
                x_path = os.path.join(DATA_DIR, "X_data.npy")
                y_path = os.path.join(DATA_DIR, "y_data.npy")
                labels_path = os.path.join(MODELS_DIR, "wlasl_labels.npy")
                
                logger.info(f"Saving data: X to {x_path}, y to {y_path}, labels to {labels_path}")
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
                logger.info("Shared state updated with WLASL preprocessing metadata")

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
            logger.info(f"WLASL preprocessing completed: {self.last_preprocess_summary}")
            return x_array, y_array
        except Exception as e:
            logger.error(f"WLASL video processing failed: {e}", exc_info=True)
            raise

    def train_lstm(
        self,
        x_values: Optional[np.ndarray] = None,
        y_values: Optional[np.ndarray] = None,
        save_model: bool = True,
        low_end: bool = False,
        source: str = "training_module_lstm",
    ):
        logger.info(f"Starting LSTM training: x_values={x_values is not None}, y_values={y_values is not None}, save_model={save_model}, low_end={low_end}")
        
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available - cannot train LSTM")
            raise RuntimeError("TensorFlow not available. Cannot train LSTM model.")
        
        try:
            if x_values is None or y_values is None:
                x_path = os.path.join(DATA_DIR, "X_data.npy")
                y_path = os.path.join(DATA_DIR, "y_data.npy")
                logger.info(f"Loading preprocessed data: X from {x_path}, y from {y_path}")
                
                if not os.path.exists(x_path) or not os.path.exists(y_path):
                    logger.error(f"Preprocessed data not found. X exists={os.path.exists(x_path)}, y exists={os.path.exists(y_path)}")
                    raise FileNotFoundError("Processed data not found. Run preprocessing first.")
                
                x_values = np.load(x_path)
                y_values = np.load(y_path)
                logger.info(f"Data loaded: X shape={x_values.shape}, y shape={y_values.shape}")

            _, y_encoded = np.unique(y_values, return_inverse=True)
            y_encoded = y_encoded.astype(np.int32)
            num_classes = int(np.max(y_encoded)) + 1
            y_cat = to_categorical(y_encoded, num_classes)
            min_samples_per_class = min(np.bincount(y_encoded))
            use_stratify = min_samples_per_class >= 2
            
            logger.info(f"Data encoding: num_classes={num_classes}, min_samples_per_class={min_samples_per_class}, use_stratify={use_stratify}")

            if use_stratify:
                x_train, x_test, y_train, y_test = train_test_split(
                    x_values, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
                )
            else:
                x_train, x_test, y_train, y_test = train_test_split(
                    x_values, y_cat, test_size=0.2, random_state=42
                )
            
            logger.debug(f"Train/test split: X_train={x_train.shape}, X_test={x_test.shape}")

            if low_end:
                lstm_units = [64, 32, 16]
                dense_units = 32
                epochs = 30
                logger.info("Using low_end configuration for LSTM")
            else:
                lstm_units = [128, 64, 32]
                dense_units = 64
                epochs = 50

            logger.info(f"Building LSTM model: lstm_units={lstm_units}, dense_units={dense_units}, epochs={epochs}")
            
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
            logger.info("LSTM model compiled successfully")

            early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
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
            logger.info(f"LSTM training completed: epochs_ran={epochs_ran}")
            
            logger.info("Evaluating LSTM on test set...")
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            logger.info(f"LSTM evaluation: test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

            if save_model:
                model_path = os.path.join(MODELS_DIR, "gesture_model.h5")
                logger.info(f"Saving LSTM model to {model_path}")
                model.save(model_path)
                logger.info(f"Model saved successfully (size={os.path.getsize(model_path)} bytes)")
                
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
                logger.info("Shared state updated with LSTM model metadata")

            self.models["lstm"] = model
            self.last_metrics["lstm"] = {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_acc),
                "epochs_ran": int(epochs_ran),
                "samples": int(x_values.shape[0]),
                "classes": int(num_classes),
            }
            logger.info(f"LSTM training completed: metrics={self.last_metrics['lstm']}")
            return model
        except Exception as e:
            logger.error(f"LSTM training failed: {e}", exc_info=True)
            raise

    def evaluate_random_forest(self, data_path: Optional[str] = None, model_path: Optional[str] = None) -> float:
        if data_path is None:
            data_path = os.path.join(DATA_DIR, "hand_alphabet_data.csv")
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "hand_alphabet_model.pkl")
        
        logger.info(f"Evaluating RandomForest: data={data_path}, model={model_path}")
        
        try:
            data = pd.read_csv(data_path, header=None, on_bad_lines="skip").dropna()
            x_values = data.iloc[:, :-1].values
            y_values = data.iloc[:, -1].values
            logger.debug(f"Data loaded for evaluation: shape={x_values.shape}")
            
            clf = joblib.load(model_path)
            logger.debug(f"Model loaded from {model_path}")
            
            y_pred = clf.predict(x_values)
            acc = accuracy_score(y_values, y_pred)
            
            self.last_metrics["random_forest_eval"] = {
                "accuracy": float(acc),
                "samples": int(len(x_values)),
                "model_path": model_path,
            }
            logger.info(f"RandomForest evaluation completed: accuracy={acc:.4f}")
            return acc
        except Exception as e:
            logger.error(f"RandomForest evaluation failed: {e}", exc_info=True)
            raise

    def package_artifacts(self, profile_name: str = "pi_zero", note: str = "") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_id = f"local_{profile_name}_{timestamp}"
        package_dir = os.path.join(MODELS_DIR, "packages")
        os.makedirs(package_dir, exist_ok=True)

        logger.info(f"Packaging artifacts: package_id={package_id}, profile={profile_name}")

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
                logger.debug(f"Copied {os.path.basename(src)} to {dst}")

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
        logger.debug(f"Metadata saved to {metadata_path}")

        report_path = os.path.join(REPORTS_DIR, f"local_training_{timestamp}.json")
        with open(report_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, indent=2)
        logger.info(f"Package artifacts completed: {len(copied_files)} files packaged, report saved to {report_path}")

        return metadata_path

    @staticmethod
    def _sha256_file(file_path: str) -> str:
        digest = hashlib.sha256()
        with open(file_path, "rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def export_training_data(
        self,
        output_dir: Optional[str] = None,
        archive_prefix: str = "training_data_export",
        include_videos: bool = True,
        include_hashes: bool = False,
    ) -> Dict[str, object]:
        if output_dir is None:
            output_dir = os.path.join(REPORTS_DIR, "exports")
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Exporting training data: output_dir={output_dir}, prefix={archive_prefix}, include_videos={include_videos}, include_hashes={include_hashes}")

        data_root = DATA_DIR
        if not os.path.isdir(data_root):
            logger.error(f"Data directory not found: {data_root}")
            raise FileNotFoundError(f"Data directory not found: {data_root}")

        if include_videos:
            videos_dir = os.path.join(data_root, "videos")
            if not os.path.isdir(videos_dir):
                logger.error(f"Videos directory not found but include_videos=True: {videos_dir}")
                raise FileNotFoundError(
                    f"Video folder not found but include_videos is enabled: {videos_dir}"
                )

        expected_files = [
            "WLASL_v0.3.json",
            "hand_alphabet_data.csv",
            "X_data.npy",
            "y_data.npy",
            "wlasl_labels.npy",
        ]
        missing_expected = [
            expected
            for expected in expected_files
            if not os.path.exists(os.path.join(data_root, expected))
        ]

        selected_files: List[str] = []
        total_bytes = 0
        video_bytes = 0
        for root, _, files in os.walk(data_root):
            for file_name in files:
                abs_path = os.path.join(root, file_name)
                rel_path = os.path.relpath(abs_path, data_root)
                if not include_videos and rel_path.startswith(f"videos{os.sep}"):
                    continue

                file_size = os.path.getsize(abs_path)
                total_bytes += file_size
                if rel_path.startswith(f"videos{os.sep}"):
                    video_bytes += file_size
                selected_files.append(abs_path)

        if not selected_files:
            logger.error("No files selected for export")
            raise ValueError("No files selected for export from data directory")

        logger.info(f"Files collected for export: {len(selected_files)} files, total={total_bytes} bytes, video={video_bytes} bytes")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_id = f"{archive_prefix}_{timestamp}"
        zip_path = os.path.join(output_dir, f"{export_id}.zip")
        manifest_path = os.path.join(output_dir, f"{export_id}_manifest.json")

        warnings: List[str] = []
        if include_videos and total_bytes > 0 and (video_bytes / total_bytes) > 0.8:
            warnings.append("Videos account for more than 80% of total export size")
            logger.warning("Videos account for >80% of export size")
        if missing_expected:
            warnings.append(
                "Some expected training files are missing: " + ", ".join(missing_expected)
            )
            logger.warning(f"Missing expected files: {missing_expected}")

        with tempfile.TemporaryDirectory(prefix="training_export_") as staging_dir:
            staged_root = os.path.join(staging_dir, "data")
            os.makedirs(staged_root, exist_ok=True)

            file_entries: List[Dict[str, object]] = []
            for src in selected_files:
                rel_path = os.path.relpath(src, data_root)
                dst = os.path.join(staged_root, rel_path)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

                entry: Dict[str, object] = {
                    "relative_path": rel_path.replace("\\", "/"),
                    "size_bytes": int(os.path.getsize(src)),
                }
                if include_hashes:
                    entry["sha256"] = self._sha256_file(src)
                file_entries.append(entry)

            manifest = {
                "export_id": export_id,
                "created_at": datetime.now().isoformat(),
                "source_data_dir": data_root,
                "include_videos": include_videos,
                "include_hashes": include_hashes,
                "file_count": int(len(file_entries)),
                "total_bytes": int(total_bytes),
                "video_bytes": int(video_bytes),
                "warnings": warnings,
                "missing_expected_files": missing_expected,
                "files": file_entries,
            }

            staged_manifest_path = os.path.join(staged_root, "export_manifest.json")
            with open(staged_manifest_path, "w", encoding="utf-8") as file_obj:
                json.dump(manifest, file_obj, indent=2)
            with open(manifest_path, "w", encoding="utf-8") as file_obj:
                json.dump(manifest, file_obj, indent=2)

            logger.info(f"Creating archive: {zip_path}")
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for root, _, files in os.walk(staged_root):
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        archive_path = os.path.relpath(file_path, staging_dir).replace("\\", "/")
                        archive.write(file_path, archive_path)

        logger.info(f"Export completed: archive={zip_path}, manifest={manifest_path}")

        return {
            "archive_path": zip_path,
            "manifest_path": manifest_path,
            "file_count": int(len(selected_files)),
            "total_bytes": int(total_bytes),
            "video_bytes": int(video_bytes),
            "include_videos": include_videos,
            "include_hashes": include_hashes,
            "warnings": warnings,
        }

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
        logger.info(f"Starting device pipeline: profile={profile_name}, note={note}")
        
        try:
            profile = self.get_profile(profile_name)
            
            logger.info("Step 1: Processing WLASL videos...")
            self.process_wlasl_videos(
                json_file=json_file,
                video_folder=video_folder,
                save_data=True,
                max_classes=max_classes if max_classes is not None else int(profile["max_classes"]),
                max_videos_per_class=max_videos_per_class if max_videos_per_class is not None else int(profile["max_videos_per_class"]),
                sequence_length=sequence_length if sequence_length is not None else int(profile["sequence_length"]),
                frame_stride=frame_stride if frame_stride is not None else int(profile["frame_stride"]),
            )
            logger.info("Step 1 completed")
            
            logger.info("Step 2: Training RandomForest...")
            rf_metrics = self.train_random_forest_from_csv(
                data_path=csv_path,
                save_model=True,
                low_end=profile_name == "pi_zero",
                profile_name=profile_name,
                source="training_module_device_pipeline",
            )
            logger.info(f"Step 2 completed: {rf_metrics}")
            
            logger.info("Step 3: Evaluating RandomForest...")
            eval_acc = self.evaluate_random_forest(data_path=csv_path)
            logger.info(f"Step 3 completed: eval_acc={eval_acc:.4f}")
            
            logger.info("Step 4: Packaging artifacts...")
            package_path = self.package_artifacts(profile_name=profile_name, note=note)
            logger.info(f"Step 4 completed: package_path={package_path}")
            
            result = {
                "random_forest": rf_metrics,
                "evaluation_accuracy": eval_acc,
                "package_path": package_path,
                "preprocess_summary": self.last_preprocess_summary,
            }
            logger.info(f"Device pipeline completed: {result}")
            return result
        except Exception as e:
            logger.error(f"Device pipeline failed: {e}", exc_info=True)
            raise

    def train_all_models(self, profile_name: str = "full") -> Dict[str, object]:
        logger.info(f"Starting train_all_models: profile={profile_name}")
        
        try:
            profile = self.get_profile(profile_name)
            
            logger.info("Step 1: Training RandomForest...")
            rf_metrics = self.train_random_forest_from_csv(
                save_model=True,
                low_end=profile_name == "pi_zero",
                profile_name=profile_name,
                source="training_module_train_all",
            )
            logger.info(f"Step 1 completed: {rf_metrics}")
            
            logger.info("Step 2: Processing WLASL videos...")
            x_values, y_values = self.process_wlasl_videos(
                save_data=True,
                max_classes=int(profile["max_classes"]),
                max_videos_per_class=int(profile["max_videos_per_class"]),
                sequence_length=int(profile["sequence_length"]),
                frame_stride=int(profile["frame_stride"]),
            )
            logger.info(f"Step 2 completed: X shape={x_values.shape}, y shape={y_values.shape}")
            
            logger.info("Step 3: Training LSTM...")
            self.train_lstm(
                x_values=x_values,
                y_values=y_values,
                save_model=True,
                low_end=bool(profile.get("lstm_low_end", False)),
                source="training_module_train_all",
            )
            logger.info(f"Step 3 completed: {self.last_metrics.get('lstm', {})}")
            
            result = {
                "random_forest": rf_metrics,
                "lstm": self.last_metrics.get("lstm", {}),
                "preprocess_summary": self.last_preprocess_summary,
            }
            logger.info(f"train_all_models completed: {result}")
            return result
        except Exception as e:
            logger.error(f"train_all_models failed: {e}", exc_info=True)
            raise
