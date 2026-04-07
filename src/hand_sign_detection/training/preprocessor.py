"""WLASL video preprocessing for LSTM training.

Extracts hand landmark sequences from WLASL dataset videos.
Supports parallel processing for faster feature extraction.
"""

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import cv2
import numpy as np

from hand_sign_detection.core.config import get_settings
from hand_sign_detection.core.shared_state import update_shared_state
from hand_sign_detection.models.features import extract_features_from_frame

logger = logging.getLogger("hand_sign_detection.training.preprocessor")


class WlaslPreprocessor:
    """Preprocessor for WLASL dataset videos.

    Extracts hand landmark sequences from videos for LSTM training.

    Usage:
        preprocessor = WlaslPreprocessor()
        x_data, y_data = preprocessor.process_videos(
            max_classes=10,
            max_videos_per_class=5,
        )
    """

    def __init__(self):
        self._settings = get_settings()
        self.last_summary: dict = {}

    def process_videos(
        self,
        json_file: str | None = None,
        video_folder: str | None = None,
        save_data: bool = True,
        max_classes: int = 5,
        max_videos_per_class: int = 3,
        sequence_length: int = 30,
        frame_stride: int = 1,
        parallel: bool = True,
        n_workers: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process WLASL videos and extract sequences.

        Args:
            json_file: Path to WLASL JSON file
            video_folder: Path to video folder
            save_data: Whether to save processed data
            max_classes: Maximum number of classes to process
            max_videos_per_class: Maximum videos per class
            sequence_length: Number of frames per sequence
            frame_stride: Frame sampling stride
            parallel: Use parallel processing (default True)
            n_workers: Number of worker processes (default: CPU count)

        Returns:
            Tuple of (X, y) numpy arrays
        """
        settings = self._settings

        if json_file is None:
            json_file = os.path.join(settings.data_dir, "WLASL_v0.3.json")
        if video_folder is None:
            video_folder = os.path.join(settings.data_dir, "videos")

        logger.info(
            "Starting WLASL video processing: json=%s, video_folder=%s",
            json_file,
            video_folder,
        )
        logger.info(
            "Config: max_classes=%d, max_videos_per_class=%d, seq_len=%d, stride=%d, parallel=%s",
            max_classes,
            max_videos_per_class,
            sequence_length,
            frame_stride,
            parallel,
        )

        if not os.path.exists(json_file):
            logger.error("JSON file not found: %s", json_file)
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        if not os.path.exists(video_folder):
            logger.error("Video folder not found: %s", video_folder)
            raise FileNotFoundError(f"Video folder not found: {video_folder}")

        with open(json_file, encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        logger.info("Loaded WLASL JSON with %d classes", len(data))

        # Collect all video tasks
        video_tasks = []
        labels = []

        for label_index, item in enumerate(data[:max_classes]):
            gloss = item["gloss"]
            labels.append(gloss)

            for instance in item["instances"][:max_videos_per_class]:
                video_id = instance["video_id"]
                video_path = os.path.join(video_folder, video_id + ".mp4")

                if os.path.exists(video_path):
                    video_tasks.append((video_path, label_index, sequence_length, frame_stride))

        logger.info("Found %d videos to process", len(video_tasks))

        # Process videos (parallel or sequential)
        if parallel and len(video_tasks) > 1:
            x_values, y_values, processed_videos, missing_videos = self._process_parallel(
                video_tasks, n_workers
            )
        else:
            x_values, y_values, processed_videos, missing_videos = self._process_sequential(
                video_tasks
            )

        logger.info(
            "Video processing completed: %d processed, %d missing",
            processed_videos,
            missing_videos,
        )

        x_array = np.array(x_values)
        y_array = np.array(y_values)
        logger.info(
            "Data arrays created: X shape=%s, y shape=%s",
            x_array.shape,
            y_array.shape,
        )

        if save_data:
            self._save_data(x_array, y_array, labels, sequence_length, frame_stride)

        self.last_summary = {
            "sequences": int(len(x_array)),
            "sequence_length": int(sequence_length),
            "feature_dims": int(x_array.shape[2]) if len(x_array.shape) > 2 else 0,
            "processed_videos": int(processed_videos),
            "missing_videos": int(missing_videos),
            "max_classes": int(max_classes),
            "max_videos_per_class": int(max_videos_per_class),
            "frame_stride": int(frame_stride),
        }
        logger.info("WLASL preprocessing completed: %s", self.last_summary)

        return x_array, y_array

    def _process_sequential(
        self,
        video_tasks: list[tuple[str, int, int, int]],
    ) -> tuple[list, list, int, int]:
        """Process videos sequentially.

        Args:
            video_tasks: List of (video_path, label_index, seq_length, stride) tuples

        Returns:
            Tuple of (x_values, y_values, processed_count, missing_count)
        """
        x_values = []
        y_values = []
        processed = 0

        for video_path, label_index, seq_length, stride in video_tasks:
            sequence = self._extract_sequence(video_path, seq_length, stride)
            if sequence is not None:
                x_values.append(sequence)
                y_values.append(label_index)
                processed += 1

        return x_values, y_values, processed, len(video_tasks) - processed

    def _process_parallel(
        self,
        video_tasks: list[tuple[str, int, int, int]],
        n_workers: int | None = None,
    ) -> tuple[list, list, int, int]:
        """Process videos in parallel using ProcessPoolExecutor.

        Args:
            video_tasks: List of (video_path, label_index, seq_length, stride) tuples
            n_workers: Number of worker processes (default: CPU count - 1)

        Returns:
            Tuple of (x_values, y_values, processed_count, missing_count)
        """
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)

        logger.info("Processing %d videos with %d workers", len(video_tasks), n_workers)

        x_values = []
        y_values = []
        processed = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    _extract_sequence_worker,
                    video_path,
                    seq_length,
                    stride
                ): (video_path, label_index)
                for video_path, label_index, seq_length, stride in video_tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_task):
                video_path, label_index = future_to_task[future]
                try:
                    sequence = future.result()
                    if sequence is not None:
                        x_values.append(sequence)
                        y_values.append(label_index)
                        processed += 1
                    else:
                        failed += 1
                except (OSError, cv2.error) as exc:
                    logger.warning("Video processing failed %s: %s", video_path, exc)
                    failed += 1

        logger.info("Parallel processing complete: %d processed, %d failed", processed, failed)
        return x_values, y_values, processed, failed

    def _extract_sequence(
        self,
        video_path: str,
        sequence_length: int,
        frame_stride: int,
    ) -> list | None:
        """Extract feature sequence from a video."""
        cap = cv2.VideoCapture(video_path)
        sequence = []
        frame_index = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_index += 1
                if frame_stride > 1 and frame_index % frame_stride != 0:
                    continue

                features = extract_features_from_frame(frame)
                sequence.append(features)

                if len(sequence) >= sequence_length:
                    break
        finally:
            cap.release()

        if len(sequence) >= sequence_length:
            sequence = sequence[:sequence_length]
            # Pad if needed
            while len(sequence) < sequence_length:
                sequence.append(np.zeros_like(sequence[0]))
            return sequence

        return None

    def _save_data(
        self,
        x_array: np.ndarray,
        y_array: np.ndarray,
        labels: list,
        sequence_length: int,
        frame_stride: int,
    ) -> None:
        """Save processed data to disk."""
        settings = self._settings

        x_path = os.path.join(settings.data_dir, "X_data.npy")
        y_path = os.path.join(settings.data_dir, "y_data.npy")
        labels_path = os.path.join(settings.models_dir, "wlasl_labels.npy")

        os.makedirs(settings.data_dir, exist_ok=True)
        os.makedirs(settings.models_dir, exist_ok=True)

        logger.info(
            "Saving data: X to %s, y to %s, labels to %s",
            x_path,
            y_path,
            labels_path,
        )

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
                "feature_schema": settings.feature_schema,
                "feature_schema_version": settings.feature_schema_version,
                "feature_dimension": int(x_array.shape[2]) if len(x_array.shape) > 2 else 0,
            },
            publisher="WlaslPreprocessor",
        )

        update_shared_state(
            "lstm",
            {
                "labels_path": labels_path,
                "feature_schema": settings.feature_schema,
                "feature_schema_version": settings.feature_schema_version,
                "feature_dimension": int(x_array.shape[2]) if len(x_array.shape) > 2 else 0,
            },
            publisher="WlaslPreprocessor",
        )

        logger.info("Shared state updated with WLASL preprocessing metadata")


def _extract_sequence_worker(
    video_path: str,
    sequence_length: int,
    frame_stride: int,
) -> list | None:
    """Worker function for parallel video processing.

    This is a module-level function for multiprocessing compatibility.

    Args:
        video_path: Path to video file
        sequence_length: Target sequence length
        frame_stride: Frame sampling stride

    Returns:
        Feature sequence or None if extraction failed
    """
    cap = cv2.VideoCapture(video_path)
    sequence = []
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            if frame_stride > 1 and frame_index % frame_stride != 0:
                continue

            features = extract_features_from_frame(frame)
            sequence.append(features)

            if len(sequence) >= sequence_length:
                break
    finally:
        cap.release()

    if len(sequence) >= sequence_length:
        sequence = sequence[:sequence_length]
        # Pad if needed
        while len(sequence) < sequence_length:
            sequence.append(np.zeros_like(sequence[0]))
        return sequence

    return None


def process_wlasl_videos(**kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Convenience function for processing WLASL videos.

    Args:
        **kwargs: Arguments passed to WlaslPreprocessor.process_videos()

    Returns:
        Tuple of (X, y) numpy arrays
    """
    preprocessor = WlaslPreprocessor()
    return preprocessor.process_videos(**kwargs)
