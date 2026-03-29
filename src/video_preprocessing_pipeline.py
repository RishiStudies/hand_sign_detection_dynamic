"""
Video Preprocessing Pipeline for Hand Sign Detection
Converts raw videos to preprocessed sequences with live progress tracking
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import pickle

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


class VideoPreprocessingPipeline:
    """Pipeline to convert videos to LSTM-ready sequences"""
    
    def __init__(
        self,
        video_dir: str = "data/videos",
        output_dir: str = "data",
        sequence_length: int = 30,
        checkpoint_file: Optional[str] = None,
    ):
        """
        Initialize the preprocessing pipeline
        
        Args:
            video_dir: Path to folder containing video files
            output_dir: Where to save X_data.npy and y_data.npy
            sequence_length: Number of frames per video sequence
            checkpoint_file: Optional checkpoint to resume from
        """
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.sequence_length = sequence_length
        self.checkpoint_file = checkpoint_file
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe if available
        self.mp_hands = None
        self.hands = None
        if HAS_MEDIAPIPE:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
            )
        
        # State tracking
        self.processed_videos = []
        self.failed_videos = []
        self.sequences = []
        self.labels = []
        self.class_map = {}
        self.next_class_id = 0
        
        # Load checkpoint if provided
        if checkpoint_file and Path(checkpoint_file).exists():
            self._load_checkpoint(checkpoint_file)
    
    def _get_hand_landmarks(self, frame) -> Optional[np.ndarray]:
        """Extract hand landmarks from a frame using MediaPipe"""
        if self.hands is None:
            return None
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            # Combine landmarks from all detected hands
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            return np.array(landmarks, dtype=np.float32)
        return None
    
    def _extract_sequence_from_video(self, video_path: Path) -> Optional[List[np.ndarray]]:
        """Extract sequence of hand landmarks from a video"""
        if not HAS_CV2:
            return None
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < self.sequence_length:
                cap.release()
                return None
            
            # Sample frames evenly across the video
            frame_indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
            sequence = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    landmarks = self._get_hand_landmarks(frame)
                    if landmarks is not None:
                        sequence.append(landmarks)
                    else:
                        # If no hands detected, use zeros
                        sequence.append(np.zeros(63, dtype=np.float32))
                else:
                    sequence.append(np.zeros(63, dtype=np.float32))
            
            cap.release()
            
            if len(sequence) == self.sequence_length:
                return sequence
            return None
        
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return None
    
    def _get_label_from_path(self, video_path: Path) -> str:
        """Extract class label from video path (assumes parent folder is class)"""
        return video_path.parent.name
    
    def process_videos(
        self,
        progress_callback=None,
        max_videos: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Process all videos in the input directory
        
        Args:
            progress_callback: Optional function called with progress updates
            max_videos: Limit to first N videos (for testing)
        
        Returns:
            (X_data, y_data, metadata)
        """
        # Find all video files
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        video_files = [
            f for f in self.video_dir.rglob('*')
            if f.is_file() and f.suffix.lower() in video_extensions
        ]
        
        # Filter out already processed
        video_files = [
            f for f in video_files
            if str(f) not in self.processed_videos
        ]
        
        if max_videos:
            video_files = video_files[:max_videos]
        
        total = len(video_files)
        
        for idx, video_path in enumerate(video_files):
            # Get label from folder structure
            label_str = self._get_label_from_path(video_path)
            
            if label_str not in self.class_map:
                self.class_map[label_str] = self.next_class_id
                self.next_class_id += 1
            
            label_id = self.class_map[label_str]
            
            # Extract sequence
            sequence = self._extract_sequence_from_video(video_path)
            
            if sequence is not None:
                # Stack sequence into 2D array (sequence_length, num_features)
                sequence_array = np.stack(sequence, axis=0)
                self.sequences.append(sequence_array)
                self.labels.append(label_id)
                self.processed_videos.append(str(video_path))
            else:
                self.failed_videos.append(str(video_path))
            
            # Progress callback
            if progress_callback:
                progress_callback({
                    'current': idx + 1,
                    'total': total,
                    'processed': len(self.processed_videos),
                    'failed': len(self.failed_videos),
                    'current_file': video_path.name,
                    'class': label_str,
                })
        
        # Convert to numpy arrays
        if self.sequences:
            X_data = np.array(self.sequences, dtype=np.float32)
            y_data = np.array(self.labels, dtype=np.int64)
        else:
            X_data = np.array([], dtype=np.float32).reshape(0, self.sequence_length, 63)
            y_data = np.array([], dtype=np.int64)
        
        metadata = {
            'sequence_length': self.sequence_length,
            'num_features': 63,
            'total_videos': total,
            'processed': len(self.processed_videos),
            'failed': len(self.failed_videos),
            'class_map': self.class_map,
            'created_at': datetime.now().isoformat(),
        }
        
        return X_data, y_data, metadata
    
    def save_data(self, X_data: np.ndarray, y_data: np.ndarray, metadata: Dict):
        """Save preprocessed data to files"""
        x_path = self.output_dir / 'X_data.npy'
        y_path = self.output_dir / 'y_data.npy'
        metadata_path = self.output_dir / f'preprocessing_metadata_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        np.save(x_path, X_data)
        np.save(y_path, y_data)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f'Saved X_data: {x_path}')
        print(f'Saved y_data: {y_path}')
        print(f'Saved metadata: {metadata_path}')
        
        return x_path, y_path, metadata_path
    
    def save_checkpoint(self, checkpoint_path: Optional[str] = None):
        """Save pipeline state for resuming"""
        if checkpoint_path is None:
            checkpoint_path = self.output_dir / f'checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        
        state = {
            'processed_videos': self.processed_videos,
            'failed_videos': self.failed_videos,
            'sequences': self.sequences,
            'labels': self.labels,
            'class_map': self.class_map,
            'next_class_id': self.next_class_id,
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        
        print(f'Checkpoint saved: {checkpoint_path}')
        return checkpoint_path
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load pipeline state from checkpoint"""
        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)
        
        self.processed_videos = state['processed_videos']
        self.failed_videos = state['failed_videos']
        self.sequences = state['sequences']
        self.labels = state['labels']
        self.class_map = state['class_map']
        self.next_class_id = state['next_class_id']
        
        print(f'Checkpoint loaded: {checkpoint_path}')


def main():
    """Example usage"""
    pipeline = VideoPreprocessingPipeline(
        video_dir='data/videos',
        output_dir='data',
        sequence_length=30,
    )
    
    def progress(info):
        print(
            f"Processing: {info['current']}/{info['total']} | "
            f"Success: {info['processed']} | Failed: {info['failed']} | "
            f"Current: {info['current_file']}"
        )
    
    X_data, y_data, metadata = pipeline.process_videos(progress_callback=progress)
    
    print(f"\nPreprocessing complete!")
    print(f"X_data shape: {X_data.shape}")
    print(f"y_data shape: {y_data.shape}")
    print(f"Metadata: {metadata}")
    
    pipeline.save_data(X_data, y_data, metadata)


if __name__ == '__main__':
    main()
