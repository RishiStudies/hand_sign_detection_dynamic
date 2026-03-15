# Hand Sign Detection Model Training

This repository contains a comprehensive training script for hand sign detection models.

## Quick Start

1. **Activate your virtual environment:**
   ```bash
   # On Windows
   .\venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

2. **Run the training script:**
   ```bash
   python model_training_orchestrator.py
   ```

The script will automatically:
- Check for available datasets
- Train appropriate models (Random Forest for CSV data, LSTM for video data)
- Save trained models to the `models/` directory
- Generate a training report

## Supported Datasets

### 1. CSV Landmark Data (`data/hand_alphabet_data.csv`)
- **Model:** Random Forest Classifier
- **Features:** MediaPipe hand landmarks (x, y, z coordinates)
- **Labels:** Alphabet letters (A-Z)
- **Use case:** Static gesture recognition

### 2. WLASL Video Dataset (`data/WLASL_v0.3.json` + `data/videos/`)
- **Model:** LSTM Neural Network
- **Features:** Sequential hand landmarks from video frames
- **Labels:** Sign language glosses
- **Use case:** Dynamic gesture recognition

### 3. Preprocessed Data (`data/X_data.npy`, `data/y_data.npy`)
- **Format:** NumPy arrays ready for LSTM training
- **Generated from:** WLASL video processing

## Training Output

Models are saved to the `models/` directory:
- `hand_alphabet_model.pkl` - Random Forest model
- `class_labels.npy` - Random Forest class labels
- `gesture_model.h5` - LSTM model
- `wlasl_labels.npy` - LSTM class labels

Training reports are saved to the `reports/` directory.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `tensorflow` - For LSTM training
- `scikit-learn` - For Random Forest
- `opencv-python` - For video processing
- `mediapipe` - For hand landmark detection
- `pandas`, `numpy` - Data processing

## Advanced Usage

### Train Specific Models

```python
from model_training_orchestrator import HandSignModelTrainer

trainer = HandSignModelTrainer()

# Train only Random Forest
trainer.train_random_forest()

# Process WLASL videos and train LSTM
X, y = trainer.process_wlasl_videos()
trainer.train_lstm(X, y)
```

### Custom Parameters

The script includes sensible defaults but can be modified for:
- Different sequence lengths
- Custom model architectures
- Alternative feature extraction methods
- Different training hyperparameters

## Troubleshooting

### Common Issues

1. **"TensorFlow not available"**
   - Install TensorFlow: `pip install tensorflow`
   - LSTM training will be disabled

2. **"MediaPipe not available"**
   - Install MediaPipe: `pip install mediapipe`
   - Falls back to histogram features

3. **No training data found**
   - Ensure datasets are in the `data/` directory
   - Check file paths and permissions

4. **Memory errors during LSTM training**
   - Reduce batch size in the script
   - Process fewer videos per class

### Data Preparation

If you have custom data:

1. **For CSV data:** Format as landmark coordinates with labels in the last column
2. **For video data:** Organize in WLASL JSON format with video files in `data/videos/`

## Model Performance

Typical performance metrics:
- **Random Forest:** 90-95% accuracy on alphabet recognition
- **LSTM:** 70-85% accuracy on dynamic gestures (depends on dataset size)

Performance varies based on:
- Quality and quantity of training data
- Hand landmark detection accuracy
- Model hyperparameters
- Video quality and lighting conditions

## Integration

Trained models can be used with the FastAPI backend in `src/api_server.py` for real-time gesture recognition.