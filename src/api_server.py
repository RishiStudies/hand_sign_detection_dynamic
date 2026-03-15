from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import List
import warnings
import time
from collections import deque

warnings.filterwarnings("ignore")


try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow not available. LSTM predictions will be disabled.")
    TENSORFLOW_AVAILABLE = False

app = FastAPI()

script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "../models")
frontend_dir = os.path.join(script_dir, "../frontend")


model = joblib.load(os.path.join(models_dir, "hand_alphabet_model.pkl"))
labels = np.load(os.path.join(models_dir, "class_labels.npy"), allow_pickle=True)
n_features = model.n_features_in_


lstm_model = None
lstm_labels = None
if TENSORFLOW_AVAILABLE:
    try:
        lstm_model_path = os.path.join(models_dir, "gesture_model.h5")
        if os.path.exists(lstm_model_path):
            lstm_model = load_model(lstm_model_path)
            lstm_labels = np.load(
                os.path.join(models_dir, "wlasl_labels.npy"), allow_pickle=True
            )
            print("✅ LSTM model loaded successfully")
        else:
            print("⚠️  LSTM model not found")
    except Exception as e:
        print(f"⚠️  Failed to load LSTM model: {e}")
        lstm_model = None


class ComboDetector:
    def __init__(self):

        self.combos = {
            "HELLO_WORLD": ["HELLO", "WORLD"],
            "THANK_YOU": ["THANK", "YOU"],
            "GOOD_MORNING": ["GOOD", "MORNING"],
            "HOW_ARE_YOU": ["HOW", "ARE", "YOU"],
            "I_LOVE_YOU": ["I", "LOVE", "YOU"],
            "PLEASE": ["PLEASE"],
            "SORRY": ["SORRY"],
            "YES_NO": ["YES", "NO"],
            "ABC": ["A", "B", "C"],
            "COUNTING": ["ONE", "TWO", "THREE"],
        }

        self.prediction_buffer = deque(maxlen=10)
        self.buffer_timeout = 5.0

    def add_prediction(self, gesture: str, confidence: float, model_type: str = "rf"):
        """Add a prediction to the buffer"""
        timestamp = time.time()
        self.prediction_buffer.append(
            {
                "gesture": gesture,
                "confidence": confidence,
                "timestamp": timestamp,
                "model": model_type,
            }
        )

    def check_combos(self, min_confidence: float = 0.7):
        """Check if any combo pattern is detected in recent predictions"""
        if len(self.prediction_buffer) < 2:
            return None

        current_time = time.time()
        recent_predictions = [
            p
            for p in self.prediction_buffer
            if current_time - p["timestamp"] <= self.buffer_timeout
            and p["confidence"] >= min_confidence
        ]

        if len(recent_predictions) < 2:
            return None

        gesture_sequence = [p["gesture"] for p in recent_predictions]

        for combo_name, combo_sequence in self.combos.items():
            if self._matches_combo(gesture_sequence, combo_sequence):

                combo_predictions = recent_predictions[-len(combo_sequence) :]
                avg_confidence = sum(p["confidence"] for p in combo_predictions) / len(
                    combo_predictions
                )

                return {
                    "combo": combo_name,
                    "sequence": combo_sequence,
                    "confidence": avg_confidence,
                    "timestamp": current_time,
                }

        return None

    def _matches_combo(
        self, gesture_sequence: List[str], combo_sequence: List[str]
    ) -> bool:
        """Check if the gesture sequence ends with the combo pattern"""
        if len(gesture_sequence) < len(combo_sequence):
            return False

        recent_gestures = gesture_sequence[-len(combo_sequence) :]
        return recent_gestures == combo_sequence

    def get_available_combos(self):
        """Return list of available combos"""
        return list(self.combos.keys())


combo_detector = ComboDetector()


def extract_features_from_frame(frame: np.ndarray) -> np.ndarray:
    """return feature vector for entire frame or ROI for prediction"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
    hist = hist.flatten() / (hist.sum() + 1e-10)
    features = list(hist)
    if len(features) < n_features:
        features.extend([0.0] * (n_features - len(features)))
    else:
        features = features[:n_features]
    return np.array(features).reshape(1, -1)


def extract_features_from_bytes(data: bytes) -> np.ndarray:
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return extract_features_from_frame(frame)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Receive an image file and return the predicted label/probability."""
    data = await file.read()
    features = extract_features_from_bytes(data)
    proba = model.predict_proba(features)[0]
    idx = int(np.argmax(proba))
    predicted_label = str(labels[idx])
    confidence = float(proba[idx])

    combo_detector.add_prediction(predicted_label, confidence, "rf")

    combo_result = combo_detector.check_combos()

    response = {"label": predicted_label, "prob": confidence}
    if combo_result:
        response["combo"] = combo_result

    return response


@app.post("/predict_sequence")
async def predict_sequence(files: List[UploadFile] = File(...)):
    """Receive a sequence of image files and return LSTM prediction."""
    if lstm_model is None:
        return {"error": "LSTM model not available"}

    if len(files) != 30:
        return {"error": f"Expected 30 frames, got {len(files)}"}

    sequence_features = []
    for file in files:
        data = await file.read()
        features = extract_features_from_bytes(data)
        sequence_features.append(features.flatten())

    X_sequence = np.array(sequence_features).reshape(1, 30, -1)

    predictions = lstm_model.predict(X_sequence, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])

    predicted_label = (
        str(lstm_labels[predicted_class])
        if lstm_labels is not None
        else str(predicted_class)
    )

    combo_detector.add_prediction(predicted_label, confidence, "lstm")

    combo_result = combo_detector.check_combos()

    response = {"label": predicted_label, "prob": confidence, "model": "lstm"}

    if combo_result:
        response["combo"] = combo_result

    return response


@app.get("/combos")
def get_combos():
    """Get list of available gesture combos"""
    return {
        "combos": combo_detector.get_available_combos(),
        "patterns": combo_detector.combos,
    }


@app.post("/clear_combos")
def clear_combo_history():
    """Clear the combo detection buffer"""
    combo_detector.prediction_buffer.clear()
    return {"status": "cleared"}


@app.get("/")
def index():
    html_path = os.path.join(frontend_dir, "detection_dashboard.html")
    html = open(html_path).read()
    return HTMLResponse(html)


@app.get("/training")
def training():
    html_path = os.path.join(frontend_dir, "training_dashboard.html")
    html = open(html_path).read()
    return HTMLResponse(html)


@app.post("/train")
async def train(samples: List[UploadFile], labels_input: List[str] = Form(...)):
    global model, labels, n_features

    X = []
    y = []
    label_to_idx = {}

    for sample, label in zip(samples, labels_input):
        if label not in label_to_idx:
            label_to_idx[label] = len(label_to_idx)

        data = await sample.read()
        features = extract_features_from_bytes(data)
        X.append(features.flatten())
        y.append(label_to_idx[label])

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X_train, y_train)

    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)

    model = clf
    labels = np.array(list(label_to_idx.keys()))
    n_features = clf.n_features_in_

    joblib.dump(clf, os.path.join(models_dir, "hand_alphabet_model.pkl"))
    np.save(os.path.join(models_dir, "class_labels.npy"), labels)

    return {
        "accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "samples": len(X),
        "gestures": len(label_to_idx),
    }


@app.post("/train_csv")
async def train_csv(file: UploadFile = File(...)):
    global model, labels, n_features

    data = await file.read()
    df = pd.read_csv(pd.io.common.BytesIO(data))

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)

    model = clf
    labels = clf.classes_
    n_features = clf.n_features_in_

    joblib.dump(clf, os.path.join(models_dir, "hand_alphabet_model.pkl"))
    np.save(os.path.join(models_dir, "class_labels.npy"), labels)

    return {
        "accuracy": float(accuracy),
        "samples": len(X),
        "gestures": len(np.unique(y)),
    }


@app.post("/process_wlasl")
async def process_wlasl():
    """Process WLASL dataset videos and extract features"""
    import subprocess
    import sys

    try:

        result = subprocess.run(
            [sys.executable, os.path.join(script_dir, "wlasl_data_preprocessor.py")],
            capture_output=True,
            text=True,
            cwd=script_dir,
        )

        if result.returncode == 0:
            return {
                "status": "success",
                "message": "WLASL dataset processed successfully",
            }
        else:
            return {"status": "error", "message": result.stderr}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/train_lstm")
async def train_lstm():
    """Train LSTM model on processed WLASL data"""
    import subprocess
    import sys

    try:

        result = subprocess.run(
            [sys.executable, os.path.join(script_dir, "lstm_trainer.py")],
            capture_output=True,
            text=True,
            cwd=script_dir,
        )

        if result.returncode == 0:
            return {"status": "success", "message": "LSTM model trained successfully"}
        else:
            return {"status": "error", "message": result.stderr}
    except Exception as e:
        return {"status": "error", "message": str(e)}


app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
