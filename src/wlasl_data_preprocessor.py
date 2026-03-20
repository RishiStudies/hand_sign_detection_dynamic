import cv2
import mediapipe as mp
import numpy as np
import json
import os
from tqdm import tqdm

try:
    from .shared_artifacts import DATA_DIR, MODELS_DIR, update_shared_state
except ImportError:
    from shared_artifacts import DATA_DIR, MODELS_DIR, update_shared_state

VIDEO_FOLDER = os.path.join(DATA_DIR, "videos")
JSON_FILE = os.path.join(DATA_DIR, "nslt_100.json")
SEQUENCE_LENGTH = 30

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)

with open(JSON_FILE) as f:
    data = json.load(f)

X = []
y = []
labels = []

for label_index, key in enumerate(tqdm(data)):
    item = data[key]
    word = item["word"]
    labels.append(word)
    for instance in item["instances"]:
        video_id = instance["video_id"]
        video_path = os.path.join(VIDEO_FOLDER, video_id + ".mp4")
        if not os.path.exists(video_path):
            continue
        cap = cv2.VideoCapture(video_path)
        sequence = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                row = []
                for lm in hand_landmarks.landmark:
                    row += [lm.x, lm.y, lm.z]
                sequence.append(row)
        cap.release()
        if len(sequence) >= SEQUENCE_LENGTH:
            sequence = sequence[:SEQUENCE_LENGTH]
            X.append(sequence)
            y.append(label_index)

X = np.array(X)
y = np.array(y)

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
        "sequence_length": SEQUENCE_LENGTH,
    },
    publisher="wlasl_data_preprocessor",
)
update_shared_state(
    "lstm",
    {"labels_path": labels_path},
    publisher="wlasl_data_preprocessor",
)

print("Dataset prepared!")
print("X shape:", X.shape)
print("y shape:", y.shape)
