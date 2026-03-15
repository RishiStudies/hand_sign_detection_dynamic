import cv2
import numpy as np
import streamlit as st
import time
from textblob import TextBlob
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = joblib.load("../models/hand_alphabet_model.pkl")

labels = np.load("../models/class_labels.npy", allow_pickle=True)


def setup_ui():
    st.set_page_config(page_title="Hand Sign Detection", layout="wide")
    st.title("Real-Time Hand Sign Recognition")
    st.markdown(
        "This app classifies hand gestures using a pre-trained Random Forest model."
    )

    st.sidebar.header("Controls")
    run_flag = st.sidebar.checkbox("Run Webcam", value=True)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

    col1, col2 = st.columns([3, 1])

    with col1:
        frame_window = st.empty()

    with col2:
        st.subheader("Prediction")
        pred_text = st.empty()
        confidence_bar = st.progress(0)
        st.markdown("---")
        st.subheader("Sentence")
        sentence_display = st.empty()

    if "sentence" not in st.session_state:
        st.session_state.sentence = ""

    return (
        run_flag,
        confidence_threshold,
        frame_window,
        pred_text,
        confidence_bar,
        sentence_display,
    )


def run_app():
    (
        run,
        confidence_threshold,
        FRAME_WINDOW,
        pred_text,
        confidence_bar,
        sentence_display,
    ) = setup_ui()

    cap = cv2.VideoCapture(0)
    last_prediction_time = 0
    prediction_cooldown = 1.5

    def autocorrect_sentence():
        if st.session_state.sentence.strip():
            corrected = str(TextBlob(st.session_state.sentence.lower()).correct())
            st.session_state.sentence = corrected.capitalize()

    st.sidebar.button("?? Autocorrect", on_click=autocorrect_sentence)

    n_features = model.n_features_in_
    st.info(f"Model expects {n_features} features (hand landmarks)")

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("?? Cannot access webcam")
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        roi_top = int(h * 0.1)
        roi_left = int(w * 0.35)
        roi_height = int(h * 0.6)
        roi_width = int(w * 0.3)

        cv2.rectangle(
            frame,
            (roi_left, roi_top),
            (roi_left + roi_width, roi_top + roi_height),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Place hand here",
            (roi_left + 10, roi_top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        roi = frame[roi_top : roi_top + roi_height, roi_left : roi_left + roi_width]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        features = []

        moments = cv2.moments(gray, binaryImage=False)

        if moments["m00"] > 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx, cy = roi_width / 2, roi_height / 2

        hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-10)

        features = list(hist)

        if len(features) < n_features:
            features.extend([0.0] * (n_features - len(features)))
        else:
            features = features[:n_features]

        features = np.array(features).reshape(1, -1)

        label = "No gesture"
        prob = 0.0

        try:
            prediction_proba = model.predict_proba(features)[0]
            idx = np.argmax(prediction_proba)
            label = labels[idx]
            prob = prediction_proba[idx]

            current_time = time.time()

            if (
                prob > confidence_threshold
                and current_time - last_prediction_time > prediction_cooldown
            ):
                if label.lower() == "space":
                    st.session_state.sentence += " "
                elif label.lower() != "none" and label.lower() != "background":
                    st.session_state.sentence += label + " "
                last_prediction_time = current_time
        except Exception as e:
            st.sidebar.error(f"Prediction error: {e}")

        FRAME_WINDOW.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Live Feed", width="stretch"
        )

        pred_text.markdown(f"### ?? {label}")
        confidence_bar.progress(min(int(prob * 100), 100))

        sentence_display.markdown(f"### ?? {st.session_state.sentence}")

    cap.release()


def main():
    run_app()


if __name__ == "__main__":
    main()

st.sidebar.info(
    "**Note:** This is a demo version using image statistics. "
    "Install mediapipe on Linux/Mac for full hand landmark detection."
)
