import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time

# Load trained model
model = tf.keras.models.load_model("sign_lang_video_model.h5")

# Example labels (replace with your actual label list)
label_map = ["book", "apple", "hello", "thanks", "water"]

IMG_SIZE = 64
SEQUENCE_LENGTH = 20
frame_window = deque(maxlen=SEQUENCE_LENGTH)

st.title("🧠 Live Sign Language Recognition")
st.write("This demo uses your webcam to predict signs in real time using a CNN+LSTM model.")

start_camera = st.button("Start Webcam")

if start_camera:
    run = st.empty()
    result_text = st.empty()
    
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        st.error("Could not open webcam.")
    else:
        st.success("Webcam started. Show a sign for prediction.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess frame
            frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            gray = gray.astype("float32") / 255.0
            frame_window.append(gray)

            # Show webcam feed in UI
            frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            run.image(frame_display, channels="RGB", use_column_width=True)

            # Predict if we have enough frames
            if len(frame_window) == SEQUENCE_LENGTH:
                input_frames = np.array(frame_window).reshape(1, SEQUENCE_LENGTH, IMG_SIZE, IMG_SIZE, 1)
                prediction = model.predict(input_frames)
                pred_class = label_map[np.argmax(prediction)]

                result_text.markdown(f"### 🤖 Prediction: **{pred_class}**")
                time.sleep(1.0)  # small delay for prediction visibility

        cap.release()
        cv2.destroyAllWindows()
