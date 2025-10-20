# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import joblib
import numpy as np
import cv2
from streamlit.components.v1 import html
import time

# === Configuration ===
MODEL_PATH = "knn_model.pkl"

# Load KNN model (precomputed features)
model = joblib.load(MODEL_PATH)

# === Browser speech function ===
def speak_browser(text):
    js_code = f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{text}");
        window.speechSynthesis.speak(msg);
    </script>
    """
    html(js_code)

# === Streamlit UI ===
st.title("🔍 Guiding Eye – Live Object Recognition (KNN Only)")
st.write("Detecting objects from your webcam live in the browser using KNN.")

# Optional WebRTC config
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# === Video transformer class ===
class ObjectDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_spoken = ""
        self.last_time = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Resize & flatten image to match KNN input shape
        img_resized = cv2.resize(img, (224, 224))  # same as precomputed features
        img_flat = img_resized.flatten().reshape(1, -1)

        # Predict every 3 seconds
        if time.time() - self.last_time > 3:
            try:
                pred = model.predict(img_flat)[0]

                # Speak if new prediction
                if pred != self.last_spoken:
                    speak_browser(pred)
                    self.last_spoken = pred

                self.last_time = time.time()
            except Exception as e:
                print("Prediction error:", e)

        # Draw prediction on frame
        if self.last_spoken:
            cv2.putText(img, f"Detected: {self.last_spoken}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

# === Start live webcam streaming ===
webrtc_streamer(
    key="guiding-eye",
    video_transformer_factory=ObjectDetectionTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)
