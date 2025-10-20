# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import joblib
import numpy as np
import cv2
from streamlit.components.v1 import html
import time
import threading
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# === Configuration ===
IMG_SIZE = (224, 224)
MODEL_PATH = "knn_model.pkl"
COOLDOWN = 5  # seconds

# === Load KNN model ===
model = joblib.load(MODEL_PATH)

# === Cache MobileNetV2 feature extractor ===
@st.cache_resource
def load_feature_extractor():
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

feature_extractor = load_feature_extractor()

# === Streamlit UI ===
st.title("🔍 Guiding Eye – Live Object Recognition")
st.write(f"Detecting objects from your webcam live (prediction every {COOLDOWN} sec)")

# Session state to store last prediction and timestamp
if "last_pred" not in st.session_state:
    st.session_state.last_pred = ""
if "last_time" not in st.session_state:
    st.session_state.last_time = 0
if "pred_thread_running" not in st.session_state:
    st.session_state.pred_thread_running = False

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# === Prediction function (runs in background thread) ===
def predict_object(frame):
    try:
        small_frame = cv2.resize(frame, IMG_SIZE)
        small_frame = img_to_array(small_frame)
        small_frame = np.expand_dims(small_frame, axis=0)
        small_frame = preprocess_input(small_frame)
        features = feature_extractor.predict(small_frame)
        pred = model.predict([features[0]])[0]

        st.session_state.last_pred = pred
        st.session_state.last_time = time.time()
        st.session_state.pred_thread_running = False
    except Exception as e:
        print("Prediction error:", e)
        st.session_state.pred_thread_running = False

# === Video processor class ===
class ObjectDetectionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Start prediction thread if cooldown passed and thread is not running
        if (time.time() - st.session_state.last_time > COOLDOWN) and not st.session_state.pred_thread_running:
            st.session_state.pred_thread_running = True
            threading.Thread(target=predict_object, args=(img.copy(),), daemon=True).start()

        # Draw last prediction on frame
        if st.session_state.last_pred:
            cv2.putText(img, f"Detected: {st.session_state.last_pred}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

# === Start live webcam ===
webrtc_streamer(
    key="guiding-eye",
    video_processor_factory=ObjectDetectionProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# === Browser speech outside processor ===
if st.session_state.last_pred:
    html(f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{st.session_state.last_pred}");
        window.speechSynthesis.speak(msg);
    </script>
    """)
