# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import joblib
import numpy as np
import cv2
import time
import queue
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_TTS import auto_play, text_to_audio

# === Configuration ===
IMG_SIZE = (224, 224)
MODEL_PATH = "knn_model.pkl"
COOLDOWN = 5  # seconds between predictions

# === Load KNN model ===
model = joblib.load(MODEL_PATH)

# === Cache MobileNetV2 feature extractor ===
@st.cache_resource
def load_feature_extractor():
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

feature_extractor = load_feature_extractor()

# === Initialize session state ===
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = ""
if "prediction_queue" not in st.session_state:
    st.session_state.prediction_queue = queue.Queue()

# === Python TTS function ===
def speak_text(text):
    """Convert text to speech and play automatically"""
    try:
        audio = text_to_audio(text, language='en')
        auto_play(audio, wait=True, lag=0.25)
    except Exception as e:
        st.error(f"TTS Error: {e}")

# === Extract features from frame ===
def extract_frame_feature(frame):
    image = cv2.resize(frame, IMG_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = feature_extractor.predict(image)
    return features[0]

# === Video transformer with queue ===
class ObjectDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_time = 0
        self.last_pred = ""
        self.prediction_queue = st.session_state.prediction_queue

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Predict only every COOLDOWN seconds
        if time.time() - self.last_time > COOLDOWN:
            try:
                feat = extract_frame_feature(img)
                pred = model.predict([feat])[0]

                # Update prediction if changed
                if pred != self.last_pred:
                    self.last_pred = pred
                    # Put new prediction in queue
                    self.prediction_queue.put(pred)

                self.last_time = time.time()
            except Exception as e:
                print("Prediction error:", e)

        # Draw prediction on frame
        if self.last_pred:
            cv2.putText(img, f"Detected: {self.last_pred}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

# === Streamlit UI ===
st.title("🔍 Guiding Eye – Live Object Recognition")
st.write(f"Detecting objects from your webcam live (prediction every {COOLDOWN} sec)")

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# === Start live webcam streaming ===
webrtc_ctx = webrtc_streamer(
    key="guiding-eye",
    video_transformer_factory=ObjectDetectionTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

# === Process predictions from queue and trigger TTS ===
if not st.session_state.prediction_queue.empty():
    try:
        new_pred = st.session_state.prediction_queue.get_nowait()
        if new_pred != st.session_state.last_prediction:
            st.session_state.last_prediction = new_pred
            # Speak the prediction
            speak_text(f"Detected {new_pred}")
            st.rerun()
    except queue.Empty:
        pass

# Display current prediction
if st.session_state.last_prediction:
    st.success(f"📢 Current detection: {st.session_state.last_prediction}")
