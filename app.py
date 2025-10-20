# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import joblib
import numpy as np
import cv2
from streamlit.components.v1 import html
import time
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

# Use session state to store last prediction for speech
if "last_pred" not in st.session_state:
    st.session_state.last_pred = ""
if "last_time" not in st.session_state:
    st.session_state.last_time = 0

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# === Transformer ===
class ObjectDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Predict every COOLDOWN seconds
        if time.time() - st.session_state.last_time > COOLDOWN:
            try:
                # Extract features and predict
                image = cv2.resize(img, IMG_SIZE)
                image = img_to_array(image)
                image = np.expand_dims(image, axis=0)
                image = preprocess_input(image)
                features = feature_extractor.predict(image)
                pred = model.predict([features[0]])[0]

                # Update session state
                st.session_state.last_pred = pred
                st.session_state.last_time = time.time()

            except Exception as e:
                print("Prediction error:", e)

        # Draw last prediction on frame
        if st.session_state.last_pred:
            cv2.putText(img, f"Detected: {st.session_state.last_pred}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

# === Start live webcam ===
webrtc_streamer(
    key="guiding-eye",
    video_transformer_factory=ObjectDetectionTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

# === Trigger browser speech outside transformer ===
if st.session_state.last_pred:
    st.experimental_rerun()  # Ensure Streamlit updates JS
    html(f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{st.session_state.last_pred}");
        window.speechSynthesis.speak(msg);
    </script>
    """)
