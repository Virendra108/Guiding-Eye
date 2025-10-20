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
COOLDOWN = 5  # seconds between predictions

# === Load KNN model ===
model = joblib.load(MODEL_PATH)

# === Cache MobileNetV2 feature extractor ===
@st.cache_resource
def load_feature_extractor():
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

feature_extractor = load_feature_extractor()

# === Browser speech function ===
def speak_browser(text):
    js_code = f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{text}");
        window.speechSynthesis.speak(msg);
    </script>
    """
    html(js_code)

# === Extract features from frame ===
def extract_frame_feature(frame):
    image = cv2.resize(frame, IMG_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = feature_extractor.predict(image)
    return features[0]

# === Streamlit UI ===
st.title("🔍 Guiding Eye – Live Object Recognition")
st.write(f"Detecting objects from your webcam live (prediction every {COOLDOWN} sec)")

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# === Initialize session state ===
if "last_pred" not in st.session_state:
    st.session_state.last_pred = ""
if "spoken_pred" not in st.session_state:
    st.session_state.spoken_pred = ""

# === Video transformer class ===
class ObjectDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_time = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Predict only every COOLDOWN seconds
        if time.time() - self.last_time > COOLDOWN:
            try:
                feat = extract_frame_feature(img)
                pred = model.predict([feat])[0]

                # Update session state if prediction changed
                if pred != st.session_state.last_pred:
                    st.session_state.last_pred = pred

                self.last_time = time.time()
            except Exception as e:
                print("Prediction error:", e)

        # Draw last prediction on frame
        if st.session_state.last_pred:
            cv2.putText(img, f"Detected: {st.session_state.last_pred}", (10, 30),
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

# === Browser speech outside video transformer ===
if st.session_state.last_pred != st.session_state.spoken_pred:
    speak_browser(st.session_state.last_pred)
    st.session_state.spoken_pred = st.session_state.last_pred
