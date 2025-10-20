# app.py
import streamlit as st
import numpy as np
import joblib
import time
from PIL import Image
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit.components.v1 import html

# === Configuration ===
IMG_SIZE = (224, 224)
MODEL_PATH = "knn_model.pkl"

# === Load KNN model ===
model = joblib.load(MODEL_PATH)

# === Load MobileNetV2 feature extractor ===
feature_extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

# === Browser speech ===
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
st.write("This app detects objects from your webcam live in the browser and speaks the result.")

# Optional: WebRTC config for better browser compatibility
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# === Video transformer for live processing ===
class ObjectDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_spoken = ""
        self.last_time = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Predict only once every 3 seconds
        if time.time() - self.last_time > 3:
            try:
                feat = extract_frame_feature(img)
                pred = model.predict([feat])[0]

                # Speak if new prediction
                if pred != self.last_spoken:
                    speak_browser(pred)
                    self.last_spoken = pred

                self.last_time = time.time()
            except Exception as e:
                print("Detection error:", e)

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
