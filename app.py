# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import joblib
import numpy as np
import cv2
import time
import queue
import tempfile
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from gtts import gTTS

# === Configuration ===
IMG_SIZE = (224, 224)
MODEL_PATH = "knn_model.pkl"
COOLDOWN = 5  # seconds between predictions

# === Initialize session state FIRST (before anything else) ===
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = ""
if "prediction_queue" not in st.session_state:
    st.session_state.prediction_queue = queue.Queue()

# === Load KNN model ===
model = joblib.load(MODEL_PATH)

# === Cache MobileNetV2 feature extractor ===
@st.cache_resource
def load_feature_extractor():
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

feature_extractor = load_feature_extractor()

# === Python TTS function using gTTS ===
def speak_text(text):
    """Convert text to speech using gTTS and play with autoplay"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_file = fp.name
            tts.save(temp_file)
            
        # Read audio file
        with open(temp_file, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        
        # Clean up temp file
        os.unlink(temp_file)
        
        return audio_bytes
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

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
        # Access the queue from session_state (already initialized above)
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
                    try:
                        self.prediction_queue.put(pred)
                    except Exception as e:
                        print(f"Queue error: {e}")

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

# Create placeholder for audio
audio_placeholder = st.empty()

# === Start live webcam streaming ===
webrtc_ctx = webrtc_streamer(
    key="guiding-eye",
    video_transformer_factory=ObjectDetectionTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

# === Process predictions from queue and trigger TTS ===
try:
    if not st.session_state.prediction_queue.empty():
        new_pred = st.session_state.prediction_queue.get_nowait()
        if new_pred != st.session_state.last_prediction:
            st.session_state.last_prediction = new_pred
            # Generate and play speech
            audio_bytes = speak_text(f"Detected {new_pred}")
            if audio_bytes:
                audio_placeholder.audio(audio_bytes, format='audio/mp3', autoplay=True)
except queue.Empty:
    pass
except Exception as e:
    print(f"TTS error: {e}")

# Display current prediction
if st.session_state.last_prediction:
    st.success(f"📢 Current detection: {st.session_state.last_prediction}")
