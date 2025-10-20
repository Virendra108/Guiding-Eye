import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import joblib, numpy as np, cv2, time
from streamlit.components.v1 import html
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# --- Config ---
IMG_SIZE = (224, 224)
MODEL_PATH = "knn_model.pkl"
COOLDOWN = 5

model = joblib.load(MODEL_PATH)

@st.cache_resource
def load_feature_extractor():
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

feature_extractor = load_feature_extractor()

def extract_frame_feature(frame):
    image = cv2.resize(frame, IMG_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = feature_extractor.predict(image)
    return features[0]

def speak_browser(text):
    js_code = f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{text}");
        msg.rate = 1.0;
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(msg);
    </script>
    """
    html(js_code)

# --- Shared session state ---
if "last_pred" not in st.session_state:
    st.session_state.last_pred = ""
if "new_pred" not in st.session_state:
    st.session_state.new_pred = ""

# --- Streamlit UI ---
st.title("🔍 Guiding Eye – Live Object Recognition")
st.write(f"Detecting objects from your webcam (prediction every {COOLDOWN}s)")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class ObjectDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_time = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Predict every cooldown interval
        if time.time() - self.last_time > COOLDOWN:
            try:
                feat = extract_frame_feature(img)
                pred = model.predict([feat])[0]
                st.session_state.new_pred = pred  # update global prediction
                self.last_time = time.time()
            except Exception as e:
                print("Prediction error:", e)

        # Draw text overlay
        pred = st.session_state.new_pred
        if pred:
            cv2.putText(img, f"Detected: {pred}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

# --- Run WebRTC ---
webrtc_streamer(
    key="guiding-eye",
    video_transformer_factory=ObjectDetectionTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

# --- Browser speech trigger ---
if st.session_state.new_pred != st.session_state.last_pred:
    speak_browser(st.session_state.new_pred)
    st.session_state.last_pred = st.session_state.new_pred
