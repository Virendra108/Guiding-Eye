import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit.components.v1 import html

IMG_SIZE = (224, 224)
MODEL_PATH = "knn_model.pkl"

model = joblib.load(MODEL_PATH)
feature_extractor = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

def speak_browser(text):
    js_code = f"""
    <script>
        var msg = new SpeechSynthesisUtterance("{text}");
        window.speechSynthesis.speak(msg);
    </script>
    """
    html(js_code)

def extract_frame_feature(frame):
    image = cv2.resize(frame, IMG_SIZE)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = feature_extractor.predict(image)
    return features[0]

st.title("🔍 Guiding Eye – Real-time Object Recognition")
st.write("This app detects objects using your webcam and speaks the result in your browser.")

uploaded_image = st.camera_input("Show your camera")

if uploaded_image is not None:
    # Convert to OpenCV format
    image = np.array(Image.open(uploaded_image))
    pred_feature = extract_frame_feature(image)
    pred = model.predict([pred_feature])[0]
    st.success(f"Detected: {pred}")
    speak_browser(pred)
