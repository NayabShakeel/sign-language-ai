import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import pathlib
import platform
from streamlit_webrtc import webrtc_streamer
import av

# --- 1. SYSTEM COMPATIBILITY FIX ---
# This ensures the model loads whether you are on Windows or Linux (Streamlit Cloud)
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

st.set_page_config(
    page_title="Sign Language AI - CEP Submission", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("🤟 Sign Language Recognition System")
st.markdown("---")

# --- 2. THE CORRECT MODEL LOADING (STRICT) ---
@st.cache_resource
def load_custom_model():
    # We use force_reload=True to clear any 'useless words' from old COCO models
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    
    # Preprocessing Settings (Matching your Training/Validation)
    model.conf = 0.50  # Minimum confidence
    model.iou = 0.45   # Overlap threshold
    model.multi_label = False  # Each hand gets exactly ONE meaning
    return model

try:
    model = load_custom_model()
    class_names = list(model.names.values())
except Exception as e:
    st.error(f"Error loading best.pt: {e}")
    st.stop()

# --- 3. SIDEBAR (CONTROLS) ---
st.sidebar.title("🛠️ Control Center")
st.sidebar.success(f"Model Loaded: {len(class_names)} Signs Detected")

# Allow the user to manually override labels if the .pt file is showing numbers
if st.sidebar.checkbox("Show Label Names Debug"):
    st.sidebar.write(model.names)

# Threshold Slider
conf_threshold = st.sidebar.slider("Sensitivity (Confidence)", 0.1, 1.0, 0.5)
model.conf = conf_threshold

# --- 4. TABS: IMAGE VS LIVE CAMERA ---
tab_img, tab_cam = st.tabs(["📸 Upload & Test (Recommended)", "🎥 Live Camera Feed"])

# --- TAB: IMAGE UPLOAD (EXACT PREPROCESSING) ---
with tab_img:
    st.subheader("Upload an image for high-precision testing")
    file = st.file_uploader("Upload Sign Image", type=['jpg', 'png', 'jpeg'])
    
    if file:
        img = Image.open(file)
        # Preprocessing: The model(img, size=640) call automatically handles 
        # the letterbox resizing exactly like the validation script did!
        results = model(img, size=640)
        
        # Display Results
        rendered_img = results.render()[0]
        st.image(rendered_img, use_column_width=True)
        
        # Meaning Output
        df = results.pandas().xyxy[0]
        if not df.empty:
            for _, row in df.iterrows():
                st.success(f"### Detected: {row['name']} ({row['confidence']:.2f})")
        else:
            st.warning("No sign detected. Try lowering the Sensitivity slider.")

# --- TAB: LIVE CAMERA (REAL-TIME PREPROCESSING) ---
with tab_cam:
    st.subheader("Real-time webcam detection")
    
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Apply the exact same 640px preprocessing used in validation
        results = model(img, size=640)
        
        # Draw the labels
        annotated_img = np.squeeze(results.render())
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

    webrtc_streamer(
        key="sign-sign",
        video_frame_callback=video_frame_callback,
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
