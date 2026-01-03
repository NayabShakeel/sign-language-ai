import streamlit as st
from streamlit_webrtc import webrtc_streamer
import torch
import numpy as np
import av
import cv2
import pathlib
import platform

# 1. Handle Path Compatibility (Prevents errors between Windows/Linux)
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

st.set_page_config(page_title="Sign Language AI", layout="wide")
st.title("🤟 Real-Time Sign Language Detection")

# 2. Load the Model
@st.cache_resource
def load_yolo():
    # Force reload ensures 'best.pt' is updated if you change the file
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    return model

model = load_yolo()

# 3. Control Panel in Sidebar
st.sidebar.header("Model Settings")

# Sensitivity Control
conf_val = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.65)
model.conf = conf_val

# Class Filtering: Select what you WANT to see
all_classes = list(model.names.values())
selected_classes = st.sidebar.multiselect(
    "Active Detection Classes:", 
    all_classes, 
    default=[c for c in all_classes if c not in ["I Love You", "Help"]]
)

# Map selected names back to IDs
selected_ids = [k for k, v in model.names.items() if v in selected_classes]
model.classes = selected_ids if selected_ids else None

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # 4. Run AI with fixed size for accuracy
    results = model(img, size=640)
    
    # Render (draw boxes)
    annotated_img = np.squeeze(results.render())
    
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# 5. Live Stream
webrtc_streamer(
    key="sign-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
