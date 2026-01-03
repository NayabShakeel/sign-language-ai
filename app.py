import streamlit as st
from streamlit_webrtc import webrtc_streamer
import torch
import numpy as np
import av
import pathlib
import platform

# Path fix
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

st.set_page_config(page_title="Debug Sign AI", layout="wide", initial_sidebar_state="expanded")
st.title("Sign Language Debugger")

@st.cache_resource
def load_yolo():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

model = load_yolo()

# --- DEBUG CONTROLS ---
st.sidebar.title("Debug Settings")
# Set this low (0.30) to see if boxes appear at all!
conf_val = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.3)
model.conf = conf_val

all_labels = list(model.names.values())
st.sidebar.write(f"✅ Model loaded {len(all_labels)} signs.")
st.sidebar.write(f"Signs detected: {all_labels}")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model(img, size=640)
    
    # This will print the confidence in the 'Manage App' logs
    if len(results.pandas().xyxy[0]) > 0:
        print(results.pandas().xyxy[0][['name', 'confidence']])
        
    annotated_img = np.squeeze(results.render())
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

webrtc_streamer(
    key="debug-sign",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)
