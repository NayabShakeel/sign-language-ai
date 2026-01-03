import streamlit as st
from streamlit_webrtc import webrtc_streamer
import torch
import numpy as np
import av
import cv2
import pathlib
import platform

# 1. Fix for Windows/Linux path issues in YOLOv5
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

st.set_page_config(page_title="Sign Language AI", layout="wide")
st.title("🤟 Real-Time Sign Language Detection")

# 2. Optimized Model Loading
@st.cache_resource
def load_yolo():
    # Load custom model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    
    # --- ADJUST THESE TO FIX THE "HELP" GUESSING ISSUE ---
    model.conf = 0.60  # Only show signs if 60% sure (Prevents false positives)
    model.iou = 0.45   # Standard NMS threshold
    # ----------------------------------------------------
    return model

model = load_yolo()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # 3. Resize to 640 for consistency with training
    # Many models get stuck if the browser sends a weird resolution
    results = model(img, size=640)
    
    # Render results on the frame
    annotated_img = np.squeeze(results.render())
    
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# 4. Camera UI with Async Processing
webrtc_streamer(
    key="sign-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True, # Keeps the video smooth while AI thinks
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            }
        ]
    },
    media_stream_constraints={"video": True, "audio": False},
)
