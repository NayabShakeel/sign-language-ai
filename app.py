import streamlit as st
from streamlit_webrtc import webrtc_streamer
import torch
import numpy as np
import av
import cv2

st.set_page_config(page_title="Sign Language Detector", layout="wide")
st.title("🤟 Real-Time Sign Language Detection")

# Load YOLOv5 model from the root of your GitHub repo
@st.cache_resource
def load_yolo():
    # 'custom' means it will look for your 'best.pt' file
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

model = load_yolo()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Inference
    results = model(img)
    
    # Draw results on the frame
    annotated_img = np.squeeze(results.render())
    
    return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# Camera UI
webrtc_streamer(
    key="sign-detection",
    video_frame_callback=video_frame_callback,
    # This configuration adds a fallback relay (TURN) for restricted networks
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
