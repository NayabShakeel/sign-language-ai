import streamlit as st
import torch
from PIL import Image
import numpy as np
import pathlib
import platform
from streamlit_webrtc import webrtc_streamer
import av

# 1. Path fix for cross-platform model loading
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

st.set_page_config(page_title="Sign Language AI Submission", layout="wide")

st.title("🤟 Sign Language Recognition System")
st.write("Project Submission Mode: Static Image Testing & Real-time AI")

# 2. Load the Model
@st.cache_resource
def load_yolo():
    # Looks for 'best.pt' in your folder
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

model = load_yolo()

# 3. Create Tabs for different testing methods
tab1, tab2 = st.tabs(["📁 Upload Image (Safe for Testing)", "🎥 Real-time Camera"])

# --- TAB 1: UPLOAD IMAGE (BEST FOR YOUR SUBMISSION) ---
with tab1:
    st.header("Test with Downloaded Images")
    st.write("Download any sign language photo from Google and upload it here to see the model work.")
    
    conf_img = st.slider("Confidence Threshold (Image)", 0.1, 1.0, 0.5, key="conf_img")
    uploaded_file = st.file_uploader("Choose a sign language image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the file to an image
        image = Image.open(uploaded_file)
        st.image(image, caption='Original Uploaded Image', use_column_width=True)
        
        # Run AI Inference
        st.write("AI is analyzing...")
        model.conf = conf_img
        results = model(image)
        
        # Show results
        annotated_img = results.render()[0] # This gets the image with boxes drawn
        st.image(annotated_img, caption='AI Prediction Result', use_column_width=True)
        
        # Show the detected sign meaning as text
        if len(results.pandas().xyxy[0]) > 0:
            for index, row in results.pandas().xyxy[0].iterrows():
                st.success(f"Detected Sign: **{row['name']}** (Confidence: {row['confidence']:.2f})")
        else:
            st.warning("No sign detected. Try lowering the Confidence Threshold.")

# --- TAB 2: LIVE CAMERA ---
with tab2:
    st.header("Real-time Camera Feed")
    conf_cam = st.slider("Confidence Threshold (Camera)", 0.1, 1.0, 0.4, key="conf_cam")
    
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        model.conf = conf_cam
        results = model(img, size=640)
        annotated_img = np.squeeze(results.render())
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

    webrtc_streamer(
        key="live-test",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
