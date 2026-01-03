import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image, ImageOps
import pathlib
import platform

# 1. POSIX Path Fix (Crucial for Streamlit Cloud)
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

st.set_page_config(page_title="Sign Language AI Debugger", layout="wide")

# 2. LOAD MODEL & AUTO-SYNC LABELS
@st.cache_resource
def load_yolo():
    # Load the model directly from best.pt
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    return model

model = load_yolo()
# GET THE ACTUAL NAMES FROM THE FILE
model_names = model.names 

st.title("🤟 Sign Language AI: Debug Mode")
st.sidebar.header("Model Statistics")
st.sidebar.write(f"**Loaded Classes:** {len(model_names)}")
st.sidebar.json(model_names) # This will show you exactly what class is 0, 1, 2...

# 3. SETTINGS
conf_val = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.40)
model.conf = conf_val

# 4. TESTING INTERFACE
st.info("💡 Tip: If you get no results, try moving closer to the camera or using a plain background.")

uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Read Image
    img = Image.open(uploaded_file).convert("RGB")
    
    # --- IMAGE PREPROCESSING ---
    # Sometimes auto-rotation from phones messes up the AI
    img = ImageOps.exif_transpose(img) 
    
    # Run Inference (exactly like validation)
    results = model(img, size=640)
    
    # Draw Boxes
    annotated_img = results.render()[0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Original Image", use_column_width=True)
    with col2:
        st.image(annotated_img, caption="AI Prediction", use_column_width=True)

    # 5. DETAILED DATA TABLE
    df = results.pandas().xyxy[0]
    if not df.empty:
        st.success("### Detection Found!")
        st.dataframe(df[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])
        
        # Best prediction
        top_sign = df.iloc[0]['name']
        st.write(f"## Predicted Sign: **{top_sign}**")
    else:
        st.error("No Sign Detected. The AI is not confident enough.")
