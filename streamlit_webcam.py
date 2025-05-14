import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Simple Streamlit Webcam App
# Requirements: pip install streamlit opencv-python pillow

st.title("Webcam Capture with Streamlit")
st.write("Use your browser's webcam to capture an image and apply basic processing.")

# Capture an image from the webcam
img_file_buffer = st.camera_input("🔴 Capture Image from Webcam")

if img_file_buffer is not None:
    # Open image from buffer
    image = Image.open(img_file_buffer)
    st.image(image, caption="Original Image", use_column_width=True)

    # Convert to numpy array for OpenCV processing
    img_array = np.array(image)
    # Convert BGR (OpenCV) <-> RGB (PIL)
    # PIL opens in RGB; convert to BGR for OpenCV functions if needed
    bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Example processing: convert to grayscale
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    st.image(gray, caption="Grayscale Image", use_column_width=True, clamp=True, channels="GRAY")

    # Example processing: edge detection
    edges = cv2.Canny(gray, 100, 200)
    st.image(edges, caption="Edges (Canny)", use_column_width=True, clamp=True, channels="GRAY")

# Run this app with:
# streamlit run streamlit_webcam_app.py
