import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import streamlit as st
import cv2
import numpy as np
from src.detector import detect_objects

MAX_WIDTH = 1024  


def resize_if_needed(img, max_width=MAX_WIDTH):
    """Resize image proportionally if it exceeds max_width."""
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


st.title("Object Detection AI")
st.write("Upload an image to detect objects using MobileNet-SSD")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

   
    image = resize_if_needed(image)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image",
             use_container_width=True)

    result = detect_objects(image, confidence_threshold=0.8)

    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Detected Objects",
             use_container_width=True)

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/result.jpg", result)
    st.success("Detection complete! Saved to outputs/result.jpg")