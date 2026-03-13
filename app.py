import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="People Counter", page_icon="🟢", layout="centered")

st.title("🟢 People Counter with YOLOv8")
st.write("Upload an image and the app will count the number of people.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.image(image, caption="Original Image", use_column_width=True)

    model = YOLO("yolov8m.pt")

    results = model(image_cv)

    count = 0

    for r in results:
        boxes = r.boxes.xyxy
        classes = r.boxes.cls

        for box, cls in zip(boxes, classes):

            if int(cls) == 0:  # person class
                count += 1

                x1, y1, x2, y2 = map(int, box)

                cv2.rectangle(image_cv,(x1,y1),(x2,y2),(0,255,0),2)

                cv2.putText(image_cv,"Person",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    st.write(f"Number of people detected: **{count}**")

    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Detected Image", use_column_width=True)