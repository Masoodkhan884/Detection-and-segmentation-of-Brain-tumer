import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO, SAM

# ---------- Load Models ----------
@st.cache_resource
def load_models():
    yolo_model = YOLO("best.pt")       # Your trained YOLOv11 model
    sam_model = SAM("sam2_b.pt")       # Your SAM2 segmentation model
    return yolo_model, sam_model

# ---------- Convert PIL to OpenCV ----------
def pil_to_cv2(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ---------- Convert OpenCV to PIL ----------
def cv2_to_pil(img_cv2):
    return Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))

# ---------- Streamlit Layout ----------
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")
st.title("🧠 Brain Tumor Detection & Segmentation")
st.markdown("Upload an MRI image to detect and segment brain tumors using YOLOv11 and SAM2.")

# ---------- File Upload ----------
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Show original image
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption="🖼️ Original Image", use_container_width=True)

    # Convert to OpenCV format
    image_cv2 = pil_to_cv2(image_pil)

    # Load models
    yolo_model, sam_model = load_models()

    if st.button("🔍 Detect and Segment Tumor"):
        # Step 1: YOLOv11 Detection
        yolo_results = yolo_model(image_cv2)
        detection_img = image_cv2.copy()

        if yolo_results[0].boxes is not None:
            boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
            class_ids = yolo_results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = yolo_results[0].boxes.conf.cpu().numpy()
            detected_classes = []

            for box, cls_id, conf in zip(boxes, class_ids, confidences):
                x1, y1, x2, y2 = map(int, box)
                label = yolo_model.names[cls_id]
                detected_classes.append(label)

                text = f"{label} ({conf:.2f})"
                cv2.rectangle(detection_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(detection_img, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show detection with classification
            st.image(cv2_to_pil(detection_img),
                     caption="📦 Tumor Detection with Classification", use_container_width=True)

            st.subheader("🧬 Tumor Types Detected:")
            for i, label in enumerate(set(detected_classes), start=1):
                st.markdown(f"**{i}. {label}**")

            # Step 2: SAM2 Segmentation (masking detected tumor)
            st.subheader("🧠 Segmentation with SAM2")
            sam_results = sam_model(image_cv2, bboxes=yolo_results[0].boxes.xyxy, verbose=False, save=False)

            if sam_results and len(sam_results):
                segmented_img = sam_results[0].plot()
                st.image(segmented_img,
                         caption="🧠 Segmented Tumor Region", use_container_width=True)
            else:
                st.warning("Segmentation failed. Try another image.")
        else:
            st.warning("No tumor detected in the image.")
