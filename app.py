import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from detector import PotholeDetector

st.set_page_config(
    page_title="Smart Road Pothole Detector",
    page_icon="🚧",
    layout="wide",
)

st.markdown("""
<style>
.main-title {font-size: 2.4rem; font-weight: 800; margin-bottom: 0;}
.subtitle {color: #777; margin-bottom: 1.2rem;}
.metric-card {padding: 1rem; border-radius: 12px; border: 1px solid #ddd;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🚧 Smart Road Pothole Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">OpenCV + YOLO + Streamlit interactive road inspection dashboard</p>', unsafe_allow_html=True)

@st.cache_resource
def load_detector(model_path, conf, iou):
    return PotholeDetector(model_path=model_path, confidence=conf, iou=iou)

with st.sidebar:
    st.header("⚙️ Detection Settings")
    source = st.radio("Input source", ["Video", "Image"], horizontal=False)

    conf = st.slider("Confidence threshold", 0.10, 0.95, 0.35, 0.05)
    iou = st.slider("IoU threshold", 0.10, 0.95, 0.45, 0.05)

    st.divider()
    st.subheader("🎨 Visualization")
    show_conf = st.checkbox("Show confidence", True)
    show_label = st.checkbox("Show severity", True)

    st.divider()
    st.info(
        "Place your trained pothole YOLO model at "
        "`models/best.pt`. The included fallback mode can still run "
        "with classical OpenCV detection for demonstration."
    )

model_path = "models/best.pt"
detector = load_detector(model_path, conf, iou)

if "history" not in st.session_state:
    st.session_state.history = []

def severity_from_area(area_ratio):
    if area_ratio >= 0.08:
        return "Severe"
    if area_ratio >= 0.025:
        return "Moderate"
    return "Low"

def process_frame(frame):
    result = detector.predict(frame)
    annotated = frame.copy()

    detections = []
    h, w = frame.shape[:2]

    for d in result:
        x1, y1, x2, y2 = d["bbox"]
        area_ratio = max(0, (x2-x1) * (y2-y1)) / float(w*h)
        severity = severity_from_area(area_ratio)

        label = "Pothole"
        if show_label:
            label += f" | {severity}"
        if show_conf:
            label += f" {d['confidence']:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            annotated, label, (x1, max(25, y1-10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            cv2.LINE_AA
        )

        detections.append({
            "confidence": d["confidence"],
            "severity": severity,
            "bbox": [x1, y1, x2, y2],
        })

    return annotated, detections

if source == "Image":
    uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if st.button("🔍 Detect Potholes", type="primary"):
            annotated, detections = process_frame(frame)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Original")
                st.image(image, use_container_width=True)
            with c2:
                st.subheader("Detection Result")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            count = len(detections)
            severe = sum(x["severity"] == "Severe" for x in detections)
            moderate = sum(x["severity"] == "Moderate" for x in detections)
            low = sum(x["severity"] == "Low" for x in detections)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🕳️ Potholes", count)
            m2.metric("🔴 Severe", severe)
            m3.metric("🟡 Moderate", moderate)
            m4.metric("🟢 Low", low)

            if detections:
                st.subheader("Detection Details")
                df = pd.DataFrame([
                    {"Pothole": i+1, "Confidence": round(x["confidence"], 3), "Severity": x["severity"]}
                    for i, x in enumerate(detections)
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.success("No potholes detected.")

else:
    uploaded = st.file_uploader("Upload a road video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded:
        suffix = Path(uploaded.name).suffix or ".mp4"
        in_path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
        with open(in_path, "wb") as f:
            f.write(uploaded.read())

        st.video(in_path)

        if st.button("▶️ Start Pothole Detection", type="primary"):
            cap = cv2.VideoCapture(in_path)

            if not cap.isOpened():
                st.error("Could not open the uploaded video.")
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (width, height)
            )

            preview = st.empty()
            progress = st.progress(0)
            status = st.empty()

            frame_no = 0
            total_detections = 0
            severity_counts = {"Severe": 0, "Moderate": 0, "Low": 0}
            records = []

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                annotated, detections = process_frame(frame)
                writer.write(annotated)

                total_detections += len(detections)
                for d in detections:
                    severity_counts[d["severity"]] += 1
                    records.append({
                        "frame": frame_no,
                        "confidence": d["confidence"],
                        "severity": d["severity"],
                    })

                if frame_no % 3 == 0:
                    preview.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )

                frame_no += 1
                if total_frames:
                    progress.progress(min(frame_no / total_frames, 1.0))
                status.write(f"Processing frame {frame_no}/{total_frames or '?'}")

            cap.release()
            writer.release()

            st.success("Detection completed.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🕳️ Total detections", total_detections)
            m2.metric("🔴 Severe", severity_counts["Severe"])
            m3.metric("🟡 Moderate", severity_counts["Moderate"])
            m4.metric("🟢 Low", severity_counts["Low"])

            st.subheader("🎬 Processed Video")
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Processed Video",
                    data=f,
                    file_name="pothole_detection_result.mp4",
                    mime="video/mp4",
                )

            if records:
                df = pd.DataFrame(records)
                st.subheader("📊 Detection Analytics")

                a, b = st.columns(2)
                with a:
                    st.bar_chart(df["severity"].value_counts())
                with b:
                    st.line_chart(df.groupby("frame")["confidence"].mean())

                st.dataframe(df, use_container_width=True, hide_index=True)
