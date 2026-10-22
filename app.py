"""
Solar Panel Thermal Anomaly Detection & Segmentation Web App
Interactive Gradio demonstration powered by YOLOv8.
"""

import os
from pathlib import Path
import numpy as np
import gradio as gr
from ultralytics import YOLO

# Resolve paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "yolov8s_seg_best.pt"
SAMPLE_DIR = BASE_DIR / "sample_images"

# Check if model checkpoint exists, otherwise fallback to base segmentation model
if MODEL_PATH.exists():
    print(f"Loading custom fine-tuned model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
else:
    print(f"Custom model not found at {MODEL_PATH}. Initializing YOLOv8s-seg base model.")
    model = YOLO("yolov8s-seg.pt")

# Severity and diagnosis lookup for each consolidated class
ANOMALY_INFO = {
    "Hotspot": {
        "severity": "Medium - High",
        "action": "Inspect cell junction for localized microcracks, solder fatigue, or localized soil accumulation. May require individual cell bypass."
    },
    "Diode Failure": {
        "severity": "Critical",
        "action": "Immediate junction box inspection required. Replace short-circuited/open bypass diode to prevent string degradation and fire risk."
    },
    "Reverse Polarity Failure": {
        "severity": "High",
        "action": "Check string electrical connection and module polarity. Inverted module connection causes severe resistive heating."
    },
    "String Outage": {
        "severity": "Critical",
        "action": "Entire series string disconnected or deactivated. Check array inverter DC switches, fuses, and cable harnesses."
    },
    "Shading": {
        "severity": "Low - Medium",
        "action": "Clean module surface from external debris, bird droppings, or foliage casting shadows over the active PV area."
    }
}


def predict_thermal_anomaly(image, conf_threshold: float = 0.25):
    """
    Runs YOLOv8 segmentation on the uploaded thermal infrared image.
    Returns:
        annotated_image: RGB image with segmentation polygons & bounding boxes
        report: Formatted diagnostic summary markdown text
    """
    if image is None:
        return None, "### ⚠️ Please upload or select a solar panel thermal image."

    results = model.predict(
        source=image,
        imgsz=640,
        conf=conf_threshold,
        verbose=False
    )

    result = results[0]

    # Generate annotated image (BGR -> RGB)
    annotated = result.plot()
    if annotated.shape[-1] == 3:
        annotated = annotated[:, :, ::-1]

    # Parse predictions
    if result.boxes is None or len(result.boxes) == 0:
        report = "### ✅ No Anomalies Detected\nThe panel appears thermally uniform at the current confidence threshold."
        return annotated, report

    names = result.names
    defect_counts = {}
    defect_lines = []

    for i in range(len(result.boxes)):
        class_id = int(result.boxes.cls[i].item())
        confidence = float(result.boxes.conf[i].item())
        class_name = names.get(class_id, f"Class {class_id}")

        defect_counts[class_name] = defect_counts.get(class_name, 0) + 1
        info = ANOMALY_INFO.get(class_name, {"severity": "Moderate", "action": "General thermal audit"})

        defect_lines.append(
            f"- **#{i+1}: {class_name}** | Confidence: **{confidence:.1%}** | Severity: `{info['severity']}`\n"
            f"  *Recommendation:* {info['action']}"
        )

    summary_counts = ", ".join([f"**{count}x {cls}**" for cls, count in defect_counts.items()])
    report = (
        f"### 🔍 Diagnostic Report\n"
        f"**Total Defects Detected:** {len(result.boxes)} ({summary_counts})\n\n"
        + "\n".join(defect_lines)
    )

    return annotated, report


# Prepare sample images for Gradio UI examples
example_images = []
if SAMPLE_DIR.exists():
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        example_images.extend([[str(p), 0.25] for p in list(SAMPLE_DIR.glob(ext))[:6]])

# Gradio Interface
with gr.Blocks(title="Solar Panel Thermal Anomaly Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ☀️ Solar Panel Thermal Anomaly Detection & Segmentation
        ### Real-Time Deep Learning Photovoltaic Defect Analysis powered by **YOLOv8**
        Upload an aerial or handheld thermal infrared (TIR) image of a solar panel array to automatically detect, 
        classify, and segment thermal anomalies (Hotspots, Diode Failures, String Outages, Reverse Polarity, Shading).
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Upload Thermal Image")
            conf_slider = gr.Slider(
                minimum=0.10, maximum=0.90, value=0.25, step=0.05,
                label="Confidence Threshold"
            )
            analyze_btn = gr.Button("🔍 Analyze Panel", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Segmented Anomaly Visualization")
            output_report = gr.Markdown(label="Diagnosis & Maintenance Guidelines")

    analyze_btn.click(
        fn=predict_thermal_anomaly,
        inputs=[input_image, conf_slider],
        outputs=[output_image, output_report]
    )

    if example_images:
        gr.Examples(
            examples=example_images,
            inputs=[input_image, conf_slider],
            outputs=[output_image, output_report],
            fn=predict_thermal_anomaly,
            cache_examples=False,
            label="Example Thermal Test Images"
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
