# -*- coding: utf-8 -*-
import gradio as gr
import torch
import mimetypes
import csv
import os
from datetime import datetime
from PIL import Image

from pipeline.face_detector import FaceDetector
from pipeline.gradcam import GradCAM
from pipeline.video_analyzer import VideoAnalyzer
from pipeline.predictor import DeepfakePredictor

# === Initialize Pipeline ===
print("Loading pipeline...")
predictor = DeepfakePredictor()
face_detector = FaceDetector()
video_analyzer = VideoAnalyzer(num_frames=10)
gradcam = GradCAM(predictor.model)
print("Pipeline ready!")

# === History Log Setup ===
HISTORY_FILE = "history/detections.csv"
os.makedirs("history", exist_ok=True)
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Filename", "Type",
                         "Face Detected", "Prediction", "Confidence"])

def log_result(filename, file_type, face_found, label, confidence):
    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            filename, file_type,
            "Yes" if face_found else "No",
            label, f"{confidence*100:.2f}%"
        ])

def interpret(conf, pred):
    if conf < 0.6:
        return "UNCERTAIN"
    return "REAL" if pred == 0 else "DEEPFAKE"

def predict_file(file_obj):
    if file_obj is None:
        return "Awaiting input...", "", None, None, ""

    path = file_obj.name
    filename = os.path.basename(path)
    mime, _ = mimetypes.guess_type(path)

    if mime and mime.startswith("image"):
        try:
            img = Image.open(path).convert("RGB")
            face_img, face_found = face_detector.extract_face(img)
            conf, pred, probs = predictor.predict(face_img)
            tensor = predictor.predict_with_grad(face_img)
            cam = gradcam.generate(tensor, class_idx=pred)
            heatmap = gradcam.overlay(face_img, cam)
            label = interpret(conf, pred)
            face_status = "Face detected" if face_found else "No face - using full image"
            full_label = f"{label} — {face_status}"
            log_result(filename, "Image", face_found, label, conf)
            return full_label, f"{conf*100:.2f}%", face_img, heatmap, ""
        except Exception as e:
            return f"Error: {e}", "", None, None, ""

    elif mime and mime.startswith("video"):
        try:
            frames = video_analyzer.extract_frames(path)
            if not frames:
                return "Error reading video", "", None, None, ""
            fake_probs = []
            first_face = None
            first_heatmap = None
            for i, frame in enumerate(frames):
                face_img, face_found = face_detector.extract_face(frame)
                conf, pred, probs = predictor.predict(face_img)
                if probs is not None:
                    fake_probs.append(probs[1].item())
                if i == 0:
                    first_face = face_img
                    tensor = predictor.predict_with_grad(face_img)
                    cam = gradcam.generate(tensor, class_idx=pred)
                    first_heatmap = gradcam.overlay(face_img, cam)
            conf, pred = video_analyzer.aggregate_predictions(fake_probs)
            label = interpret(conf, pred)
            full_label = f"{label} — {len(frames)} frames analyzed"
            frame_results = video_analyzer.get_frame_results(fake_probs)
            log_result(filename, "Video", True, label, conf)
            return full_label, f"{conf*100:.2f}%", first_face, first_heatmap, frame_results
        except Exception as e:
            return f"Error: {e}", "", None, None, ""
    else:
        return "Unsupported file type", "", None, None, ""

# === Custom CSS ===
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

:root {
    --bg-primary: #020408;
    --bg-secondary: #080f17;
    --bg-card: #0d1821;
    --accent-cyan: #00d4ff;
    --accent-red: #ff2d55;
    --accent-green: #00ff88;
    --accent-orange: #ff9500;
    --text-primary: #e0f0ff;
    --text-secondary: #5a7a9a;
    --border: #0a2040;
    --glow-cyan: 0 0 20px rgba(0, 212, 255, 0.3);
    --glow-red: 0 0 20px rgba(255, 45, 85, 0.4);
    --glow-green: 0 0 20px rgba(0, 255, 136, 0.3);
}

* { box-sizing: border-box; }

body, .gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'Rajdhani', sans-serif !important;
    color: var(--text-primary) !important;
}

/* Animated grid background */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

/* Header */
.header-section {
    text-align: center;
    padding: 40px 20px 30px;
    position: relative;
}

.header-title {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 3em !important;
    font-weight: 700 !important;
    letter-spacing: 8px !important;
    color: var(--accent-cyan) !important;
    text-shadow: var(--glow-cyan) !important;
    text-transform: uppercase !important;
    margin: 0 !important;
    animation: flicker 4s infinite !important;
}

.header-sub {
    font-size: 1em !important;
    color: var(--text-secondary) !important;
    letter-spacing: 4px !important;
    text-transform: uppercase !important;
    margin-top: 8px !important;
    font-family: 'Share Tech Mono', monospace !important;
}

@keyframes flicker {
    0%, 95%, 100% { opacity: 1; }
    96% { opacity: 0.8; }
    97% { opacity: 1; }
    98% { opacity: 0.7; }
}

/* Status bar */
.status-bar {
    display: flex;
    justify-content: center;
    gap: 30px;
    padding: 10px;
    margin: 0 auto 20px;
    max-width: 600px;
    border: 1px solid var(--border);
    background: var(--bg-secondary);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75em;
    color: var(--accent-green);
    letter-spacing: 2px;
}

/* Upload zone */
.upload-zone {
    border: 1px solid var(--accent-cyan) !important;
    background: var(--bg-card) !important;
    border-radius: 0 !important;
    box-shadow: var(--glow-cyan) !important;
    transition: all 0.3s ease !important;
}

.upload-zone:hover {
    box-shadow: 0 0 40px rgba(0, 212, 255, 0.5) !important;
    border-color: #00ffff !important;
}

/* Result boxes */
.result-box {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.1em !important;
    color: var(--accent-cyan) !important;
    letter-spacing: 2px !important;
}

/* Labels */
label, .label-wrap {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75em !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: var(--text-secondary) !important;
}

/* Image panels */
.image-panel {
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    border-radius: 0 !important;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 20px 0;
}

/* Info section */
.info-section {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75em;
    color: var(--text-secondary);
    letter-spacing: 2px;
    text-align: center;
    padding: 15px;
    border-top: 1px solid var(--border);
}

/* Buttons */
button {
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--accent-cyan); }

/* Textarea */
textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--accent-green) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85em !important;
    letter-spacing: 1px !important;
    border-radius: 0 !important;
}

/* Input */
input[type="text"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--accent-cyan) !important;
    font-family: 'Share Tech Mono', monospace !important;
    border-radius: 0 !important;
}
"""

# === Gradio UI ===
with gr.Blocks(
    title="Deepfake Detector",
    css=custom_css
) as demo:

    gr.HTML("""
    <div class="header-section">
        <div class="header-title">DEEPFAKE DETECTOR</div>
        <div class="header-sub">AI-Powered Media Authenticity Analysis System</div>
        <div class="status-bar">
            <span>SYS: ONLINE</span>
            <span>MODEL: EFFICIENTNET-B0</span>
            <span>PIPELINE: ACTIVE</span>
        </div>
    </div>
    """)

    with gr.Row():
        file_input = gr.File(
            label="[ UPLOAD TARGET — IMAGE OR VIDEO ]",
            file_types=[".jpg", ".jpeg", ".png", ".mp4", ".mov"],
            elem_classes=["upload-zone"]
        )

    with gr.Row():
        prediction = gr.Textbox(
            label="[ VERDICT ]",
            interactive=False,
            elem_classes=["result-box"]
        )
        confidence = gr.Textbox(
            label="[ CONFIDENCE SCORE ]",
            interactive=False,
            elem_classes=["result-box"]
        )

    with gr.Row():
        preview = gr.Image(
            label="[ DETECTED FACE ]",
            elem_classes=["image-panel"]
        )
        heatmap = gr.Image(
            label="[ GRAD-CAM HEATMAP — SUSPICIOUS REGIONS ]",
            elem_classes=["image-panel"]
        )

    frame_results = gr.Textbox(
        label="[ FRAME-BY-FRAME ANALYSIS — VIDEO ONLY ]",
        interactive=False,
        lines=6
    )

    gr.HTML("""
    <div class="info-section">
        <p>HEATMAP: RED = HIGH SUSPICION | BLUE = LOW SUSPICION</p>
        <p>UNCERTAIN = CONFIDENCE BELOW 60% — SUBMIT HIGHER QUALITY INPUT</p>
        <p>ALL DETECTIONS LOGGED TO history/detections.csv</p>
    </div>
    """)

    file_input.change(
        fn=predict_file,
        inputs=file_input,
        outputs=[prediction, confidence, preview, heatmap, frame_results]
    )

demo.launch()