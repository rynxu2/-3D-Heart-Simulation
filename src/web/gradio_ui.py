"""Gradio Web UI — Face upload → Classification → 3D Heart Simulation."""

import gradio as gr
import cv2
import numpy as np
import json
from pathlib import Path
from loguru import logger

from src.config import Config, load_heart_config
from src.data.face_detector import FaceDetector
from src.heart_simulation.heartbeat_engine import HeartbeatEngine
from src.heart_simulation.heart_model_anatomy import AnatomicalHeartModel
from src.data.dataset import LABEL_NAMES

# Global instances (lazy-loaded)
_face_detector = None
_predictor = None
_anatomy_model = None


def get_face_detector():
    global _face_detector
    if _face_detector is None:
        _face_detector = FaceDetector(target_size=224)
    return _face_detector


def get_anatomy_model():
    global _anatomy_model
    if _anatomy_model is None:
        _anatomy_model = AnatomicalHeartModel()
    return _anatomy_model


def predict_face(image):
    """Process uploaded face image and return classification results."""
    if image is None:
        return "❌ Vui lòng upload ảnh", None, "{}"

    detector = get_face_detector()
    annotated = detector.draw_detections(image)
    faces = detector.detect_faces(image)

    if not faces:
        return "❌ Không phát hiện khuôn mặt trong ảnh", annotated, "{}"

    # For demo without trained model: simulate prediction
    # In production, replace with: predictor.predict_image(image)
    demo_result = _demo_prediction()

    label = demo_result["label"]
    conf = demo_result["confidence"]
    bpm = demo_result["bpm"]

    label_vi = {"normal": "Bình thường", "abnormal": "Bất thường", "infarction": "Nhồi máu cơ tim"}

    result_text = (
        f"## 🫀 Kết quả phân loại\n\n"
        f"**Trạng thái:** {label_vi.get(label, label)}\n\n"
        f"**Độ tin cậy:** {conf:.1%}\n\n"
        f"**Nhịp tim ước tính:** {bpm['min']}-{bpm['max']} BPM\n\n"
        f"**Kiểu nhịp:** {bpm['pattern']}\n\n"
        f"---\n"
        f"### Xác suất từng lớp:\n"
        f"- Bình thường: {demo_result['all_probs']['normal']:.1%}\n"
        f"- Bất thường: {demo_result['all_probs']['abnormal']:.1%}\n"
        f"- Nhồi máu: {demo_result['all_probs']['infarction']:.1%}\n"
    )

    # Generate heart animation config
    engine = HeartbeatEngine.from_prediction(demo_result)
    anatomy = get_anatomy_model()
    heart_config = anatomy.get_animation_config(engine, condition=label)

    return result_text, annotated, json.dumps(heart_config, indent=2, ensure_ascii=False)


def simulate_heart(bpm, condition):
    """Manual heart simulation with custom BPM and condition."""
    pattern_map = {"Bình thường": "regular", "Bất thường": "irregular", "Nhồi máu cơ tim": "rapid_irregular"}
    label_map = {"Bình thường": "normal", "Bất thường": "abnormal", "Nhồi máu cơ tim": "infarction"}

    pattern = pattern_map.get(condition, "regular")
    label = label_map.get(condition, "normal")

    engine = HeartbeatEngine(bpm=int(bpm), pattern=pattern)
    anatomy = get_anatomy_model()
    config = anatomy.get_animation_config(engine, condition=label)

    info_text = (
        f"## 🫀 Mô phỏng tim\n\n"
        f"**BPM:** {int(bpm)}\n\n"
        f"**Trạng thái:** {condition}\n\n"
        f"**Chu kỳ:** {engine.cycle_duration:.3f}s\n\n"
        f"**Mô tả:** {engine.pattern_config['description']}\n"
    )

    return info_text, json.dumps(config, indent=2, ensure_ascii=False)


def compare_conditions():
    """Compare all 3 heart conditions side by side."""
    configs = {}
    conditions = [
        ("Bình thường", 72, "regular", "normal"),
        ("Bất thường", 110, "irregular", "abnormal"),
        ("Nhồi máu cơ tim", 120, "rapid_irregular", "infarction"),
    ]

    anatomy = get_anatomy_model()
    comparison = "## 🫀 So sánh 3 trạng thái tim\n\n"
    comparison += "| Trạng thái | BPM | Kiểu nhịp | Chu kỳ |\n"
    comparison += "|------------|-----|-----------|--------|\n"

    for name, bpm, pattern, label in conditions:
        engine = HeartbeatEngine(bpm=bpm, pattern=pattern)
        config = anatomy.get_animation_config(engine, condition=label)
        configs[label] = config
        comparison += f"| {name} | {bpm} | {pattern} | {engine.cycle_duration:.3f}s |\n"

    return comparison, json.dumps(configs, indent=2, ensure_ascii=False)


def _demo_prediction():
    """Demo prediction (replace with real model in production)."""
    import random
    labels = ["normal", "abnormal", "infarction"]
    chosen = random.choice(labels)

    config = Config.from_yaml()
    bpm_info = config.label_to_bpm.get(chosen)
    if not bpm_info:
        bpm_info = type("BPM", (), {"min": 72, "max": 72, "pattern": "regular"})()

    probs = [0.1, 0.1, 0.1]
    idx = labels.index(chosen)
    probs[idx] = 0.8

    return {
        "label": chosen,
        "confidence": probs[idx],
        "all_probs": {l: p for l, p in zip(labels, probs)},
        "bpm": {"min": bpm_info.min, "max": bpm_info.max, "pattern": bpm_info.pattern},
    }


def create_gradio_app():
    """Build the Gradio interface."""
    with gr.Blocks(
        title="🫀 Heart Condition Detection & 3D Simulation",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container { max-width: 1200px !important; }
        .result-box { font-size: 1.1em; }
        """
    ) as app:

        gr.Markdown(
            "# 🫀 Nhận Dạng Triệu Chứng Tim Qua Khuôn Mặt & Mô Phỏng 3D\n"
            "Upload ảnh khuôn mặt → AI phân loại trạng thái tim → Mô phỏng 3D tim đập"
        )

        with gr.Tabs():
            # Tab 1: Classification
            with gr.TabItem("📷 Phân Loại Khuôn Mặt"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(label="Upload ảnh khuôn mặt", type="numpy")
                        classify_btn = gr.Button("🔍 Phân tích", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        result_text = gr.Markdown(label="Kết quả", elem_classes=["result-box"])
                        detected_image = gr.Image(label="Face Detection")

                heart_config_json = gr.Code(label="Heart Animation Config (JSON)", language="json", visible=False)

                classify_btn.click(
                    fn=predict_face,
                    inputs=[input_image],
                    outputs=[result_text, detected_image, heart_config_json],
                )

            # Tab 2: Heart Simulation
            with gr.TabItem("🫀 Mô Phỏng 3D Tim"):
                with gr.Row():
                    with gr.Column(scale=1):
                        bpm_slider = gr.Slider(30, 180, value=72, step=1, label="BPM (Nhịp tim/phút)")
                        condition_dropdown = gr.Dropdown(
                            choices=["Bình thường", "Bất thường", "Nhồi máu cơ tim"],
                            value="Bình thường",
                            label="Trạng thái tim",
                        )
                        sim_btn = gr.Button("▶️ Mô phỏng", variant="primary")

                    with gr.Column(scale=1):
                        sim_info = gr.Markdown(label="Thông tin mô phỏng")

                sim_config_json = gr.Code(label="Animation Config", language="json")

                sim_btn.click(
                    fn=simulate_heart,
                    inputs=[bpm_slider, condition_dropdown],
                    outputs=[sim_info, sim_config_json],
                )

            # Tab 3: Comparison
            with gr.TabItem("📊 So Sánh"):
                compare_btn = gr.Button("🔄 So sánh 3 trạng thái", variant="primary")
                compare_text = gr.Markdown()
                compare_json = gr.Code(label="Configs", language="json")

                compare_btn.click(
                    fn=compare_conditions,
                    inputs=[],
                    outputs=[compare_text, compare_json],
                )

        gr.Markdown("---\n*Đồ án tốt nghiệp — Nhận dạng triệu chứng tim qua khuôn mặt*")

    return app


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
