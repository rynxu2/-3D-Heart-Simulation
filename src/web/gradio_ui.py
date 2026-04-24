"""Gradio Web UI — Face upload → Classification → ECG + Heart Simulation."""

import gradio as gr
import cv2
import torch
import numpy as np
import json
import tempfile
from pathlib import Path
from loguru import logger

from src.config import SynPainConfig, load_app_config
from src.data.augmentation import get_val_transforms
from src.models.classifier_2d import HeartFaceClassifier2D
from src.heart_simulation.heartbeat_engine import HeartbeatEngine
from src.heart_simulation.heart_model_anatomy import AnatomicalHeartModel
from src.heart_simulation.heart_renderer import HeartRenderer

# SynPAIN binary labels
SYNPAIN_LABELS = {0: "no_pain", 1: "pain"}
LABEL_DISPLAY = {"no_pain": "Không đau - Bình thường", "pain": "Đau - Nghi ngờ bất thường tim"}
LABEL_TO_CONDITION = {"no_pain": "normal", "pain": "infarction"}
LABEL_TO_BPM = {
    "no_pain": {"min": 60, "max": 80, "pattern": "regular"},
    "pain": {"min": 90, "max": 130, "pattern": "rapid_irregular"},
}

# Globals (lazy-loaded)
_face_detector = None
_model = None
_transform = None
_renderer = None
_anatomy = None


def _get_detector():
    global _face_detector
    if _face_detector is None:
        from src.data.face_detector import FaceDetector
        _face_detector = FaceDetector(target_size=224)
    return _face_detector


def _get_model():
    global _model, _transform
    if _model is None:
        app_cfg = load_app_config()
        model_path = Path(app_cfg.get("paths", {}).get("model_checkpoint", "models/synpain_best_model.pth"))
        device = "cuda" if torch.cuda.is_available() else "cpu"

        _model = HeartFaceClassifier2D(
            backbone="efficientnet_b4", num_classes=2, pretrained=False, dropout=0.3
        ).to(device)

        if model_path.exists():
            ckpt = torch.load(str(model_path), map_location=device, weights_only=True)
            _model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            logger.info(f"✅ Model loaded: {model_path}")
        else:
            logger.warning(f"⚠️ Model not found: {model_path}")

        _model.eval()
        _transform = get_val_transforms(224)
    return _model, _transform


def _get_renderer():
    global _renderer
    if _renderer is None:
        _renderer = HeartRenderer(output_dir=tempfile.mkdtemp())
    return _renderer


def _get_anatomy():
    global _anatomy
    if _anatomy is None:
        _anatomy = AnatomicalHeartModel()
    return _anatomy


def predict_face(image):
    """Process uploaded face → model prediction → ECG visualization."""
    if image is None:
        return "❌ Vui lòng upload ảnh", None, None

    detector = _get_detector()
    annotated = detector.draw_detections(image)
    face = detector.process_image(image)

    if face is None:
        return "❌ Không phát hiện khuôn mặt trong ảnh", annotated, None

    # Real model inference
    model, transform = _get_model()
    device = next(model.parameters()).device
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    tensor = transform(image=face_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    pred_label = SYNPAIN_LABELS[pred_id]
    confidence = float(probs[pred_id])
    condition = LABEL_TO_CONDITION[pred_label]
    bpm_info = LABEL_TO_BPM[pred_label]
    avg_bpm = (bpm_info["min"] + bpm_info["max"]) // 2

    result_text = (
        f"## 🫀 Kết quả phân loại\n\n"
        f"**Trạng thái:** {LABEL_DISPLAY[pred_label]}\n\n"
        f"**Độ tin cậy:** {confidence:.1%}\n\n"
        f"**Nhịp tim ước tính:** {bpm_info['min']}-{bpm_info['max']} BPM\n\n"
        f"**Kiểu nhịp:** {bpm_info['pattern']}\n\n"
        f"---\n"
        f"### Xác suất:\n"
        f"- 🟢 Không đau (Bình thường): {probs[0]:.1%}\n"
        f"- 🔴 Đau (Bất thường): {probs[1]:.1%}\n"
    )

    # Generate ECG waveform image
    engine = HeartbeatEngine(bpm=avg_bpm, pattern=bpm_info["pattern"])
    renderer = _get_renderer()
    ecg_path = renderer.render_ecg_waveform(engine, condition, duration=5.0)
    ecg_img = cv2.cvtColor(cv2.imread(str(ecg_path)), cv2.COLOR_BGR2RGB)

    return result_text, annotated, ecg_img


def simulate_heart(bpm, condition_vi):
    """Manual heart simulation."""
    pattern_map = {"Bình thường": "regular", "Bất thường": "irregular", "Nhồi máu cơ tim": "rapid_irregular"}
    label_map = {"Bình thường": "normal", "Bất thường": "abnormal", "Nhồi máu cơ tim": "infarction"}

    pattern = pattern_map.get(condition_vi, "regular")
    condition = label_map.get(condition_vi, "normal")

    engine = HeartbeatEngine(bpm=int(bpm), pattern=pattern)
    renderer = _get_renderer()
    ecg_path = renderer.render_ecg_waveform(engine, condition, duration=5.0)
    ecg_img = cv2.cvtColor(cv2.imread(str(ecg_path)), cv2.COLOR_BGR2RGB)

    info_text = (
        f"## 🫀 Mô phỏng tim\n\n"
        f"**BPM:** {int(bpm)}\n\n"
        f"**Trạng thái:** {condition_vi}\n\n"
        f"**Chu kỳ:** {engine.cycle_duration:.3f}s\n\n"
        f"**Mô tả:** {engine.pattern_config['description']}\n"
    )

    return info_text, ecg_img


def compare_conditions():
    """Compare all 3 conditions side by side."""
    renderer = _get_renderer()
    path = renderer.render_comparison(duration=5.0)
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

    text = (
        "## 🫀 So sánh 3 trạng thái tim\n\n"
        "| Trạng thái | BPM | Kiểu nhịp | Đặc điểm ECG |\n"
        "|------------|-----|-----------|---------------|\n"
        "| 🟢 Bình thường | 72 | regular | PQRST chuẩn |\n"
        "| 🟡 Bất thường | 110 | irregular | Nhịp không đều |\n"
        "| 🔴 Nhồi máu | 120 | rapid_irregular | ST chênh lên |\n"
    )
    return text, img


def create_gradio_app():
    """Build the Gradio interface."""
    with gr.Blocks(title="🫀 Heart Condition Detection") as app:

        gr.Markdown(
            "# 🫀 Nhận Dạng Triệu Chứng Tim Qua Khuôn Mặt & Mô Phỏng 3D\n"
            "Upload ảnh khuôn mặt → AI phân loại (SynPAIN model) → ECG mô phỏng"
        )

        with gr.Tabs():
            # Tab 1: Classification
            with gr.TabItem("📷 Phân Loại Khuôn Mặt"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(label="Upload ảnh khuôn mặt", type="numpy")
                        classify_btn = gr.Button("🔍 Phân tích", variant="primary")
                    with gr.Column(scale=1):
                        result_text = gr.Markdown(label="Kết quả")
                        detected_image = gr.Image(label="Face Detection")

                gr.Markdown("### 📈 Điện tâm đồ (ECG) mô phỏng")
                ecg_output = gr.Image(label="ECG Waveform", height=250)

                classify_btn.click(
                    fn=predict_face,
                    inputs=[input_image],
                    outputs=[result_text, detected_image, ecg_output],
                )

            # Tab 2: Manual Simulation
            with gr.TabItem("🫀 Mô Phỏng Tim"):
                with gr.Row():
                    with gr.Column(scale=1):
                        bpm_slider = gr.Slider(30, 180, value=72, step=1, label="BPM (Nhịp tim/phút)")
                        condition_dd = gr.Dropdown(
                            choices=["Bình thường", "Bất thường", "Nhồi máu cơ tim"],
                            value="Bình thường", label="Trạng thái tim",
                        )
                        sim_btn = gr.Button("▶️ Mô phỏng", variant="primary")
                    with gr.Column(scale=1):
                        sim_info = gr.Markdown()

                sim_ecg = gr.Image(label="ECG Waveform", height=250)
                sim_btn.click(fn=simulate_heart, inputs=[bpm_slider, condition_dd], outputs=[sim_info, sim_ecg])

            # Tab 3: Comparison
            with gr.TabItem("📊 So Sánh"):
                compare_btn = gr.Button("🔄 So sánh 3 trạng thái", variant="primary")
                compare_text = gr.Markdown()
                compare_img = gr.Image(label="ECG So sánh", height=400)
                compare_btn.click(fn=compare_conditions, inputs=[], outputs=[compare_text, compare_img])

        gr.Markdown("---\n*Đồ án tốt nghiệp — Nhận dạng triệu chứng tim qua khuôn mặt | SynPAIN Model*")

    return app


if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
    )
