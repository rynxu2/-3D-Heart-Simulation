"""FastAPI backend — REST API for heart condition prediction and 3D simulation."""

import os
import uuid
from pathlib import Path
from typing import Optional

import cv2
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.config import Config, load_app_config, load_heart_config
from src.heart_simulation.heartbeat_engine import HeartbeatEngine
from src.heart_simulation.heart_model_anatomy import AnatomicalHeartModel
from src.heart_simulation.heart_renderer import HeartRenderer

# ── Label mappings (3-class + binary fallback) ────────────
LABEL_3CLASS = {0: "normal", 1: "abnormal", 2: "infarction"}
LABEL_BINARY = {0: "no_pain", 1: "pain"}

LABEL_DISPLAY = {
    "normal": "Bình thường - Tim khỏe mạnh",
    "abnormal": "Bất thường - Rối loạn nhịp tim",
    "infarction": "Nhồi máu cơ tim - Nguy hiểm",
    "no_pain": "Không đau - Bình thường",
    "pain": "Đau - Bất thường",
}
LABEL_TO_CONDITION = {
    "normal": "normal", "no_pain": "normal",
    "abnormal": "abnormal",
    "infarction": "infarction", "pain": "infarction",
}
LABEL_TO_BPM = {
    "normal": {"min": 60, "max": 80, "pattern": "regular"},
    "no_pain": {"min": 60, "max": 80, "pattern": "regular"},
    "abnormal": {"min": 40, "max": 120, "pattern": "irregular"},
    "infarction": {"min": 90, "max": 130, "pattern": "rapid_irregular"},
    "pain": {"min": 90, "max": 130, "pattern": "rapid_irregular"},
}

# ── App Setup ──────────────────────────────────────────────
app_config = load_app_config()
config = Config.from_yaml()

app = FastAPI(
    title="🫀 Heart Condition Detection API",
    description="Face image → AI classification → 3D Heart simulation",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("data/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global Instances (lazy) ────────────────────────────────
_face_detector = None
_model = None
_transform = None
_anatomy_model = None
_renderer = None


def get_face_detector():
    global _face_detector
    if _face_detector is None:
        from src.data.face_detector import FaceDetector
        _face_detector = FaceDetector(target_size=config.image_size)
        logger.info("FaceDetector initialized")
    return _face_detector


def get_model():
    """Lazy-load face classification model (2D CNN)."""
    global _model, _transform
    if _model is None:
        from src.models.classifier_2d import HeartFaceClassifier2D
        from src.data.augmentation import get_val_transforms

        model_path = Path(app_config.get("paths", {}).get(
            "model_checkpoint", "models/synpain_best_model.pth"
        ))
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Detect num_classes from checkpoint
        num_classes = 2  # default SynPAIN binary
        if model_path.exists():
            ckpt = torch.load(str(model_path), map_location=device, weights_only=True)
            state = ckpt.get("model_state_dict", ckpt)
            fc_keys = [k for k in state if "classifier" in k and "weight" in k]
            if fc_keys:
                num_classes = state[fc_keys[-1]].shape[0]

        _model = HeartFaceClassifier2D(
            backbone="efficientnet_b4", num_classes=num_classes, pretrained=False, dropout=0.3,
        ).to(device)

        if model_path.exists():
            _model.load_state_dict(state)
            logger.info(f"✅ Model loaded: {model_path} (classes={num_classes})")
        else:
            logger.warning(f"⚠️ Model not found: {model_path}")

        _model.eval()
        _transform = get_val_transforms(config.image_size)

    return _model, _transform


_model_3d = None


def get_model_3d():
    """Lazy-load 3D CNN model for video prediction."""
    global _model_3d
    if _model_3d is None:
        model_path = Path("models/checkpoints/3dcnn_best.pth")
        if not model_path.exists():
            return None

        from src.models.classifier_3d import HeartFaceClassifier3D
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model_3d = HeartFaceClassifier3D.from_checkpoint(model_path, device=device)
        logger.info(f"✅ 3D CNN loaded: {model_path}")
    return _model_3d


def get_anatomy_model():
    global _anatomy_model
    if _anatomy_model is None:
        _anatomy_model = AnatomicalHeartModel()
    return _anatomy_model


def get_renderer():
    global _renderer
    if _renderer is None:
        _renderer = HeartRenderer(output_dir=str(OUTPUT_DIR))
    return _renderer


# ── Health Check ───────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


# ── Predict from Image ────────────────────────────────────
@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    """Upload face image → AI classification → heart condition label."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (jpg/png)")

    file_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename or "img.jpg").suffix
    save_path = UPLOAD_DIR / f"{file_id}{ext}"

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    image = cv2.imread(str(save_path))
    if image is None:
        raise HTTPException(400, "Cannot read image file")

    detector = get_face_detector()
    face = detector.process_image(image)

    if face is None:
        return JSONResponse(content={
            "face_detected": False,
            "message": "Không phát hiện khuôn mặt trong ảnh",
        })

    model, transform = get_model()
    device = next(model.parameters()).device
    num_classes = model.num_classes
    label_map = LABEL_3CLASS if num_classes == 3 else LABEL_BINARY

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    tensor = transform(image=face_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    pred_label = label_map[pred_id]
    confidence = float(probs[pred_id])
    bpm_info = LABEL_TO_BPM.get(pred_label, LABEL_TO_BPM["normal"])

    return JSONResponse(content={
        "face_detected": True,
        "label": pred_label,
        "label_vi": LABEL_DISPLAY.get(pred_label, pred_label),
        "confidence": confidence,
        "all_probs": {label_map[i]: float(probs[i]) for i in range(len(probs))},
        "bpm": bpm_info,
        "condition": LABEL_TO_CONDITION.get(pred_label, pred_label),
        "file_id": file_id,
        "model_type": "2d_cnn",
        "num_classes": num_classes,
    })


# ── Predict from Video (3D CNN) ───────────────────────────
@app.post("/api/predict-video")
async def predict_video(file: UploadFile = File(...)):
    """Upload face video → 3D CNN classification → heart condition label."""
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video (mp4/avi/webm)")

    model_3d = get_model_3d()
    if model_3d is None:
        raise HTTPException(503, "3D CNN model not available. Train it first.")

    file_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename or "vid.mp4").suffix
    save_path = UPLOAD_DIR / f"{file_id}{ext}"

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    from src.models.predictor import HeartPredictor3D
    predictor = HeartPredictor3D(
        model_path=Path("models/checkpoints/3dcnn_best.pth"),
    )
    result = predictor.predict_video(save_path)

    if result.get("face_detected"):
        result["label_vi"] = LABEL_DISPLAY.get(result["label"], result["label"])
        result["condition"] = LABEL_TO_CONDITION.get(result["label"], result["label"])
        result["file_id"] = file_id

    return JSONResponse(content=result)


# ── Heart Simulation Params ───────────────────────────────
@app.get("/api/heart-params/{label}")
async def get_heart_params(label: str, bpm: Optional[int] = None):
    """Get heart animation parameters for a condition label."""
    # Support both old 3-class and new binary labels
    label_bpm_map = {**LABEL_TO_BPM}
    label_bpm_map["abnormal"] = {"min": 40, "max": 120, "pattern": "irregular"}
    label_bpm_map["infarction"] = {"min": 90, "max": 130, "pattern": "rapid_irregular"}
    label_bpm_map["normal"] = {"min": 60, "max": 80, "pattern": "regular"}

    if label not in label_bpm_map:
        raise HTTPException(400, f"Invalid label: {label}")

    info = label_bpm_map[label]
    actual_bpm = bpm or ((info["min"] + info["max"]) // 2)
    condition = LABEL_TO_CONDITION.get(label, label)

    engine = HeartbeatEngine(bpm=actual_bpm, pattern=info["pattern"])
    anatomy = get_anatomy_model()
    animation_config = anatomy.get_animation_config(engine, condition=condition)

    return JSONResponse(content=animation_config)


# ── ECG Waveform ──────────────────────────────────────────
@app.get("/api/ecg/{label}")
async def get_ecg_waveform(label: str, bpm: Optional[int] = None, duration: float = 5.0):
    """Generate and return ECG waveform image."""
    label_bpm_map = {**LABEL_TO_BPM}
    label_bpm_map["abnormal"] = {"min": 40, "max": 120, "pattern": "irregular"}
    label_bpm_map["infarction"] = {"min": 90, "max": 130, "pattern": "rapid_irregular"}
    label_bpm_map["normal"] = {"min": 60, "max": 80, "pattern": "regular"}

    if label not in label_bpm_map:
        raise HTTPException(400, f"Invalid label: {label}")

    info = label_bpm_map[label]
    actual_bpm = bpm or ((info["min"] + info["max"]) // 2)
    condition = LABEL_TO_CONDITION.get(label, label)

    engine = HeartbeatEngine(bpm=actual_bpm, pattern=info["pattern"])
    renderer = get_renderer()

    output_path = OUTPUT_DIR / f"ecg_{label}_{actual_bpm}bpm.png"
    renderer.render_ecg_waveform(engine, condition, duration=duration, output_path=output_path)

    return FileResponse(str(output_path), media_type="image/png")


# ── ECG Comparison ────────────────────────────────────────
@app.get("/api/ecg-comparison")
async def get_ecg_comparison(duration: float = 5.0):
    renderer = get_renderer()
    output_path = OUTPUT_DIR / "ecg_comparison.png"
    renderer.render_comparison(duration=duration, output_path=output_path)
    return FileResponse(str(output_path), media_type="image/png")


# ── Run Server ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    server_cfg = app_config.get("server", {})
    uvicorn.run(
        "src.web.app:app",
        host=server_cfg.get("host", "0.0.0.0"),
        port=server_cfg.get("port", 7860),
        reload=server_cfg.get("debug", True),
    )
