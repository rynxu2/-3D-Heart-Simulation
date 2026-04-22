"""FastAPI backend — REST API for heart condition prediction and 3D simulation."""

import os
import uuid
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.config import Config, load_app_config, load_heart_config
from src.data.face_detector import FaceDetector
from src.heart_simulation.heartbeat_engine import HeartbeatEngine
from src.heart_simulation.heart_model_anatomy import AnatomicalHeartModel
from src.heart_simulation.heart_renderer import HeartRenderer
from src.data.dataset import LABEL_NAMES


# ── App Setup ──────────────────────────────────────────────
app_config = load_app_config()
config = Config.from_yaml()

app = FastAPI(
    title="🫀 Heart Condition Detection API",
    description="Face image → AI classification → 3D Heart simulation",
    version="1.0.0",
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
_face_detector: Optional[FaceDetector] = None
_anatomy_model: Optional[AnatomicalHeartModel] = None
_renderer: Optional[HeartRenderer] = None


def get_face_detector() -> FaceDetector:
    global _face_detector
    if _face_detector is None:
        _face_detector = FaceDetector(target_size=config.image_size)
    return _face_detector


def get_anatomy_model() -> AnatomicalHeartModel:
    global _anatomy_model
    if _anatomy_model is None:
        _anatomy_model = AnatomicalHeartModel()
    return _anatomy_model


def get_renderer() -> HeartRenderer:
    global _renderer
    if _renderer is None:
        _renderer = HeartRenderer(output_dir=str(OUTPUT_DIR))
    return _renderer


# ── Health Check ───────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    import torch
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


# ── Predict from Image ────────────────────────────────────
@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    """Upload face image → classification result + BPM."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (jpg/png)")

    # Save uploaded file
    file_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename or "img.jpg").suffix
    save_path = UPLOAD_DIR / f"{file_id}{ext}"

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Read image
    image = cv2.imread(str(save_path))
    if image is None:
        raise HTTPException(400, "Cannot read image file")

    # Face detection
    detector = get_face_detector()
    faces = detector.detect_faces(image)

    if not faces:
        return JSONResponse(content={
            "face_detected": False,
            "message": "Không phát hiện khuôn mặt trong ảnh",
        })

    # Demo prediction (replace with real model in production)
    result = _demo_prediction()
    result["face_detected"] = True
    result["face_count"] = len(faces)
    result["file_id"] = file_id

    return JSONResponse(content=result)


# ── Heart Simulation Params ───────────────────────────────
@app.get("/api/heart-params/{label}")
async def get_heart_params(label: str, bpm: Optional[int] = None):
    """Get heart animation parameters for a condition label."""
    valid = {"normal", "abnormal", "infarction"}
    if label not in valid:
        raise HTTPException(400, f"Invalid label. Must be one of: {valid}")

    bpm_info = config.label_to_bpm.get(label)
    if bpm_info is None:
        raise HTTPException(500, f"No BPM config for label: {label}")

    actual_bpm = bpm or ((bpm_info.min + bpm_info.max) // 2)
    engine = HeartbeatEngine(bpm=actual_bpm, pattern=bpm_info.pattern)

    anatomy = get_anatomy_model()
    animation_config = anatomy.get_animation_config(engine, condition=label)

    return JSONResponse(content=animation_config)


# ── ECG Waveform ──────────────────────────────────────────
@app.get("/api/ecg/{label}")
async def get_ecg_waveform(label: str, bpm: Optional[int] = None, duration: float = 5.0):
    """Generate and return ECG waveform image for a condition."""
    valid = {"normal", "abnormal", "infarction"}
    if label not in valid:
        raise HTTPException(400, f"Invalid label. Must be one of: {valid}")

    bpm_info = config.label_to_bpm.get(label)
    actual_bpm = bpm or ((bpm_info.min + bpm_info.max) // 2)

    engine = HeartbeatEngine(bpm=actual_bpm, pattern=bpm_info.pattern)
    renderer = get_renderer()

    output_path = OUTPUT_DIR / f"ecg_{label}_{actual_bpm}bpm.png"
    renderer.render_ecg_waveform(engine, label, duration=duration, output_path=output_path)

    return FileResponse(str(output_path), media_type="image/png")


# ── ECG Comparison ────────────────────────────────────────
@app.get("/api/ecg-comparison")
async def get_ecg_comparison(duration: float = 5.0):
    """Generate side-by-side ECG comparison of all 3 conditions."""
    renderer = get_renderer()
    output_path = OUTPUT_DIR / "ecg_comparison.png"
    renderer.render_comparison(duration=duration, output_path=output_path)
    return FileResponse(str(output_path), media_type="image/png")


# ── Demo Prediction ───────────────────────────────────────
def _demo_prediction() -> dict:
    """Simulated prediction for demo (replace with real model)."""
    import random
    labels = ["normal", "abnormal", "infarction"]
    chosen = random.choice(labels)

    bpm_info = config.label_to_bpm.get(chosen)
    probs = {l: round(random.uniform(0.02, 0.15), 3) for l in labels}
    probs[chosen] = round(random.uniform(0.7, 0.95), 3)

    return {
        "label": chosen,
        "label_vi": {"normal": "Bình thường", "abnormal": "Bất thường", "infarction": "Nhồi máu cơ tim"}[chosen],
        "confidence": probs[chosen],
        "all_probs": probs,
        "bpm": {
            "min": bpm_info.min if bpm_info else 60,
            "max": bpm_info.max if bpm_info else 80,
            "pattern": bpm_info.pattern if bpm_info else "regular",
        },
    }


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
