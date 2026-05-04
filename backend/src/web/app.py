"""FastAPI backend — REST API for PainFormer cardiac detection and 3D simulation."""

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

from src.config import load_app_config, load_heart_config
from src.heart_simulation.heartbeat_engine import HeartbeatEngine
from src.heart_simulation.heart_model_anatomy import AnatomicalHeartModel
from src.heart_simulation.heart_renderer import HeartRenderer

# ── Label mappings ─────────────────────────────────────────
LABEL_DISPLAY = {
    "normal": "Bình thường - Tim khỏe mạnh",
    "abnormal": "Bất thường - Rối loạn nhịp tim",
    "infarction": "Nhồi máu cơ tim - Nguy hiểm",
}
LABEL_TO_CONDITION = {
    "normal": "normal",
    "abnormal": "abnormal",
    "infarction": "infarction",
}
LABEL_TO_BPM = {
    "normal": {"min": 60, "max": 100, "pattern": "regular"},
    "abnormal": {"min": 40, "max": 120, "pattern": "irregular"},
    "infarction": {"min": 30, "max": 160, "pattern": "rapid_irregular"},
}

# ── App Setup ──────────────────────────────────────────────
app_config = load_app_config()

app = FastAPI(
    title="🫀 Heart Condition Detection API",
    description="Face image → PainFormer AI → 3D Heart simulation",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("data/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def preload_models():
    """Eagerly load PainFormer model at server startup."""
    logger.info("⏳ Pre-loading PainFormer model...")
    get_model_painformer()
    logger.info("✅ PainFormer ready — no cold-start delay")

# ── Global Instances (lazy) ────────────────────────────────
_painformer = None
_anatomy_model = None
_renderer = None


def get_model_painformer():
    """Lazy-load PainFormer zero-shot cardiac classifier."""
    global _painformer
    if _painformer is None:
        from src.models.painformer_zeroshot import PainFormerZeroShot
        _painformer = PainFormerZeroShot(
            backbone_path=str(Path("tools/PainFormer/painformer.pth")),
            config_path=str(Path("configs/painformer_config.yaml")),
        )
        logger.info("✅ PainFormer zero-shot classifier loaded")
    return _painformer


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


def pain_score_to_bpm(score: float) -> int:
    """Fallback BPM estimation from pain_score (bidirectional logic is in PainFormer)."""
    if score <= 0.30:
        return int(60 + (score / 0.30) * 40)  # Normal: 60-100
    elif score <= 0.475:
        t = (score - 0.30) / 0.175
        return int(60 - t * 20)  # Abnormal brady: 60→40
    elif score <= 0.65:
        t = (score - 0.475) / 0.175
        return int(100 + t * 20)  # Abnormal tachy: 100→120
    elif score <= 0.825:
        t = (score - 0.65) / 0.175
        return int(50 - t * 20)  # Infarction brady: 50→30
    else:
        t = min(1.0, (score - 0.825) / 0.175)
        return int(120 + t * 40)  # Infarction tachy: 120→160


# ── Health Check ───────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


# ── Predict with PainFormer (Zero-Shot) ────────────────────
@app.post("/api/predict-painformer")
async def predict_painformer(file: UploadFile = File(...)):
    """Upload face image → PainFormer zero-shot → cardiac condition."""
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

    classifier = get_model_painformer()
    result = classifier.classify(image)

    pain_score = result["pain_score"]
    estimated_bpm = pain_score_to_bpm(pain_score)
    label = result["label"]
    bpm_info = LABEL_TO_BPM.get(label, LABEL_TO_BPM["normal"])

    return JSONResponse(content={
        "face_detected": True,
        "label": label,
        "label_vi": LABEL_DISPLAY.get(label, label),
        "confidence": result["confidence"],
        "pain_score": pain_score,
        "all_probs": {
            "normal": max(0, 1.0 - pain_score * 1.5) if label == "normal" else 0.0,
            "abnormal": min(1.0, pain_score * 2) if label == "abnormal" else 0.0,
            "infarction": min(1.0, (pain_score - 0.5) * 2) if label == "infarction" else pain_score,
        },
        "bpm": {
            "min": bpm_info["min"],
            "max": bpm_info["max"],
            "estimated": estimated_bpm,
            "pattern": bpm_info["pattern"],
        },
        "condition": LABEL_TO_CONDITION.get(label, label),
        "file_id": file_id,
        "model_type": "painformer_zeroshot",
        "num_classes": 3,
    })


# ── Heart Simulation Params ───────────────────────────────
@app.get("/api/heart-params/{label}")
async def get_heart_params(label: str, bpm: Optional[int] = None):
    """Get heart animation parameters for a condition label."""
    if label not in LABEL_TO_BPM:
        raise HTTPException(400, f"Invalid label: {label}")

    info = LABEL_TO_BPM[label]
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
    if label not in LABEL_TO_BPM:
        raise HTTPException(400, f"Invalid label: {label}")

    info = LABEL_TO_BPM[label]
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
