# 🫀 Đồ Án: Nhận Dạng Triệu Chứng Tim Qua Khuôn Mặt & Mô Phỏng 3D Tim

## Overview

**Đề tài:** Nhận dạng trạng thái tim mạch từ khuôn mặt bằng AI, kết hợp mô phỏng 3D quả tim theo nhịp đập tương ứng.

**Bối cảnh:** Đồ án tốt nghiệp | RTX 3080 (10GB VRAM) | Python

**Luồng xử lý chính:**
```
Khuôn mặt (ảnh/video) → AI phân loại → Nhãn tim mạch → Mô phỏng 3D tim đập
```

**3 Nhãn phân loại:**
| Nhãn | Mô tả | Nhịp tim (BPM) | Biểu hiện 3D |
|------|--------|----------------|--------------|
| **Bình thường** | Tim khỏe mạnh | 60-80 BPM | Đập đều, nhịp nhàng |
| **Bất thường (Arrhythmia)** | Rối loạn nhịp | <60 hoặc >100 BPM | Đập không đều, loạn nhịp |
| **Nhồi máu cơ tim** | Myocardial Infarction | 90-120+ BPM | Đập nhanh, vùng tổn thương đỏ |

**3 Mức mô hình 3D tim (user tự chọn):**
- **A) Đơn giản:** Hình tim co bóp theo BPM
- **B) Giải phẫu:** Tim 4 buồng, van tim, animation y khoa
- **C) Y tế:** Từ CT/MRI scan, mô phỏng chính xác

---

## Success Criteria

| # | Tiêu chí | Đo lường |
|---|----------|----------|
| 1 | Dataset khuôn mặt đủ 3 nhãn | ≥500 ảnh/nhãn (tổng ≥1500) |
| 2 | Model phân loại chính xác | Accuracy ≥85%, F1 ≥0.80 |
| 3 | 3D tim Level A hoạt động | Animation co bóp theo BPM |
| 4 | 3D tim Level B hoạt động | Tim giải phẫu + animation |
| 5 | 3D tim Level C hoạt động | Medical model + simulation |
| 6 | Web UI hoàn chỉnh | Upload ảnh → phân loại → xem 3D |
| 7 | Evaluation đầy đủ | Confusion matrix, ROC, metrics |
| 8 | Báo cáo đồ án | Hoàn chỉnh theo chuẩn trường |

---

## Tech Stack

| Thành phần | Công nghệ | Lý do |
|------------|-----------|-------|
| **Language** | Python 3.10+ | Hệ sinh thái ML/DL |
| **Deep Learning** | PyTorch 2.x + CUDA 12.1 | GPU, community lớn |
| **CNN Model** | EfficientNet-B4 / ResNet-50 | Proven cho image classification |
| **3D CNN (video)** | ResNet3D / SlowFast | Temporal facial analysis |
| **Face Detection** | MediaPipe / MTCNN | Crop khuôn mặt chính xác |
| **Data Augmentation** | Albumentations | Tăng dữ liệu training |
| **Synthetic Faces** | Stable Diffusion / StyleGAN | Tạo ảnh giả lập |
| **3D Heart (A)** | Three.js + custom mesh | Đơn giản, web-based |
| **3D Heart (B)** | Three.js + Sketchfab model | Giải phẫu, free 3D model |
| **3D Heart (C)** | VTK / PyVista + medical data | CT/MRI segmentation |
| **Web Backend** | FastAPI | Async, nhanh |
| **Web UI** | Gradio + Three.js | ML-friendly + 3D viewer |
| **Visualization** | Matplotlib, Plotly | Biểu đồ evaluation |

---

## File Structure

```
heart-face-3d/
├── docs/
│   └── PLAN-3d-model-gen.md
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   │
│   ├── data/                          # Data pipeline
│   │   ├── __init__.py
│   │   ├── dataset.py                 # PyTorch Dataset class
│   │   ├── data_collector.py          # Thu thập ảnh khuôn mặt
│   │   ├── face_detector.py           # Detect + crop face
│   │   ├── augmentation.py            # Data augmentation
│   │   ├── synthetic_generator.py     # Tạo ảnh synthetic
│   │   └── label_manager.py           # Quản lý nhãn
│   │
│   ├── models/                        # AI Models
│   │   ├── __init__.py
│   │   ├── classifier_2d.py           # 2D CNN (EfficientNet/ResNet)
│   │   ├── classifier_3d.py           # 3D CNN (video-based)
│   │   ├── trainer.py                 # Training loop
│   │   └── predictor.py               # Inference
│   │
│   ├── heart_simulation/              # 3D Heart
│   │   ├── __init__.py
│   │   ├── heart_model_simple.py      # Level A: đơn giản
│   │   ├── heart_model_anatomy.py     # Level B: giải phẫu
│   │   ├── heart_model_medical.py     # Level C: y tế
│   │   ├── heartbeat_engine.py        # BPM → animation params
│   │   └── heart_renderer.py          # Export 3D scenes
│   │
│   ├── evaluation/                    # Đánh giá
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Accuracy, F1, ROC, AUC
│   │   └── visualizer.py             # Confusion matrix, charts
│   │
│   ├── web/                           # Web UI
│   │   ├── __init__.py
│   │   ├── app.py                     # FastAPI
│   │   ├── gradio_ui.py               # Gradio interface
│   │   ├── static/
│   │   │   ├── heart_viewer.js        # Three.js 3D heart
│   │   │   ├── heart_simple.glb       # Model A
│   │   │   ├── heart_anatomy.glb      # Model B
│   │   │   └── style.css
│   │   └── templates/
│   │       └── index.html
│   │
│   └── utils/
│       ├── __init__.py
│       ├── gpu_manager.py
│       └── logger.py
│
├── data/
│   ├── raw/                           # Ảnh gốc
│   │   ├── normal/                    # Bình thường
│   │   ├── abnormal/                  # Bất thường
│   │   └── infarction/                # Nhồi máu
│   ├── processed/                     # Ảnh đã xử lý
│   ├── synthetic/                     # Ảnh AI-generated
│   └── medical_3d/                    # CT/MRI data (Level C)
│
├── configs/
│   ├── train_config.yaml
│   ├── heart_config.yaml
│   └── app_config.yaml
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── tests/
│   ├── test_dataset.py
│   ├── test_classifier.py
│   └── test_heart_sim.py
│
├── models/                            # Saved models
│   └── checkpoints/
│
├── scripts/
│   ├── setup_env.py
│   ├── download_models.py
│   └── run_training.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Task Breakdown

### Phase 1: Nghiên Cứu & Môi Trường (Tuần 1-2)

#### Task 1.1: Nghiên cứu lý thuyết
- **Agent:** `project-planner`
- **INPUT:** Papers về facial expression analysis, heart disease detection, 3D heart modeling
- **OUTPUT:** Tổng hợp lý thuyết: CNN, 3D CNN, facial landmarks, heart physiology
- **VERIFY:** Viết được chương cơ sở lý thuyết
- **Priority:** P0

#### Task 1.2: Cài đặt môi trường
- **Agent:** `backend-specialist`
- **INPUT:** RTX 3080, Windows, Python 3.10+
- **OUTPUT:** Conda env đầy đủ dependencies
- **VERIFY:** `torch.cuda.is_available() == True`
- **Priority:** P0

---

### Phase 2: Thu Thập & Xử Lý Dữ Liệu (Tuần 2-4)

#### Task 2.1: Thu thập khuôn mặt thật
- **Agent:** `backend-specialist`
- **INPUT:** Camera/webcam, tình nguyện viên
- **OUTPUT:** `data_collector.py` — chụp ảnh khuôn mặt, phân loại thư mục
- **VERIFY:** ≥200 ảnh thật cho mỗi nhãn
- **Priority:** P0

#### Task 2.2: Tạo ảnh synthetic
- **Agent:** `backend-specialist`
- **INPUT:** Text prompts mô tả biểu hiện đau tim
- **OUTPUT:** `synthetic_generator.py` — dùng Stable Diffusion/StyleGAN tạo ảnh giả lập
- **VERIFY:** ≥300 ảnh synthetic/nhãn, chất lượng đủ realistic
- **Priority:** P0
- **Chi tiết prompts:**
  - Normal: "healthy person, relaxed face, calm expression"
  - Abnormal: "person with chest discomfort, slight grimace, sweating"
  - Infarction: "person in severe chest pain, distressed face, pale skin, sweating"

#### Task 2.3: Tìm dataset có sẵn
- **Agent:** `backend-specialist`
- **INPUT:** Tìm trên Kaggle, PhysioNet, medical datasets
- **OUTPUT:** Download + merge vào dataset chung
- **VERIFY:** Tổng dataset ≥1500 ảnh (≥500/nhãn)
- **Priority:** P1

#### Task 2.4: Face Detection & Preprocessing
- **Agent:** `backend-specialist`
- **INPUT:** Ảnh raw
- **OUTPUT:** `face_detector.py` — detect, crop, align, resize 224×224
- **VERIFY:** Face detected >95% ảnh, output đồng nhất
- **Priority:** P0
- **Dependencies:** Task 2.1, 2.2

#### Task 2.5: Data Augmentation & Labeling
- **Agent:** `backend-specialist`
- **INPUT:** Ảnh đã crop
- **OUTPUT:** `augmentation.py` + `label_manager.py` — flip, rotate, brightness, CSV labels
- **VERIFY:** Dataset tăng 3-5x, labels chính xác
- **Priority:** P0
- **Dependencies:** Task 2.4

#### Task 2.6: PyTorch Dataset Class
- **Agent:** `backend-specialist`
- **INPUT:** Processed images + labels
- **OUTPUT:** `dataset.py` — train/val/test split (70/15/15)
- **VERIFY:** DataLoader chạy, batch shape đúng
- **Priority:** P0
- **Dependencies:** Task 2.5

---

### Phase 3: Training Model AI (Tuần 4-7)

#### Task 3.1: 2D CNN Classifier (Recommended - tối ưu nhất)
- **Agent:** `backend-specialist`
- **INPUT:** Dataset ảnh 2D khuôn mặt
- **OUTPUT:** `classifier_2d.py` — EfficientNet-B4 fine-tuned, 3 classes
- **VERIFY:** Val accuracy ≥85%, không overfit
- **Priority:** P0
- **Dependencies:** Task 2.6
- **Chi tiết:**
  - Backbone: EfficientNet-B4 (pretrained ImageNet)
  - Replace head: → FC(1792, 512) → ReLU → Dropout(0.3) → FC(512, 3)
  - Loss: CrossEntropyLoss (class weights nếu imbalanced)
  - Optimizer: AdamW, lr=1e-4, weight_decay=1e-5
  - Scheduler: CosineAnnealingLR
  - Epochs: 50, early stopping patience=10
  - VRAM: ~4GB → ✅ RTX 3080

#### Task 3.2: 3D CNN Classifier (Video-based)
- **Agent:** `backend-specialist`
- **INPUT:** Video clips khuôn mặt (16-32 frames)
- **OUTPUT:** `classifier_3d.py` — ResNet3D-18, temporal analysis
- **VERIFY:** Val accuracy ≥80%
- **Priority:** P1
- **Dependencies:** Task 2.6
- **Chi tiết:**
  - Model: torchvision.models.video.r3d_18 (pretrained Kinetics-400)
  - Input: (B, 3, 16, 224, 224) — 16 frames
  - Captures micro-expressions over time
  - VRAM: ~6GB → ✅ RTX 3080

#### Task 3.3: Training Pipeline
- **Agent:** `backend-specialist`
- **INPUT:** Model + Dataset
- **OUTPUT:** `trainer.py` — training loop, logging, checkpointing
- **VERIFY:** Training curves smooth, best model saved
- **Priority:** P0
- **Dependencies:** Task 3.1
- **Features:**
  - Mixed precision (FP16) training
  - TensorBoard logging
  - Best model checkpoint saving
  - Learning rate warmup

#### Task 3.4: Inference Module
- **Agent:** `backend-specialist`
- **INPUT:** Trained model + new image
- **OUTPUT:** `predictor.py` — predict label + confidence + BPM mapping
- **VERIFY:** Inference <100ms/ảnh, output đúng format
- **Priority:** P0
- **Dependencies:** Task 3.3
- **Label → BPM Mapping:**
  ```python
  LABEL_TO_BPM = {
      "normal": {"min": 60, "max": 80, "pattern": "regular"},
      "abnormal": {"min": 40, "max": 120, "pattern": "irregular"},
      "infarction": {"min": 90, "max": 130, "pattern": "rapid_irregular"}
  }
  ```

---

### Phase 4: Mô Phỏng 3D Tim (Tuần 7-10)

#### Task 4.1: Heartbeat Engine
- **Agent:** `backend-specialist`
- **INPUT:** BPM value + pattern type
- **OUTPUT:** `heartbeat_engine.py` — tính toán animation parameters
- **VERIFY:** Animation params chính xác cho từng BPM
- **Priority:** P0
- **Dependencies:** Task 3.4
- **Chi tiết:**
  - Convert BPM → frequency (Hz)
  - Generate contraction/relaxation cycle
  - Systole (co) = 1/3 cycle, Diastole (giãn) = 2/3 cycle
  - Irregular pattern: random jitter ±15% cho abnormal
  - Infarction: highlight affected region + faster rate

#### Task 4.2: Level A — Tim Đơn Giản (Three.js)
- **Agent:** `frontend-specialist`
- **INPUT:** Heart mesh đơn giản + BPM params
- **OUTPUT:** `heart_model_simple.py` + `heart_viewer.js`
- **VERIFY:** Tim co bóp đều theo BPM trên web
- **Priority:** P0
- **Dependencies:** Task 4.1
- **Chi tiết:**
  - Sphere/ellipsoid deformation → co bóp
  - Color: đỏ (normal), vàng (abnormal), đỏ đậm+đen (infarction)
  - BPM counter hiển thị trên UI
  - Scale animation: 1.0 → 0.85 → 1.0 (systole-diastole)

#### Task 4.3: Level B — Tim Giải Phẫu
- **Agent:** `frontend-specialist`
- **INPUT:** Anatomical heart .glb từ Sketchfab (free)
- **OUTPUT:** `heart_model_anatomy.py` — load + animate
- **VERIFY:** 4 buồng tim co bóp, van mở/đóng
- **Priority:** P1
- **Dependencies:** Task 4.1
- **Chi tiết:**
  - Download free anatomical heart model (.glb)
  - Morph targets hoặc bone animation
  - Separate animation cho atria vs ventricles
  - Highlight vùng tổn thương cho infarction (shader effect)

#### Task 4.4: Level C — Tim Y Tế
- **Agent:** `backend-specialist`
- **INPUT:** CT/MRI heart data (public dataset)
- **OUTPUT:** `heart_model_medical.py` — segment + render
- **VERIFY:** Mesh từ medical data render thành công
- **Priority:** P2
- **Dependencies:** Task 4.1
- **Chi tiết:**
  - Dataset: MMWHS hoặc ACDC cardiac MRI
  - Segmentation: TotalSegmentator hoặc nnU-Net
  - Convert segmentation → mesh (marching cubes)
  - Render với PyVista/VTK hoặc export .glb cho Three.js

#### Task 4.5: Heart Renderer & Export
- **Agent:** `backend-specialist`
- **INPUT:** Heart model + animation
- **OUTPUT:** `heart_renderer.py` — render video/GIF, export scene
- **VERIFY:** Video output smooth, 30fps
- **Priority:** P1
- **Dependencies:** Task 4.2, 4.3

---

### Phase 5: Web UI (Tuần 10-12)

#### Task 5.1: FastAPI Backend
- **Agent:** `backend-specialist`
- **INPUT:** Predictor + Heart simulation modules
- **OUTPUT:** `app.py` + `api_routes.py`
- **VERIFY:** API endpoints hoạt động
- **Priority:** P1
- **Dependencies:** Phase 3, 4
- **Endpoints:**
  - `POST /api/predict` — Upload ảnh → {label, confidence, bpm}
  - `POST /api/predict-video` — Upload video → {label, confidence, bpm}
  - `GET /api/heart-params/{label}` — Lấy animation params
  - `GET /api/health` — Health check

#### Task 5.2: Gradio Interface
- **Agent:** `frontend-specialist`
- **INPUT:** API endpoints
- **OUTPUT:** `gradio_ui.py` — 3 tabs
- **VERIFY:** Upload → phân loại → xem 3D tim
- **Priority:** P1
- **Dependencies:** Task 5.1
- **Layout:**
  - **Tab 1: Phân Loại Khuôn Mặt**
    - Upload ảnh/video khuôn mặt
    - Hiện kết quả: nhãn + confidence + BPM
    - Hiện face detection bounding box
  - **Tab 2: Mô Phỏng 3D Tim**
    - 3D viewer (Three.js embedded)
    - Chọn level (A/B/C)
    - Slider BPM thủ công để test
    - Auto-link từ Tab 1 kết quả
  - **Tab 3: So Sánh**
    - Hiện 3 trạng thái tim cạnh nhau
    - Normal vs Abnormal vs Infarction

#### Task 5.3: Three.js Heart Viewer
- **Agent:** `frontend-specialist`
- **INPUT:** Heart .glb models
- **OUTPUT:** `heart_viewer.js` — interactive 3D
- **VERIFY:** Orbit controls, BPM animation, responsive
- **Priority:** P1
- **Dependencies:** Task 5.2

---

### Phase 6: Evaluation (Tuần 12-13)

#### Task 6.1: Classification Metrics
- **Agent:** `backend-specialist`
- **INPUT:** Test set predictions
- **OUTPUT:** `metrics.py` — Accuracy, Precision, Recall, F1, ROC-AUC
- **VERIFY:** Metrics tính đúng, visualize rõ ràng
- **Priority:** P0
- **Dependencies:** Phase 3
- **Metrics chi tiết:**

| Metric | Mục đích |
|--------|----------|
| Accuracy | Tỷ lệ đúng tổng thể |
| Precision/Recall/F1 per class | Chất lượng từng nhãn |
| Confusion Matrix | Ma trận nhầm lẫn |
| ROC-AUC (multi-class) | Phân biệt các class |
| Inference Time | Tốc độ dự đoán |
| Model Size | Kích thước model (MB) |

#### Task 6.2: So sánh 2D CNN vs 3D CNN
- **Agent:** `backend-specialist`
- **INPUT:** Kết quả cả 2 models
- **OUTPUT:** `visualizer.py` — bảng so sánh + charts
- **VERIFY:** Charts rõ ràng, kết luận hợp lý
- **Priority:** P1
- **Dependencies:** Task 6.1

---

### Phase 7: Documentation (Tuần 13-15)

#### Task 7.1: README & Docs
- **Agent:** `documentation-writer`
- **OUTPUT:** README.md + hướng dẫn cài đặt
- **Priority:** P1

#### Task 7.2: Báo cáo đồ án
- **Agent:** `documentation-writer`
- **Priority:** P0
- **Cấu trúc:**
  1. Giới thiệu (vấn đề, mục tiêu)
  2. Cơ sở lý thuyết (CNN, 3D CNN, facial analysis, heart physiology)
  3. Thu thập & xử lý dữ liệu (3 nguồn data)
  4. Mô hình AI phân loại (2D CNN + 3D CNN)
  5. Mô phỏng 3D tim (3 levels)
  6. Hệ thống Web (kiến trúc, UI)
  7. Thực nghiệm & đánh giá
  8. Kết luận & hướng phát triển

---

## Dependency Graph

```
Task 1.1 (Nghiên cứu) ─────────────────────────────────┐
Task 1.2 (Môi trường) ──────────────────────────────────┤
                                                         │
Task 2.1 (Thu thập ảnh thật) ◄───────────────────────────┘
Task 2.2 (Synthetic) ◄──────────────────────────────────┘
Task 2.3 (Dataset có sẵn) ◄─────────────────────────────┘
    │
Task 2.4 (Face detect) ◄── 2.1 + 2.2 + 2.3
    │
Task 2.5 (Augment + Label) ◄── 2.4
    │
Task 2.6 (PyTorch Dataset) ◄── 2.5
    │
    ├──→ Task 3.1 (2D CNN) ──→ Task 3.3 (Trainer) ──→ Task 3.4 (Predictor)
    └──→ Task 3.2 (3D CNN) ──┘                              │
                                                              │
Task 4.1 (Heartbeat Engine) ◄─────────────────────────────────┘
    │
    ├──→ Task 4.2 (Level A: Simple)
    ├──→ Task 4.3 (Level B: Anatomy)
    ├──→ Task 4.4 (Level C: Medical)
    └──→ Task 4.5 (Renderer)
                │
Task 5.1 (FastAPI) ◄── Phase 3 + Phase 4
    ├──→ Task 5.2 (Gradio UI)
    └──→ Task 5.3 (Three.js Viewer)
                │
Task 6.1 (Metrics) ◄── Phase 3
    └──→ Task 6.2 (Comparison)
                │
Task 7.1 (README) ◄── All
Task 7.2 (Báo cáo) ◄── All
```

---

## Timeline (15 Tuần)

| Tuần | Phase | Milestone |
|------|-------|-----------|
| 1-2 | Phase 1+2a | ✅ Môi trường + bắt đầu thu thập data |
| 2-4 | Phase 2 | ✅ Dataset ≥1500 ảnh, preprocessed |
| 4-6 | Phase 3a | ✅ 2D CNN trained, accuracy ≥85% |
| 6-7 | Phase 3b | ✅ 3D CNN trained + inference module |
| 7-9 | Phase 4a | ✅ Heart Level A+B hoạt động |
| 9-10 | Phase 4b | ✅ Heart Level C + renderer |
| 10-12 | Phase 5 | ✅ Web UI hoàn chỉnh |
| 12-13 | Phase 6 | ✅ Evaluation + so sánh |
| 13-15 | Phase 7 | ✅ Báo cáo hoàn chỉnh |

---

## Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| Thiếu data khuôn mặt thật | Cao | Bù bằng synthetic + augmentation |
| Accuracy thấp (<80%) | Cao | Thử nhiều backbone, ensemble |
| Medical 3D data khó tìm | Trung bình | Fallback sang Level A/B |
| VRAM không đủ 3D CNN | Thấp | Giảm batch size, FP16 |
| Three.js animation phức tạp | Trung bình | Bắt đầu Level A, nâng dần |

---

## Phase X: Verification Checklist

- [ ] Dataset ≥1500 ảnh, 3 nhãn cân bằng
- [ ] 2D CNN accuracy ≥85% trên test set
- [ ] 3D CNN hoạt động (accuracy reported)
- [ ] Confusion matrix + ROC curves
- [ ] Heart Level A: co bóp theo BPM ✓
- [ ] Heart Level B: giải phẫu animate ✓
- [ ] Heart Level C: medical render ✓
- [ ] Web UI: upload ảnh → nhãn → 3D tim
- [ ] So sánh 2D vs 3D CNN
- [ ] Unit tests pass
- [ ] README đầy đủ
- [ ] Báo cáo đồ án hoàn chỉnh
- [ ] Demo video

---

## Hardware Requirements

| Task | VRAM | Time | Note |
|------|------|------|------|
| EfficientNet-B4 training | ~4GB | ~30min/epoch | ✅ RTX 3080 |
| ResNet3D-18 training | ~6GB | ~45min/epoch | ✅ RTX 3080 |
| Stable Diffusion (synthetic) | ~6GB | ~5s/ảnh | ✅ RTX 3080 |
| Face Detection (MediaPipe) | ~1GB | real-time | ✅ CPU cũng được |
| Three.js rendering | GPU shared | real-time | ✅ Browser |
| VTK/PyVista (Level C) | ~2GB | ~10s/render | ✅ RTX 3080 |
