# PLAN: Chuyển giao diện Web sang JS Framework + GLB 3D Model

## Mô tả

Chuyển giao diện web từ **Gradio (Python)** sang **Vite + Vanilla JS + Three.js**, sử dụng file `.glb` tim y tế thật của người dùng (`human_heart_realistic_anatomical_model.glb`, 2.4MB) thay cho hình học cơ bản. Backend giữ nguyên **FastAPI** làm REST API, frontend gọi API qua `fetch()`.

## Hiện trạng

| Component | Hiện tại | Vấn đề |
|-----------|----------|--------|
| Frontend | Gradio (Python) | Không render được 3D, chỉ hiện JSON, Gradio 6.0 gây lỗi |
| 3D Heart | `gr.Code(json)` | Không có visualization thực tế |
| Backend API | FastAPI (`app.py`) | Hoạt động tốt, có endpoints `/api/predict`, `/api/heart-params`, `/api/ecg` |
| GLB Model | `test/human_heart_realistic_anatomical_model.glb` | Có sẵn, chưa tích hợp |
| Test prototype | `test/index.js` | Đã có animation nhịp đập hoạt động trên Vite |

## Kiến trúc mới

```
┌──────────────────────────────────────────────────────┐
│  Frontend (Vite + Vanilla JS + Three.js)             │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Upload   │  │ Results  │  │ 3D Heart Viewer  │   │
│  │ Panel    │→ │ Panel    │→ │ (Three.js + GLB) │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
│                                      ↕               │
│                              ┌──────────────┐        │
│                              │ ECG Canvas   │        │
│                              └──────────────┘        │
└──────────────────┬───────────────────────────────────┘
                   │ fetch('/api/...')
┌──────────────────▼───────────────────────────────────┐
│  Backend (FastAPI - Python)                          │
│  ┌──────────┐  ┌────────────┐  ┌──────────────┐     │
│  │ /predict │  │ /heart-    │  │ /ecg/{label} │     │
│  │ (model)  │  │  params    │  │ (waveform)   │     │
│  └──────────┘  └────────────┘  └──────────────┘     │
│       ↓                                              │
│  SynPAIN Model (synpain_best_model.pth)             │
└──────────────────────────────────────────────────────┘
```

## Quyết định thiết kế

> [!IMPORTANT]
> ### Tại sao Vite + Vanilla JS (không dùng React/Vue)?
> 1. **Bạn đã có prototype** `test/index.js` chạy tốt trên Vite + Three.js
> 2. **Đồ án tốt nghiệp** — đơn giản, không cần SPA framework phức tạp
> 3. **Three.js performance** — ít overhead, direct DOM manipulation
> 4. **Tương thích** với code hiện có trong `test/`

## Open Questions

> [!WARNING]
> **Q1:** Bạn muốn **giữ nguyên 2 nhãn SynPAIN** (Pain/NoPain) hay cần hiển thị 3 trạng thái (normal/abnormal/infarction) trên UI?
>
> **Q2:** File GLB chỉ có ở `test/` — bạn muốn copy nó vào `frontend/public/models/` hay giữ ở vị trí khác?
>
> **Q3:** Backend FastAPI hiện vẫn dùng `_demo_prediction()` (mock random). Cần sửa sang dùng model thật luôn không?

---

## Proposed Changes

### Component 1: Frontend (Vite + Vanilla JS)

> Tạo folder `frontend/` mới (tách biệt khỏi Python backend)

#### [NEW] `frontend/package.json`
- Vite dev server
- Dependencies: `three` (^0.184.0), `vite`

#### [NEW] `frontend/index.html`
- Layout chính: 3 panels (Upload | Results | 3D Viewer)
- Responsive, dark theme medical UI
- Import `main.js` as module

#### [NEW] `frontend/src/main.js`
- App entry point
- Initialize UI panels, event listeners
- Call API endpoints via `fetch()`

#### [NEW] `frontend/src/heart-viewer.js`
- Three.js scene setup (từ code `test/index.js` hiện có)
- Load file `.glb` via `GLTFLoader`
- Heartbeat animation (non-uniform scaling, torsion twist)
- Nhận params từ API: `bpm`, `pattern`, `condition`
- Thay đổi màu/glow theo condition (normal=green, abnormal=orange, infarction=red)
- Damage zone visualization cho infarction

#### [NEW] `frontend/src/ecg-renderer.js`
- Canvas-based ECG waveform (thay vì dùng matplotlib image)
- PQRST complex rendering realtime
- Animate theo BPM từ prediction result

#### [NEW] `frontend/src/api-client.js`
- `predictImage(file)` → POST `/api/predict`
- `getHeartParams(label)` → GET `/api/heart-params/{label}`
- `getECG(label)` → GET `/api/ecg/{label}`

#### [NEW] `frontend/src/styles.css`
- Dark medical UI theme (#1a1a2e, #0f3460)
- Split-panel layout (upload left, results center, 3D right)
- Responsive grid
- Animation transitions

#### [NEW] `frontend/public/models/human_heart_realistic_anatomical_model.glb`
- Copy từ `test/` → phục vụ static qua Vite

---

### Component 2: Backend FastAPI Updates

#### [MODIFY] [app.py](file:///d:/Antigravity/-3D-Heart-Simulation/src/web/app.py)
- **Replace `_demo_prediction()`** → dùng model SynPAIN thật
- Lazy-load `FaceDetector` (tránh TensorFlow import chậm)
- Lazy-load `HeartFaceClassifier2D` từ `synpain_best_model.pth`
- Map SynPAIN labels (Pain/NoPain) → heart conditions
- Cập nhật `/api/predict` trả thêm `ecg_url` và `heart_params`
- **Serve frontend build** tại `/` (optional, cho production)

#### [MODIFY] [app_config.yaml](file:///d:/Antigravity/-3D-Heart-Simulation/configs/app_config.yaml)
- Thêm `frontend.build_dir` path
- Thêm `model.num_classes: 2` cho SynPAIN

---

### Component 3: Cleanup

#### [DELETE] `test/` folder (merged vào `frontend/`)
- Code `test/index.js` sẽ được tái sử dụng trong `frontend/src/heart-viewer.js`

#### [KEEP] `src/web/gradio_ui.py`
- Giữ lại như backup, không xóa

---

## File Structure (sau khi hoàn thành)

```
frontend/
├── index.html              # Main page
├── package.json            # Vite + Three.js
├── vite.config.js          # Proxy /api → FastAPI
├── public/
│   └── models/
│       └── human_heart_realistic_anatomical_model.glb
└── src/
    ├── main.js             # App entry
    ├── heart-viewer.js     # Three.js 3D heart
    ├── ecg-renderer.js     # Canvas ECG
    ├── api-client.js       # fetch() wrapper
    └── styles.css          # Dark medical UI
```

## Task Breakdown

| # | Task | Files | Ước tính |
|---|------|-------|----------|
| 1 | Khởi tạo Vite project + copy GLB | `frontend/*` | 5 min |
| 2 | Tạo layout HTML + CSS (dark medical UI) | `index.html`, `styles.css` | 15 min |
| 3 | Port heart-viewer.js từ test/ (GLB + animation) | `heart-viewer.js` | 10 min |
| 4 | Tạo ECG canvas renderer | `ecg-renderer.js` | 10 min |
| 5 | Tạo API client + kết nối UI | `api-client.js`, `main.js` | 10 min |
| 6 | Fix FastAPI backend (real model, lazy imports) | `app.py` | 10 min |
| 7 | Vite proxy config (dev mode) | `vite.config.js` | 3 min |
| 8 | Test end-to-end | - | 5 min |

**Tổng: ~70 phút**

## Verification Plan

### Dev Testing
1. `cd frontend && npm run dev` → Vite chạy ở port 5173
2. `cd .. && python -m src.web.app` → FastAPI chạy ở port 7860
3. Vite proxy `/api/*` → `localhost:7860`
4. Upload ảnh → nhận prediction → 3D heart đập đúng BPM + ECG hiển thị

### Visual Checks
- [ ] GLB model load thành công, xoay/zoom được
- [ ] Heartbeat animation chạy đúng theo BPM
- [ ] Màu sắc thay đổi theo condition (green/orange/red)
- [ ] ECG waveform animate realtime
- [ ] Face detection overlay hiển thị
- [ ] Responsive trên mobile
