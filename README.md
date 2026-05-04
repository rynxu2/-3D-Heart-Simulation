# 🫀 3D Heart Simulation - Surgical Monitor Edition

Hệ thống phân tích biểu cảm khuôn mặt để dự đoán trạng thái tim mạch bằng AI (PainFormer Zero-shot), kết hợp mô phỏng 3D quả tim tương tác thời gian thực với giao diện Surgical Cockpit chuyên nghiệp.

## 🌟 Tính năng chính

- **AI PainFormer Zero-shot**: Phân tích `pain_score` từ khuôn mặt mà không cần quá trình training phức tạp trên dataset lớn. Tự động load ngay khi khởi động backend.
- **Biomechanical 3D Heart**: Mô phỏng sự co bóp của quả tim theo **Biểu đồ Wiggers** (Tâm nhĩ co, tâm thất tống máu, giãn đồng thể tích).
- **Bidirectional BPM Logic**: Nhịp tim được nội suy hai chiều linh hoạt dựa vào `pain_score`:
  - Có thể biểu hiện nhịp chậm (Bradycardia) hoặc nhịp nhanh (Tachycardia) ở trạng thái bất thường.
- **Surgical Cockpit UI**: Giao diện phòng mổ chuyên nghiệp, tối giản ánh sáng 3D ảo, sử dụng nền holographic grid và particle tạo chiều sâu (Parallax).

## 🚀 Pipeline hoạt động

```
Webcam/Camera → Face Detection → PainFormer (Extract 160-D Embeddings) → Tính Pain Score → Chọn dải BPM & Trạng thái → 3D Heart Simulation + ECG Waveform
```


## 📁 Cấu trúc thư mục

```
├── frontend/          # Vite + Three.js (Surgical Monitor UI)
│   ├── src/
│   │   ├── main.js          # App entry + camera logic + backoff retry
│   │   ├── heart-viewer.js  # 3D heart (GPU vertex shader, Wiggers-synced)
│   │   ├── ecg-renderer.js  # ECG waveform canvas
│   │   ├── api-client.js    # Fetch wrapper (gọi backend)
│   │   └── styles.css       # Surgical Dark Theme
│   └── index.html
│
├── backend/           # Python + FastAPI (PainFormer API)
│   ├── src/
│   │   ├── web/app.py                 # FastAPI endpoints + startup events
│   │   ├── models/painformer_zeroshot.py # Core AI logic
│   │   ├── heart_simulation/          # Logic tạo params cho 3D/ECG
│   │   └── config.py                  # Load yaml configs
│   ├── configs/       # Configurations (painformer_config.yaml)
│   └── tools/         # Chứa weights (PainFormer/painformer.pth)
│
└── docs/              # Tài liệu, kế hoạch (Plans)
```

## 🛠️ Cài đặt

Yêu cầu: `Python 3.10+` và `Node.js 18+`.

### 1. Backend
```bash
cd backend
conda create -n heart3d python=3.10
conda activate heart3d
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

*Lưu ý: Đảm bảo đã có file `backend/tools/PainFormer/painformer.pth` trước khi chạy.*

### 2. Frontend
```bash
cd frontend
npm install
```

## 🏃 Sử dụng

**Mở 2 terminal để chạy song song Backend và Frontend:**

### Terminal 1 (Chạy Backend):
```bash
cd backend
python -m src.web.app
```
*(Backend sẽ tự động load model PainFormer vào RAM/VRAM ngay lúc khởi động, sẵn sàng phục vụ không bị delay nhịp đầu).*

### Terminal 2 (Chạy Frontend):
```bash
cd frontend
npm run dev
```

Mở trình duyệt tại đường dẫn http://localhost:5173/ (hoặc theo log của Vite). Cấp quyền truy cập Camera và bắt đầu mô phỏng.

## 🏗️ Tech Stack

- **AI Backend**: FastAPI, PyTorch, timm (PainFormer Backbone).
- **Web Frontend**: Vite, React (nếu mở rộng), Vanilla JS.
- **3D Rendering**: Three.js (Tùy chỉnh Vertex Shader theo Wiggers Diagram, Post-processing Bloom).
- **Giao diện**: Surgical Cockpit / Brutalist Medical UI.
