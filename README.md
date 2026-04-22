# 🫀 Heart Condition Detection & 3D Simulation

Nhận dạng trạng thái tim mạch từ khuôn mặt bằng AI, kết hợp mô phỏng 3D quả tim.

## Pipeline

```
Ảnh khuôn mặt → Face Detection → CNN Classification → Nhãn tim → 3D Heart Simulation
```

## 3 Nhãn phân loại

| Nhãn | BPM | Mô tả |
|------|-----|-------|
| Bình thường | 60-80 | Tim khỏe mạnh |
| Bất thường | <60 / >100 | Rối loạn nhịp |
| Nhồi máu cơ tim | 90-130 | Myocardial Infarction |

## Cài đặt

```bash
# Tạo môi trường
conda create -n heart3d python=3.10
conda activate heart3d

# Cài PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Cài dependencies
pip install -r requirements.txt
```

## Sử dụng

```bash
# Chạy Web UI
python -m src.web.gradio_ui

# Training model
python scripts/run_training.py

# Evaluation
python -m src.evaluation.metrics

# FastAPI
python -m src.web.app
```

## Tech Stack

- **AI:** PyTorch, EfficientNet-B4, ResNet3D-18
- **Face:** MediaPipe
- **3D:** Three.js, VTK, trimesh
- **Web:** Gradio, FastAPI
