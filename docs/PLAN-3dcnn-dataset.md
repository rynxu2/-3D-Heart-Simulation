# 🗺️ KẾ HOẠCH TẠO DATASET CHO MODEL 3D CNN

## Tình trạng hiện tại

| Thư mục | Nội dung | Trạng thái |
|---------|----------|------------|
| `data/raw/normal/` | Rỗng | ❌ Cần tải video |
| `data/raw/abnormal/` | Rỗng | ❌ Cần tải video |
| `data/raw/infarction/` | Rỗng | ❌ Cần tải video |
| `data/video_clips/` | Rỗng | ❌ Sẽ được sinh ra |

> [!IMPORTANT]
> **Mục tiêu:** Tối thiểu **300 video clips** (100 mỗi class) để train 3D CNN đạt kết quả chấp nhận được.
> Lý tưởng: **1000+ clips** mỗi class.

---

## PIPELINE TỔNG QUAN

```
Bước 1: Thu thập video gốc (.mp4)
   ↓
Bước 2: Trích xuất frames có khuôn mặt (.jpg)
   ↓
Bước 3: Sinh pseudo-video clips (16 frames/clip)
   ↓
Bước 4: Upload lên Kaggle + Train
   ↓
Bước 5: Download checkpoint về project
```

---

## BƯỚC 1: Thu thập video gốc

### Nguồn miễn phí (khuyên dùng)

| Nguồn | Label | Từ khóa tìm kiếm | Số lượng cần |
|-------|-------|-------------------|-------------|
| [Pexels](https://www.pexels.com/search/videos/) | normal | "calm face", "relaxed person", "meditation face" | 10-20 video |
| [Pexels](https://www.pexels.com/search/videos/) | abnormal | "worried face", "anxious person", "stressed face" | 10-20 video |
| [Pexels](https://www.pexels.com/search/videos/) | infarction | "chest pain", "heart attack", "person in pain" | 10-20 video |
| [Pixabay](https://pixabay.com/videos/) | Tất cả | Tương tự | 10-20 video |
| [Mixkit](https://mixkit.co/free-stock-video/) | Tất cả | Tương tự | 5-10 video |

### Cách tải
1. Tải video .mp4 từ các nguồn trên
2. Đặt vào đúng thư mục:
```
data/raw/
├── normal/         ← video mặt bình thường, thư giãn
├── abnormal/       ← video mặt lo lắng, khó chịu
└── infarction/     ← video mặt đau đớn, nắm ngực
```

### Webcam (bổ sung thêm)
```bash
# Tự quay webcam giả lập biểu cảm
python scripts/collect_faces.py --label normal --mode video --num 10
python scripts/collect_faces.py --label abnormal --mode video --num 10
python scripts/collect_faces.py --label infarction --mode video --num 10
```

---

## BƯỚC 2: Trích xuất frames từ video

Sau khi có video .mp4, chạy script tự động detect khuôn mặt + crop:

```bash
python scripts/extract_frames_from_videos.py --data-dir data/raw --fps 2 --max-frames 200
```

| Tham số | Ý nghĩa | Khuyến nghị |
|---------|---------|-------------|
| `--fps 2` | Lấy 2 frame mỗi giây video | 2-3 cho video đa dạng |
| `--max-frames 200` | Tối đa frames/video | 100-200 |
| `--no-face-detect` | Bỏ qua face detection | Chỉ dùng khi video đã crop sẵn |

**Kết quả:** Mỗi video 10s → ~20 ảnh .jpg khuôn mặt crop trong `data/raw/{label}/`

---

## BƯỚC 3: Sinh pseudo-video clips

Mỗi ảnh .jpg → 1 clip 16 frames với motion profile riêng theo condition:

```bash
# Cách 1: Script chuyên dụng (nhanh, đơn giản)
python scripts/generate_pseudo_videos.py --input-dir data/raw --output-dir data/video_clips

# Cách 2: Qua DataCollector API (tích hợp vào project)
python scripts/collect_faces.py --mode generate --source-dir data/raw --output-dir data/video_clips
```

| Condition | Hiệu ứng giả lập |
|-----------|-------------------|
| **normal** | Xoay ±0.5°, thở đều, sáng ±2 |
| **abnormal** | Xoay ±1.5°, thở bất thường, sáng ±5, giảm 8% saturation |
| **infarction** | Xoay ±3°, run mạnh, sáng ±8 + tối hơn, giảm 15% saturation (tái) |

**Kết quả kỳ vọng:**
```
data/video_clips/
├── normal/          ← ~100-200 clip folders
│   ├── clip_00000/  ← frame_00.jpg ... frame_15.jpg (16 files)
│   ├── clip_00001/
│   └── ...
├── abnormal/        ← ~100-200 clip folders
└── infarction/      ← ~100-200 clip folders
```

---

## BƯỚC 4: Upload Kaggle + Train

### 4.1 Nén và upload
```bash
# Nén thư mục (dùng 7-Zip hoặc chuột phải → Compress)
# Kích thước dự kiến: 200-500 MB
```
- Vào https://www.kaggle.com/datasets → **New Dataset** → Upload `video_clips.zip`

### 4.2 Train trên Kaggle
- Upload `notebooks/train_3dcnn_kaggle.ipynb`
- Bật **GPU T4×2** trong Settings → Accelerator
- Sửa `DATA_DIR` cho đúng path dataset
- **Run All Cells** → đợi ~30-60 phút

### 4.3 Kiểm tra kết quả
- Xem confusion matrix + classification report
- **Mục tiêu:** Accuracy ≥ 70% (với data ít), ≥ 85% (data đủ)

---

## BƯỚC 5: Download checkpoint

1. Tải `3dcnn_best.pth` từ Kaggle Output
2. Đặt tại: `models/checkpoints/3dcnn_best.pth`
3. Backend sẽ tự động load khi khởi động

---

## TỔNG KẾT LỆNH CHẠY THEO THỨ TỰ

```bash
# 1. Tải video .mp4 vào data/raw/{label}/ (THỦ CÔNG)

# 2. Extract frames
python scripts/extract_frames_from_videos.py --data-dir data/raw --fps 2

# 3. Sinh pseudo-videos
python scripts/generate_pseudo_videos.py --input-dir data/raw --output-dir data/video_clips

# 4. Kiểm tra số lượng
python scripts/collect_faces.py --stats --output-dir data/video_clips

# 5. Nén upload Kaggle (THỦ CÔNG)

# 6. Train trên Kaggle (NOTEBOOK)

# 7. Download checkpoint về models/checkpoints/3dcnn_best.pth (THỦ CÔNG)
```

---

## CÂU HỎI MỞ

1. Bạn có muốn tôi viết script **tự động tải video từ Pexels** bằng API key không?
2. Bạn có GPU NVIDIA trên máy để train local thay vì Kaggle không?
3. Bạn muốn bao nhiêu clips tối thiểu cho mỗi class trước khi train?
