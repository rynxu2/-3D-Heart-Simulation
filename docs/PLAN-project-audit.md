# Khảo sát & Kế hoạch Nâng cấp Dự án (Project Audit & Upgrade Plan)

> [!NOTE]
> Báo cáo khảo sát: Dự án hiện tại của bạn đã làm rất tốt phần core AI (PainFormer Zero-shot) và logic render 3D Shader mô phỏng tim (Wiggers diagram). Tuy nhiên, vì mục tiêu là "Surgical Monitor" thời gian thực, có 2 điểm nghẽn (bottlenecks) lớn nhất hiện tại:
> 1. **Giao tiếp API (HTTP Polling)**: Việc dùng `fetch` gửi ảnh mỗi 1.5s tạo ra overhead HTTP lớn và làm gián đoạn trải nghiệm "thời gian thực".
> 2. **Kiến trúc Frontend (Vanilla JS)**: Việc thao tác trực tiếp DOM trong `main.js` với lượng logic UI tăng lên sẽ rất khó bảo trì. `heart-viewer.js` sử dụng Three.js thuần khá cồng kềnh.

## Open Questions (Cần bạn xác nhận)
- Bạn có muốn chuyển giao diện sang **React.js (với React Three Fiber)** để dễ mở rộng và bảo trì sau này không? Hay vẫn muốn giữ nguyên Vanilla JS?
- Bạn có muốn đổi từ HTTP Fetch API sang **WebSockets** ở Backend để luồng video phân tích mượt mà hơn (Real-time Streaming) không?

## Proposed Changes / Tech Stack Nâng Cấp Đề Xuất

- **Frontend**: Nâng cấp từ Vanilla JS sang **React 18+** và `@react-three/fiber`. Sử dụng `zustand` để quản lý state (BPM, Condition, Pain Score) xuyên suốt app.
- **Backend**: Cấu hình thêm Endpoint **WebSocket** trong FastAPI để stream dữ liệu ảnh với độ trễ thấp nhất.
- **Tối ưu Camera**: Đưa việc capture frame (`canvas.toBlob`) vào Web Worker để không block Main Thread của giao diện 3D.

---

## Task Breakdown

### Task 1: Nâng cấp Backend (WebSockets) [P1]
- **Agent**: `backend-specialist`
- **Skill**: `python-patterns`
- **INPUT**: `backend/src/web/app.py` endpoint `/api/predict-painformer`
- **OUTPUT**: Endpoint `fastapi.websockets.WebSocket` tại `/ws/analyze` nhận frame liên tục và trả về JSON `pain_score`, `bpm` qua vòng lặp. Bổ sung Rate Limiting / Size Limiting để tránh nghẽn RAM server.
- **VERIFY**: Có thể dùng Python Client kết nối WS và gửi/nhận dữ liệu liên tục không bị disconnect.

### Task 2: Thiết lập Kiến trúc Frontend React [P2]
- **Agent**: `frontend-specialist`
- **Skill**: `react-best-practices`
- **INPUT**: Cấu hình Vite hiện tại
- **OUTPUT**: Cấu trúc thư mục mới: `src/components/`, `src/store/`, `src/hooks/` sử dụng React, Tailwind CSS và Zustand.
- **VERIFY**: Chạy `npm run dev` hiển thị giao diện React cơ bản.

### Task 3: Chuyển đổi 3D Component sang React Three Fiber [P2]
- **Agent**: `frontend-specialist`
- **Skill**: `frontend-design`
- **INPUT**: Logic shader từ `frontend/src/heart-viewer.js`
- **OUTPUT**: Component `<HeartViewer3D />` sử dụng `@react-three/fiber` giữ nguyên shader mô phỏng nhịp tim.
- **VERIFY**: Quả tim 3D render tốt, shader hoạt động và có thể thay đổi tốc độ đập bằng cách update state (Props/Zustand).

### Task 4: Chuyển đổi ECG Component sang React [P2]
- **Agent**: `frontend-specialist`
- **Skill**: `frontend-design`
- **INPUT**: `frontend/src/ecg-renderer.js`
- **OUTPUT**: Component `<ECGWaveform />` render Canvas trong React.
- **VERIFY**: Đồ thị điện tâm đồ chạy mượt, thay đổi đồng bộ theo State.

### Task 5: Tích hợp WebSocket Hook & Tối ưu luồng Camera [P2]
- **Agent**: `frontend-specialist`
- **Skill**: `react-best-practices`
- **INPUT**: Hàm `captureFrame` trong `frontend/src/main.js`
- **OUTPUT**: Custom hook `useCameraWebSocket` quản lý stream camera và WebSocket, có thể sử dụng Web Worker để resize/capture ảnh nhằm không block UI.
- **VERIFY**: Giao diện cập nhật liên tục kết quả AI, FPS của 3D Heart vẫn đạt mức cao ổn định (60fps).

---

## ✅ PHASE X: Verification Checklist
- [ ] Backend: Endpoint WS khởi chạy và test connection OK.
- [ ] Frontend: Lệnh `npm run build` đóng gói không lỗi.
- [ ] Hiệu năng: FPS > 50 khi mở Camera liên tục.
- [ ] Bảo mật: Request WebSocket có validate header/kích thước ảnh.
