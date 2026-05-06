/**
 * Main entry — wire up UI, 3D viewer, ECG, and API.
 */
import { HeartViewer } from './heart-viewer.js';
import { ECGRenderer } from './ecg-renderer.js';
// import { predictImage } from './api-client.js';

// ── Initialize components ───────────────────────────
const heartViewer = new HeartViewer('heart-container');
const ecgRenderer = new ECGRenderer('ecg-canvas');

// ── DOM elements ────────────────────────────────────
const btnStartCamera = document.getElementById('start-camera-btn');
const videoFeed = document.getElementById('video-feed');
const cameraOverlay = document.querySelector('.camera-overlay');

const resultLabel = document.getElementById('result-label');
const resultConfidence = document.getElementById('result-confidence');
const resultBpm = document.getElementById('result-bpm');
const probBars = document.getElementById('prob-bars');

const ecgBpm = document.querySelector('.ecg-bpm');
const ecgPattern = document.querySelector('.ecg-pattern');

// Create a hidden canvas for capturing frames
const captureCanvas = document.createElement('canvas');


// ── Media Handling (Camera & Video) ────────────────
let stream = null;
let isAnalyzing = false;
let analysisInterval = null;

btnStartCamera.addEventListener('click', async () => {
  if (isAnalyzing) {
    stopCamera();
  } else {
    await startCamera();
  }
});

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoFeed.srcObject = stream;
    cameraOverlay.style.display = 'none';
    
    isAnalyzing = true;
    btnStartCamera.innerHTML = '<i data-lucide="power-off"></i> Dừng Camera';
    btnStartCamera.classList.add('btn-danger'); // Add a red style or similar
    
    // Auto-start analysis
    startAnalysis();
    
    // Re-render lucide icons if defined
    if (window.lucide) window.lucide.createIcons();
  } catch (err) {
    alert('Không thể truy cập Camera: ' + err.message);
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  videoFeed.srcObject = null;
  cameraOverlay.style.display = 'flex';
  
  isAnalyzing = false;
  btnStartCamera.innerHTML = '<i data-lucide="power"></i> Khởi Động Camera';
  btnStartCamera.classList.remove('btn-danger');
  
  stopAnalysis();
  
  if (window.lucide) window.lucide.createIcons();
}

// ── Realtime Analysis (WebSocket) ────────────────────
let ws = null;
let wsReconnectTimeout = null;
let consecutiveErrors = 0;
const BASE_INTERVAL = 1500;
const MAX_INTERVAL = 30000;

function connectWebSocket() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  
  // Use current hostname, default backend port 7860
  const wsUrl = `ws://${window.location.hostname}:7860/ws/analyze`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log('[WS] Connected to ' + wsUrl);
    consecutiveErrors = 0;
    if (isAnalyzing) sendNextFrame();
  };

  ws.onmessage = (event) => {
    try {
      const result = JSON.parse(event.data);
      if (result.error) {
        console.warn('[WS] Error from server:', result.error);
      } else {
        displayResult(result);
      }
    } catch (e) {
      console.error('[WS] Parse error', e);
    }
    
    // Ping-pong mechanism to prevent backpressure
    if (isAnalyzing && ws.readyState === WebSocket.OPEN) {
      requestAnimationFrame(sendNextFrame);
    }
  };

  ws.onclose = () => {
    console.warn('[WS] Disconnected');
    ws = null;
    if (isAnalyzing) {
      consecutiveErrors++;
      const delay = Math.min(BASE_INTERVAL * Math.pow(2, consecutiveErrors), MAX_INTERVAL);
      
      if (consecutiveErrors >= 3) {
        resultLabel.textContent = '⚠ BACKEND OFFLINE';
        resultLabel.className = 'result-label';
        resultConfidence.textContent = 'Mất kết nối Server AI';
        resultBpm.textContent = `Thử lại sau ${(delay / 1000).toFixed(0)}s...`;
      }
      
      clearTimeout(wsReconnectTimeout);
      wsReconnectTimeout = setTimeout(connectWebSocket, delay);
    }
  };

  ws.onerror = (err) => {
    console.error('[WS] Error:', err);
    // ws.close() will trigger onclose which handles the reconnection
    if (ws.readyState === WebSocket.OPEN) ws.close();
  };
}

function captureFrame() {
  const ctx = captureCanvas.getContext('2d');
  captureCanvas.width = 300;
  captureCanvas.height = 300 * (videoFeed.videoHeight / videoFeed.videoWidth);
  ctx.drawImage(videoFeed, 0, 0, captureCanvas.width, captureCanvas.height);
  
  return new Promise(resolve => {
    captureCanvas.toBlob(blob => {
      resolve(blob);
    }, 'image/jpeg', 0.8);
  });
}

function stopAnalysis() {
  clearTimeout(wsReconnectTimeout);
  if (ws) {
    ws.close();
    ws = null;
  }
  resultLabel.textContent = 'ĐANG CHỜ TÍN HIỆU...';
  resultLabel.className = 'result-label';
  resultConfidence.textContent = 'Độ Tin Cậy: --%';
  resultBpm.textContent = 'Dự Đoán: -- BPM';
  probBars.innerHTML = '';
}

function startAnalysis() {
  if (!videoFeed.srcObject) return;
  connectWebSocket();
}

async function sendNextFrame() {
  if (!isAnalyzing || !ws || ws.readyState !== WebSocket.OPEN) return;
  if (videoFeed.paused || videoFeed.ended) {
    setTimeout(sendNextFrame, 100);
    return;
  }
  
  const blob = await captureFrame();
  ws.send(blob);
}

function displayResult(result) {
  if (!result.face_detected) {
    resultLabel.textContent = 'KHÔNG THẤY KHUÔN MẶT';
    resultLabel.className = 'result-label';
    return;
  }

  const label = result.label;
  const condition = result.condition || label;

  // 3-class display mapping
  const labelDisplay = {
    normal: { text: '🟢 Bình thường — Tim khỏe mạnh', cls: 'normal' },
    no_pain: { text: '🟢 Không đau — Bình thường', cls: 'normal' },
    abnormal: { text: '🟡 Bất thường — Rối loạn nhịp tim', cls: 'abnormal' },
    infarction: { text: '🔴 Nhồi máu cơ tim — Nguy hiểm!', cls: 'infarction' },
    pain: { text: '🔴 Đau — Nghi ngờ bất thường tim', cls: 'pain' },
  };

  const display = labelDisplay[label] || labelDisplay.normal;
  resultLabel.textContent = display.text;
  resultLabel.className = `result-label ${display.cls}`;

  resultConfidence.textContent = `Độ tin cậy: ${(result.confidence * 100).toFixed(1)}%`;

  const bpm = result.bpm;
  const estBpm = bpm.estimated || Math.round((bpm.min + bpm.max) / 2);
  resultBpm.textContent = `Nhịp tim: ~${estBpm} BPM (${bpm.min}-${bpm.max})`;

  // Removed painScore and resultModel to simplify Option B layout

  // Probability bars
  const probs = result.all_probs || {};
  const labelNameMap = {
    no_pain: 'Không đau', pain: 'Đau',
    normal: 'Bình thường', abnormal: 'Bất thường', infarction: 'Nhồi máu',
  };
  const labelColorMap = {
    no_pain: 'var(--accent-green)', pain: 'var(--accent-red)',
    normal: 'var(--accent-green)', abnormal: '#f59e0b', infarction: 'var(--accent-red)',
  };

  probBars.innerHTML = Object.entries(probs).map(([name, prob]) => {
    const pct = (prob * 100).toFixed(1);
    const color = labelColorMap[name] || 'var(--accent-green)';
    const displayName = labelNameMap[name] || name;
    return `
      <div class="prob-bar">
        <span class="prob-bar-label">${displayName}</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" style="width:${pct}%;background:${color}"></div>
        </div>
        <span class="prob-bar-value">${pct}%</span>
      </div>
    `;
  }).join('');

  // Update 3D and ECG with the actual condition
  heartViewer.setBPM(estBpm);
  heartViewer.setCondition(condition);
  ecgRenderer.setBPM(estBpm);
  ecgRenderer.setCondition(condition);

  ecgBpm.textContent = `${estBpm} BPM`;

  const patternNames = {
    regular: 'Nhịp đều', irregular: 'Nhịp không đều', rapid_irregular: 'Nhịp nhanh không đều',
  };
  ecgPattern.textContent = patternNames[bpm.pattern] || bpm.pattern;
}
