/**
 * Main entry — wire up UI, 3D viewer, ECG, and API.
 */
import { HeartViewer } from './heart-viewer.js';
import { ECGRenderer } from './ecg-renderer.js';
import { predictImage } from './api-client.js';

// ── Initialize components ───────────────────────────
const heartViewer = new HeartViewer('heart-container');
const ecgRenderer = new ECGRenderer('ecg-canvas');

// ── DOM elements ────────────────────────────────────
const fileInput = document.getElementById('file-input');
const uploadZone = document.getElementById('upload-zone');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const previewImg = document.getElementById('preview-img');
const analyzeBtn = document.getElementById('analyze-btn');
const resultsCard = document.getElementById('results-card');
const resultLabel = document.getElementById('result-label');
const resultConfidence = document.getElementById('result-confidence');
const resultBpm = document.getElementById('result-bpm');
const probBars = document.getElementById('prob-bars');
const loadingOverlay = document.getElementById('loading-overlay');
const bpmSlider = document.getElementById('bpm-slider');
const bpmValue = document.getElementById('bpm-value');
const conditionSelect = document.getElementById('condition-select');
const ecgBpm = document.querySelector('.ecg-bpm');
const ecgPattern = document.querySelector('.ecg-pattern');

let selectedFile = null;

// ── Upload handling ─────────────────────────────────
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewImg.style.display = 'block';
  uploadPlaceholder.style.display = 'none';
  analyzeBtn.disabled = false;
}

// ── Analysis ────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  loadingOverlay.style.display = 'flex';
  analyzeBtn.disabled = true;

  try {
    const result = await predictImage(selectedFile);
    displayResult(result);
  } catch (err) {
    console.error('Prediction error:', err);
    resultLabel.textContent = `❌ Lỗi: ${err.message}`;
    resultLabel.className = 'result-label';
    resultsCard.style.display = 'block';
  } finally {
    loadingOverlay.style.display = 'none';
    analyzeBtn.disabled = false;
  }
});

function displayResult(result) {
  resultsCard.style.display = 'block';

  if (!result.face_detected) {
    resultLabel.textContent = '❌ Không phát hiện khuôn mặt';
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
  resultBpm.textContent = `Nhịp tim: ${bpm.min}-${bpm.max} BPM (${bpm.pattern})`;

  // Probability bars — support both 2 and 3 classes
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
  const avgBpm = Math.round((bpm.min + bpm.max) / 2);

  heartViewer.setBPM(avgBpm);
  heartViewer.setCondition(condition);
  ecgRenderer.setBPM(avgBpm);
  ecgRenderer.setCondition(condition);

  bpmSlider.value = avgBpm;
  bpmValue.textContent = avgBpm;
  conditionSelect.value = condition;
  ecgBpm.textContent = `${avgBpm} BPM`;

  const patternNames = {
    regular: 'Nhịp đều', irregular: 'Nhịp không đều', rapid_irregular: 'Nhịp nhanh không đều',
  };
  ecgPattern.textContent = patternNames[bpm.pattern] || bpm.pattern;
}

// ── Manual Controls ─────────────────────────────────
bpmSlider.addEventListener('input', () => {
  const bpm = parseInt(bpmSlider.value);
  bpmValue.textContent = bpm;
  heartViewer.setBPM(bpm);
  ecgRenderer.setBPM(bpm);
  ecgBpm.textContent = `${bpm} BPM`;
});

conditionSelect.addEventListener('change', () => {
  const cond = conditionSelect.value;
  heartViewer.setCondition(cond);
  ecgRenderer.setCondition(cond);

  // Auto-set typical BPM for selected condition
  const condBpm = {
    normal: { bpm: 72, text: 'Nhịp đều' },
    abnormal: { bpm: 90, text: 'Nhịp không đều' },
    infarction: { bpm: 110, text: 'Nhịp nhanh không đều' },
  };
  const info = condBpm[cond] || condBpm.normal;

  heartViewer.setBPM(info.bpm);
  ecgRenderer.setBPM(info.bpm);
  bpmSlider.value = info.bpm;
  bpmValue.textContent = info.bpm;
  ecgBpm.textContent = `${info.bpm} BPM`;
  ecgPattern.textContent = info.text;
});
