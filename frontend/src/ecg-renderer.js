/**
 * Canvas-based ECG Renderer — real-time PQRST waveform.
 */

const ECG_COLORS = {
  normal: '#00d68f',
  pain: '#ff4757',
  abnormal: '#f59e0b',
  infarction: '#ff4757',
};

export class ECGRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.bpm = 72;
    this.condition = 'normal';
    this.running = true;
    this.startTime = performance.now();

    this._resize();
    new ResizeObserver(() => this._resize()).observe(this.canvas.parentElement);
    this._draw();
  }

  setBPM(bpm) { this.bpm = bpm; }
  setCondition(cond) { this.condition = cond; }

  _resize() {
    const parent = this.canvas.parentElement;
    this.canvas.width = parent.clientWidth;
    this.canvas.height = Math.max(parent.clientHeight - 50, 250);
  }

  _ecgValue(pos, cycleIndex = 0) {
    // PQRST complex simulation with condition-specific morphology
    let val = 0;

    if (this.condition === 'abnormal') {
      // Irregular rhythm: vary timing per beat
      const shift = Math.sin(cycleIndex * 2.7) * 0.06;
      pos = Math.max(0, Math.min(1, pos + shift));
      // Variable amplitude
      const ampFactor = 0.7 + Math.abs(Math.sin(cycleIndex * 1.3)) * 0.6;

      // P wave — sometimes absent
      if (pos > 0.0 && pos < 0.1 && cycleIndex % 3 !== 0) {
        val = 0.12 * ampFactor * Math.sin((pos / 0.1) * Math.PI);
      }
      // QRS — variable width and height
      if (pos > 0.12 && pos < 0.22) {
        const qrs = (pos - 0.12) / 0.10;
        if (qrs < 0.2) val = -0.1 * ampFactor * Math.sin((qrs / 0.2) * Math.PI);
        else if (qrs < 0.5) val = 1.0 * ampFactor * Math.sin(((qrs - 0.2) / 0.3) * Math.PI);
        else val = -0.25 * ampFactor * Math.sin(((qrs - 0.5) / 0.5) * Math.PI);
      }
      // T wave — inverted sometimes
      if (pos > 0.28 && pos < 0.48) {
        const sign = cycleIndex % 4 === 0 ? -1 : 1;
        val = sign * 0.2 * ampFactor * Math.sin(((pos - 0.28) / 0.2) * Math.PI);
      }
      // Extra baseline noise
      val += (Math.random() - 0.5) * 0.05;
      return val;
    }

    if (this.condition === 'infarction' || this.condition === 'pain') {
      // P wave
      if (pos > 0.0 && pos < 0.08) {
        val = 0.18 * Math.sin((pos / 0.08) * Math.PI);
      }
      // QRS — tall, narrow
      if (pos > 0.10 && pos < 0.18) {
        const qrs = (pos - 0.10) / 0.08;
        if (qrs < 0.2) val = -0.12 * Math.sin((qrs / 0.2) * Math.PI);
        else if (qrs < 0.5) val = 1.4 * Math.sin(((qrs - 0.2) / 0.3) * Math.PI);
        else val = -0.35 * Math.sin(((qrs - 0.5) / 0.5) * Math.PI);
      }
      // ST elevation (key indicator of MI)
      if (pos > 0.18 && pos < 0.30) {
        val += 0.30;
      }
      // T wave — peaked or inverted
      if (pos > 0.30 && pos < 0.45) {
        val = -0.25 * Math.sin(((pos - 0.30) / 0.15) * Math.PI);
      }
      val += (Math.random() - 0.5) * 0.02;
      return val;
    }

    // Normal sinus rhythm
    // P wave (atrial depolarization)
    if (pos > 0.0 && pos < 0.1) {
      val = 0.15 * Math.sin((pos / 0.1) * Math.PI);
    }
    // QRS complex
    else if (pos > 0.12 && pos < 0.2) {
      const qrs = (pos - 0.12) / 0.08;
      if (qrs < 0.2) val = -0.1 * Math.sin((qrs / 0.2) * Math.PI);
      else if (qrs < 0.5) val = 1.2 * Math.sin(((qrs - 0.2) / 0.3) * Math.PI);
      else val = -0.3 * Math.sin(((qrs - 0.5) / 0.5) * Math.PI);
    }
    // T wave
    else if (pos > 0.25 && pos < 0.45) {
      val = 0.3 * Math.sin(((pos - 0.25) / 0.2) * Math.PI);
    }

    return val;
  }

  _draw() {
    if (!this.running) return;
    requestAnimationFrame(() => this._draw());

    const { ctx, canvas } = this;
    const w = canvas.width;
    const h = canvas.height;
    const elapsed = (performance.now() - this.startTime) / 1000;
    const cycle = 60 / this.bpm;
    const color = ECG_COLORS[this.condition] || ECG_COLORS.normal;

    // Background
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < w; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    // Baseline
    const baseY = h * 0.55;
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.beginPath(); ctx.moveTo(0, baseY); ctx.lineTo(w, baseY); ctx.stroke();

    // ECG trace (scrolling)
    const pixelsPerSec = 150;
    const visibleDuration = w / pixelsPerSec;

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.shadowColor = color;
    ctx.shadowBlur = 6;
    ctx.beginPath();

    for (let px = 0; px < w; px++) {
      const t = elapsed - (w - px) / pixelsPerSec;
      if (t < 0) continue;
      const pos = ((t % cycle) / cycle);
      const cycleIndex = Math.floor(t / cycle);
      const val = this._ecgValue(pos, cycleIndex);
      const y = baseY - val * (h * 0.35);

      if (px === 0) ctx.moveTo(px, y);
      else ctx.lineTo(px, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Scanning line
    const scanX = (elapsed * pixelsPerSec) % w;
    ctx.strokeStyle = `${color}40`;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(scanX, 0); ctx.lineTo(scanX, h); ctx.stroke();
  }

  destroy() { this.running = false; }
}
