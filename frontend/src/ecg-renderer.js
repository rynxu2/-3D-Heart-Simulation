/**
 * Canvas-based ECG Renderer — real-time PQRST waveform.
 *
 * Medically accurate morphology per condition:
 * - Normal: regular sinus rhythm with proper PQRST intervals
 * - Abnormal: irregularly irregular (AF-like), absent P waves, variable amplitude
 * - Infarction: ST elevation, pathologic Q waves, peaked/inverted T waves
 */

const ECG_COLORS = {
  normal: '#00d68f',
  no_pain: '#00d68f',
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
    // Route to condition-specific morphology
    if (this.condition === 'abnormal') return this._ecgAbnormal(pos, cycleIndex);
    if (this.condition === 'infarction' || this.condition === 'pain') return this._ecgInfarction(pos, cycleIndex);
    return this._ecgNormal(pos);
  }

  // ── Normal Sinus Rhythm ─────────────────────────────
  // PR: 160ms, QRS: 80ms, isoelectric ST, upright asymmetric T
  _ecgNormal(pos) {
    let val = 0;

    // P wave — smooth, rounded, 80ms duration (0.00–0.10)
    if (pos > 0.00 && pos < 0.10) {
      val = 0.15 * Math.sin((pos / 0.10) * Math.PI);
    }
    // PR segment — isoelectric (0.10–0.14)
    // QRS complex — narrow 80ms (0.14–0.22)
    else if (pos > 0.14 && pos < 0.22) {
      const qrs = (pos - 0.14) / 0.08;
      if (qrs < 0.15) {
        // Q wave — small downward
        val = -0.08 * Math.sin((qrs / 0.15) * Math.PI);
      } else if (qrs < 0.50) {
        // R wave — tall upward
        val = 1.2 * Math.sin(((qrs - 0.15) / 0.35) * Math.PI);
      } else {
        // S wave — small downward
        val = -0.20 * Math.sin(((qrs - 0.50) / 0.50) * Math.PI);
      }
    }
    // ST segment — isoelectric (0.22–0.28)
    // T wave — upright, asymmetric: slower upstroke, faster downstroke (0.28–0.48)
    else if (pos > 0.28 && pos < 0.48) {
      const tPos = (pos - 0.28) / 0.20;
      // Asymmetric T: rise over 60%, fall over 40%
      if (tPos < 0.60) {
        val = 0.30 * Math.sin((tPos / 0.60) * Math.PI * 0.5);
      } else {
        val = 0.30 * Math.cos(((tPos - 0.60) / 0.40) * Math.PI * 0.5);
      }
    }

    return val;
  }

  // ── Abnormal (Arrhythmia / Atrial Fibrillation) ─────
  // Irregularly irregular RR, absent P waves, variable QRS amplitude
  _ecgAbnormal(pos, cycleIndex) {
    let val = 0;

    // RR interval variation — true irregular timing
    const shift = Math.sin(cycleIndex * 2.7 + 0.3) * 0.08
                + Math.cos(cycleIndex * 1.9) * 0.04;
    pos = Math.max(0, Math.min(1, pos + shift));

    // Variable amplitude per beat (EF 35-55%)
    const ampFactor = 0.65 + Math.abs(Math.sin(cycleIndex * 1.3 + 0.7)) * 0.55;

    // P wave — ABSENT in AF (replaced by fibrillatory baseline)
    // Only present in ~20% of beats (occasional sinus capture)
    if (pos > 0.00 && pos < 0.10 && cycleIndex % 5 === 0) {
      val = 0.10 * ampFactor * Math.sin((pos / 0.10) * Math.PI);
    }

    // QRS complex — variable width and height
    if (pos > 0.12 && pos < 0.22) {
      const qrs = (pos - 0.12) / 0.10;
      if (qrs < 0.18) {
        val = -0.10 * ampFactor * Math.sin((qrs / 0.18) * Math.PI);
      } else if (qrs < 0.50) {
        val = 1.0 * ampFactor * Math.sin(((qrs - 0.18) / 0.32) * Math.PI);
      } else {
        val = -0.22 * ampFactor * Math.sin(((qrs - 0.50) / 0.50) * Math.PI);
      }
    }

    // ST segment — subtle depression in some beats
    if (pos > 0.22 && pos < 0.28) {
      val = (cycleIndex % 3 === 0) ? -0.06 : 0.0;
    }

    // T wave — sometimes inverted (25% of beats)
    if (pos > 0.28 && pos < 0.46) {
      const sign = (cycleIndex % 4 === 1) ? -1 : 1;
      val = sign * 0.18 * ampFactor * Math.sin(((pos - 0.28) / 0.18) * Math.PI);
    }

    // Fibrillatory baseline undulation (chaotic, replaces P waves)
    val += Math.sin(pos * 47 + cycleIndex * 3.1) * 0.04
         + Math.sin(pos * 73 + cycleIndex * 5.7) * 0.025
         + (Math.random() - 0.5) * 0.03;

    return val;
  }

  // ── Infarction (Acute MI) ───────────────────────────
  // Sinus tachycardia, ST ELEVATION, pathologic Q, peaked→inverted T
  _ecgInfarction(pos, cycleIndex) {
    let val = 0;

    // P wave — peaked (atrial overload from fast rate)
    if (pos > 0.00 && pos < 0.08) {
      val = 0.20 * Math.sin((pos / 0.08) * Math.PI);
    }

    // QRS complex with pathologic Q wave
    if (pos > 0.10 && pos < 0.20) {
      const qrs = (pos - 0.10) / 0.10;
      if (qrs < 0.12) {
        // Pathologic Q wave — deeper and wider than normal (necrosis marker)
        val = -0.18 * Math.sin((qrs / 0.12) * Math.PI);
      } else if (qrs < 0.50) {
        // Tall R wave
        val = 1.4 * Math.sin(((qrs - 0.12) / 0.38) * Math.PI);
      } else {
        // S wave
        val = -0.30 * Math.sin(((qrs - 0.50) / 0.50) * Math.PI);
      }
    }

    // ★ ST SEGMENT ELEVATION — key MI indicator ★
    // Elevated plateau (0.20–0.32) — the hallmark of acute STEMI
    if (pos > 0.20 && pos < 0.32) {
      const stProgress = (pos - 0.20) / 0.12;
      // Smooth rise to plateau at 0.30 (representing ~3mm elevation)
      val = 0.30 * Math.sin(stProgress * Math.PI * 0.5);
    }

    // T wave — evolving pattern based on MI phase
    // Hyperacute: peaked tall T → later: deeply inverted T
    if (pos > 0.32 && pos < 0.48) {
      const tPos = (pos - 0.32) / 0.16;
      // Use cycleIndex to show evolution: early beats peaked, later inverted
      const phase = (cycleIndex % 6) / 6;
      if (phase < 0.5) {
        // Hyperacute: peaked tall T (early MI)
        val = 0.35 * Math.sin(tPos * Math.PI);
      } else {
        // Evolving: deeply inverted T (ischemic)
        val = -0.28 * Math.sin(tPos * Math.PI);
      }
    }

    // Subtle baseline artifact
    val += (Math.random() - 0.5) * 0.015;

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
