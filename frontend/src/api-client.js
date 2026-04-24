/**
 * API Client — fetch wrapper for FastAPI backend.
 */

const BASE = '';  // Same origin (Vite proxy handles /api)

export async function predictImage(file) {
  const form = new FormData();
  form.append('file', file);

  const res = await fetch(`${BASE}/api/predict`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Prediction failed: ${res.status}`);
  return res.json();
}

export async function getHeartParams(label, bpm = null) {
  let url = `${BASE}/api/heart-params/${label}`;
  if (bpm) url += `?bpm=${bpm}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Heart params failed: ${res.status}`);
  return res.json();
}

export async function getECG(label, bpm = null) {
  let url = `${BASE}/api/ecg/${label}`;
  if (bpm) url += `?bpm=${bpm}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`ECG failed: ${res.status}`);
  return res.blob();
}

export async function healthCheck() {
  const res = await fetch(`${BASE}/api/health`);
  return res.json();
}
