/**
 * API Client — fetch wrapper for PainFormer backend.
 */

const BASE = 'http://localhost:7860';

export async function predictImage(file) {
  const form = new FormData();
  form.append('file', file);

  const res = await fetch(`${BASE}/api/predict-painformer`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`Prediction failed: ${res.status}`);
  return res.json();
}

export async function healthCheck() {
  const res = await fetch(`${BASE}/api/health`);
  return res.json();
}
