/**
 * API utility for communicating with the SARC backend.
 */

const API_BASE = '';  // Uses Vite proxy

export async function compressImage(file, config = {}) {
  const formData = new FormData();
  formData.append('file', file);

  // Append config values as form fields
  for (const [key, value] of Object.entries(config)) {
    if (value !== null && value !== undefined) {
      formData.append(key, value.toString());
    }
  }

  const response = await fetch(`${API_BASE}/api/compress`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export async function checkHealth() {
  const response = await fetch(`${API_BASE}/api/health`);
  if (!response.ok) throw new Error('Health check failed');
  return response.json();
}

export async function getExperiments(limit = 50) {
  const response = await fetch(`${API_BASE}/api/experiments?limit=${limit}`);
  if (!response.ok) throw new Error('Failed to fetch experiments');
  return response.json();
}

export async function getExperiment(id) {
  const response = await fetch(`${API_BASE}/api/experiments/${id}`);
  if (!response.ok) throw new Error('Experiment not found');
  return response.json();
}

export async function deleteExperiment(id) {
  const response = await fetch(`${API_BASE}/api/experiments/${id}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to delete experiment');
  return response.json();
}
