const BASE_URL =
  import.meta.env.VITE_API_URL ?? 'http://localhost:8000/api/v1'

async function request(method, path, body) {
  try {
    const options = {
      method,
      headers: { 'Content-Type': 'application/json' },
    }
    if (body !== undefined) {
      options.body = JSON.stringify(body)
    }

    const res = await fetch(`${BASE_URL}${path}`, options)

    if (!res.ok) {
      let message = `HTTP ${res.status}`
      try {
        const json = await res.json()
        message = json?.detail ?? json?.error ?? message
      } catch {
        // non-JSON error body
      }
      return { data: null, error: message }
    }

    const data = await res.json()
    return { data, error: null }
  } catch (err) {
    return { data: null, error: err?.message ?? 'Network error' }
  }
}

export const floodDetectApi = {
  /** POST /api/v1/flood-detect — Submit a flood detection run. */
  submitDetection(payload) {
    return request('POST', '/flood-detect', payload)
  },

  /** GET /api/v1/flood-detect/{runId} — Poll detection status. */
  getDetectionStatus(runId) {
    return request('GET', `/flood-detect/${runId}`)
  },
}
