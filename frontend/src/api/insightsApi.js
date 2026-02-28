const BASE_URL =
  import.meta.env.VITE_INSIGHTS_API_URL ?? 'https://aish2005deshmukh-hackx-flood-backend.hf.space'

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

export const insightsApi = {
  /** POST /analyze — Trigger a new analysis run */
  analyze(payload) {
    return request('POST', '/analyze', payload)
  },

  /** GET /runs — Fetch all historical runs */
  getRuns() {
    return request('GET', '/runs')
  },

  /** GET /runs/{run_id} — Fetch full detail for one run */
  getRunDetail(runId) {
    return request('GET', `/runs/${runId}`)
  },
}
