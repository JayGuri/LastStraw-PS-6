const NOMINATIM_BASE = 'https://nominatim.openstreetmap.org'

async function nominatimFetch(path, params) {
  try {
    const qs = new URLSearchParams(params)
    const res = await fetch(`${NOMINATIM_BASE}${path}?${qs}`, {
      headers: { 'User-Agent': 'COSMEON-FloodDetection/1.0' },
    })
    if (!res.ok) return { data: null, error: `HTTP ${res.status}` }
    const data = await res.json()
    return { data, error: null }
  } catch (err) {
    return { data: null, error: err?.message ?? 'Geocoding error' }
  }
}

export const geocodeApi = {
  /** Free-text search. Returns top results with optional boundary GeoJSON. */
  search(query, options = {}) {
    const params = {
      q: query,
      format: 'json',
      addressdetails: '1',
      limit: String(options.limit ?? 5),
      polygon_geojson: '1',
    }
    if (options.countrycodes) params.countrycodes = options.countrycodes
    if (options.featuretype) params.featuretype = options.featuretype
    return nominatimFetch('/search', params)
  },

  /** Get states/provinces for a country code (e.g. 'IN'). */
  getStates(countryCode) {
    return this.search('', {
      featuretype: 'state',
      countrycodes: countryCode.toLowerCase(),
      limit: 50,
    })
  },

  /** Get cities within a state for a country. */
  getCities(stateName, countryCode) {
    return this.search(stateName, {
      featuretype: 'city',
      countrycodes: countryCode.toLowerCase(),
      limit: 50,
    })
  },
}
