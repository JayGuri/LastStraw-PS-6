import React, { useState, useCallback, useRef } from 'react'
import { motion } from 'framer-motion'
import { useGlobeStore } from '../../stores/globeStore.js'
import { useAppStore } from '../../stores/appStore.js'
import { geocodeApi } from '../../api/geocodeApi.js'
import { floodDetectApi } from '../../api/floodDetectApi.js'
import { mockFloodResponse } from '../../data/mockFloodResponse.js'

export default function RegionForm() {
  const store = useGlobeStore()
  const isMockMode = useAppStore(s => s.isMockMode)
  const showNotification = useAppStore(s => s.showNotification)

  const [countryQuery, setCountryQuery] = useState('')
  const [countrySuggestions, setCountrySuggestions] = useState([])
  const [states, setStates] = useState([])
  const [cities, setCities] = useState([])
  const [isSearching, setIsSearching] = useState(false)

  const debounceRef = useRef(null)

  // Debounced country search
  const searchCountry = useCallback((q) => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    if (q.length < 2) { setCountrySuggestions([]); return }
    debounceRef.current = setTimeout(async () => {
      setIsSearching(true)
      const { data } = await geocodeApi.search(q, { limit: 5 })
      if (data) setCountrySuggestions(data)
      setIsSearching(false)
    }, 300)
  }, [])

  const selectCountry = async (item) => {
    const country = {
      name: item.display_name.split(',')[0].trim(),
      code: item.address?.country_code?.toUpperCase() ?? '',
      lat: parseFloat(item.lat),
      lon: parseFloat(item.lon),
    }
    const bbox = item.boundingbox?.map(Number)
    store.setCountry(country)
    store.setGeocoded({
      lat: country.lat,
      lon: country.lon,
      bbox: bbox ? [bbox[2], bbox[0], bbox[3], bbox[1]] : null,
      boundary_geojson: item.geojson ?? null,
      display_name: country.name,
    })
    setCountrySuggestions([])
    setCountryQuery(country.name)
    setStates([])
    setCities([])

    // Fetch states
    if (country.code) {
      const { data } = await geocodeApi.getStates(country.code)
      if (data) setStates(data)
    }
  }

  const selectState = async (item) => {
    if (!item) return
    const stateObj = {
      name: item.display_name.split(',')[0].trim(),
      lat: parseFloat(item.lat),
      lon: parseFloat(item.lon),
    }
    const bbox = item.boundingbox?.map(Number)
    store.setState(stateObj)
    store.setGeocoded({
      lat: stateObj.lat,
      lon: stateObj.lon,
      bbox: bbox ? [bbox[2], bbox[0], bbox[3], bbox[1]] : null,
      boundary_geojson: item.geojson ?? null,
      display_name: `${stateObj.name}, ${store.country?.name ?? ''}`,
    })
    setCities([])

    // Fetch cities
    if (store.country?.code) {
      const { data } = await geocodeApi.getCities(stateObj.name, store.country.code)
      if (data) setCities(data)
    }
  }

  const selectCity = (item) => {
    if (!item) return
    const cityObj = {
      name: item.display_name.split(',')[0].trim(),
      lat: parseFloat(item.lat),
      lon: parseFloat(item.lon),
    }
    const bbox = item.boundingbox?.map(Number)
    store.setCity(cityObj)
    store.setGeocoded({
      lat: cityObj.lat,
      lon: cityObj.lon,
      bbox: bbox ? [bbox[2], bbox[0], bbox[3], bbox[1]] : null,
      boundary_geojson: item.geojson ?? null,
      display_name: `${cityObj.name}, ${store.state?.name ?? ''}, ${store.country?.name ?? ''}`,
    })
  }

  // ── Submit analysis ──
  const handleAnalyze = async () => {
    if (!store.geocoded) {
      showNotification('Select a region first', 'warning')
      return
    }

    store.resetRun()
    store.setRunState({ status: 'queued', progress: 0 })

    if (isMockMode) {
      simulateMockPolling()
      return
    }

    const payload = {
      region: {
        center: { lat: store.geocoded.lat, lon: store.geocoded.lon },
        bbox: store.geocoded.bbox,
        boundary_geojson: store.geocoded.boundary_geojson,
        display_name: store.geocoded.display_name,
      },
      date: store.analysisDate || new Date().toISOString().slice(0, 10),
      options: { sensor: 'S1_GRD', detector: 'unet' },
    }

    const { data, error } = await floodDetectApi.submitDetection(payload)
    if (error) {
      store.setRunState({ status: 'failed', error })
      showNotification(error, 'error')
      return
    }

    store.setRunState({ runId: data.run_id, status: data.status })
    startPolling(data.run_id)
  }

  const startPolling = (runId) => {
    const poll = setInterval(async () => {
      const { data, error } = await floodDetectApi.getDetectionStatus(runId)
      if (error) {
        clearInterval(poll)
        store.setRunState({ status: 'failed', error })
        showNotification(error, 'error')
        return
      }
      store.setRunState({ status: data.status, progress: data.progress ?? 0 })
      if (data.status === 'completed') {
        clearInterval(poll)
        store.setResult(data.result)
        showNotification('Flood detection complete', 'success')
      }
      if (data.status === 'failed') {
        clearInterval(poll)
        store.setRunState({ error: data.error })
        showNotification(data.error ?? 'Detection failed', 'error')
      }
    }, 3000)
  }

  const simulateMockPolling = () => {
    const stages = ['queued', 'preprocessing', 'detecting', 'scoring', 'completed']
    let idx = 0
    const timer = setInterval(() => {
      idx++
      if (idx < stages.length) {
        store.setRunState({
          status: stages[idx],
          progress: Math.round((idx / (stages.length - 1)) * 100),
        })
      }
      if (stages[idx] === 'completed') {
        clearInterval(timer)
        store.setResult(mockFloodResponse)
        showNotification('Flood detection complete (mock)', 'success')
      }
    }, 1500)
  }

  const isRunning = store.isRunning()

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="card-glass rounded-2xl glow-border p-4 sm:p-5 space-y-4"
    >
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-mono text-gold/60 uppercase tracking-wider">
          Region Selection
        </span>
        {isMockMode && (
          <span className="text-[9px] font-mono text-medium/60 px-2 py-0.5 rounded-full border border-medium/20 bg-medium/5">
            MOCK
          </span>
        )}
      </div>

      {/* Country search */}
      <div className="relative">
        <label className="text-[10px] font-mono text-text-3 mb-1 block uppercase tracking-wider">
          Country / Region
        </label>
        <input
          type="text"
          value={countryQuery}
          onChange={e => { setCountryQuery(e.target.value); searchCountry(e.target.value) }}
          placeholder="Search country or region..."
          className="w-full bg-bg-2 border border-white/10 rounded-lg px-3 py-2
                     text-sm text-text font-body placeholder:text-text-3
                     focus:border-gold/40 focus:outline-none transition-colors"
        />
        {isSearching && (
          <div className="absolute right-3 top-7 text-[10px] text-gold/50 font-mono">...</div>
        )}
        {countrySuggestions.length > 0 && (
          <div className="absolute z-50 w-full mt-1 bg-bg-card border border-white/10
                          rounded-lg overflow-hidden shadow-card max-h-48 overflow-y-auto">
            {countrySuggestions.map(item => (
              <button
                key={item.place_id}
                onClick={() => selectCountry(item)}
                className="w-full text-left px-3 py-2 text-xs text-text
                           hover:bg-white/5 border-b border-white/[0.03] truncate"
              >
                {item.display_name}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* State dropdown */}
      {states.length > 0 && (
        <div>
          <label className="text-[10px] font-mono text-text-3 mb-1 block uppercase tracking-wider">
            State / Province
          </label>
          <select
            onChange={e => { const idx = e.target.value; if (idx !== '') selectState(states[idx]) }}
            defaultValue=""
            className="w-full bg-bg-2 border border-white/10 rounded-lg px-3 py-2
                       text-sm text-text font-body focus:border-gold/40 focus:outline-none"
          >
            <option value="" disabled>Select state...</option>
            {states.map((s, i) => (
              <option key={s.place_id} value={i}>
                {s.display_name.split(',')[0].trim()}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* City dropdown */}
      {cities.length > 0 && (
        <div>
          <label className="text-[10px] font-mono text-text-3 mb-1 block uppercase tracking-wider">
            City / District
          </label>
          <select
            onChange={e => { const idx = e.target.value; if (idx !== '') selectCity(cities[idx]) }}
            defaultValue=""
            className="w-full bg-bg-2 border border-white/10 rounded-lg px-3 py-2
                       text-sm text-text font-body focus:border-gold/40 focus:outline-none"
          >
            <option value="" disabled>Select city...</option>
            {cities.map((c, i) => (
              <option key={c.place_id} value={i}>
                {c.display_name.split(',')[0].trim()}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Date picker */}
      <div>
        <label className="text-[10px] font-mono text-text-3 mb-1 block uppercase tracking-wider">
          Analysis Date
        </label>
        <input
          type="date"
          value={store.analysisDate || ''}
          onChange={e => store.setAnalysisDate(e.target.value)}
          className="w-full bg-bg-2 border border-white/10 rounded-lg px-3 py-2
                     text-sm text-text font-body focus:border-gold/40 focus:outline-none"
        />
      </div>

      {/* Analyze button */}
      <motion.button
        whileHover={{ scale: 1.01 }}
        whileTap={{ scale: 0.98 }}
        onClick={handleAnalyze}
        disabled={!store.geocoded || isRunning}
        className="w-full py-3 rounded-xl font-display font-bold text-sm
                   bg-gold/20 border border-gold/30 text-gold-lt
                   hover:bg-gold/30 transition-all
                   disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {isRunning ? 'Analyzing...' : 'Analyze Flood Risk'}
      </motion.button>
    </motion.div>
  )
}
