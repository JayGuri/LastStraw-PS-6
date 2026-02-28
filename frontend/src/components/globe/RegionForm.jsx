import React, { useState, useCallback, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useGlobeStore } from '../../stores/globeStore.js'
import { useAppStore } from '../../stores/appStore.js'
import { geocodeApi, parseNominatimResult } from '../../api/geocodeApi.js'
import { floodDetectApi } from '../../api/floodDetectApi.js'
import { mockFloodResponse } from '../../data/mockFloodResponse.js'

// ── Inline autocomplete input ──────────────────────────────────────────────
function GeoSearchInput({ label, placeholder, value, onChange, results, onSelect, isSearching, onClear }) {
  const [open, setOpen] = useState(false)
  const wrapRef = useRef(null)

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e) => { if (!wrapRef.current?.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  // Open when results arrive
  useEffect(() => { if (results.length > 0) setOpen(true) }, [results])

  return (
    <div ref={wrapRef} className="relative">
      <label className="text-[10px] font-mono text-text-3 mb-1.5 block uppercase tracking-wider">
        {label}
      </label>
      <div className="relative flex items-center">
        <input
          type="text"
          value={value}
          onChange={e => { onChange(e.target.value); setOpen(true) }}
          onFocus={() => { if (results.length > 0) setOpen(true) }}
          placeholder={placeholder}
          className="w-full bg-bg-2 border border-white/10 rounded-xl px-3.5 py-2.5
                     text-sm text-text font-body placeholder:text-text-3/50
                     focus:border-gold/40 focus:ring-1 focus:ring-gold/10 focus:outline-none
                     transition-all pr-8"
          autoComplete="off"
        />
        {isSearching && (
          <span className="absolute right-3 text-[10px] text-gold/50 font-mono animate-pulse">···</span>
        )}
        {!isSearching && value && (
          <button
            onClick={() => { onClear(); setOpen(false) }}
            className="absolute right-3 text-text-3 hover:text-text transition-colors text-xs"
          >
            ✕
          </button>
        )}
      </div>

      <AnimatePresence>
        {open && results.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.12 }}
            className="absolute z-50 w-full mt-1.5 bg-bg-card border border-white/10
                       rounded-xl overflow-hidden shadow-[0_8px_32px_rgba(0,0,0,0.5)]
                       max-h-52 overflow-y-auto"
          >
            {results.map(item => {
              const parts = item.display_name.split(',').map(s => s.trim())
              const primary = parts[0]
              const secondary = parts.slice(1, 3).join(', ')
              return (
                <button
                  key={item.place_id}
                  onMouseDown={e => { e.preventDefault(); onSelect(item); setOpen(false) }}
                  className="w-full text-left px-3.5 py-2.5 flex flex-col gap-0.5
                             hover:bg-white/[0.04] border-b border-white/[0.04]
                             transition-colors last:border-0"
                >
                  <span className="text-xs text-text font-medium truncate">{primary}</span>
                  {secondary && (
                    <span className="text-[10px] text-text-3 font-mono truncate">{secondary}</span>
                  )}
                </button>
              )
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
export default function RegionForm() {
  const store        = useGlobeStore()
  const isMockMode   = useAppStore(s => s.isMockMode)
  const showNotification = useAppStore(s => s.showNotification)

  // Three independent field states
  const [cityQuery,    setCityQuery]    = useState('')
  const [stateQuery,   setStateQuery]   = useState('')
  const [countryQuery, setCountryQuery] = useState('')

  const [cityResults,    setCityResults]    = useState([])
  const [stateResults,   setStateResults]   = useState([])
  const [countryResults, setCountryResults] = useState([])

  const [searchingCity,    setSearchingCity]    = useState(false)
  const [searchingState,   setSearchingState]   = useState(false)
  const [searchingCountry, setSearchingCountry] = useState(false)

  const cityDebounce    = useRef(null)
  const stateDebounce   = useRef(null)
  const countryDebounce = useRef(null)

  // ── Debounced searches ──
  const searchCity = useCallback((q) => {
    clearTimeout(cityDebounce.current)
    if (q.length < 2) { setCityResults([]); return }
    cityDebounce.current = setTimeout(async () => {
      setSearchingCity(true)
      const { data } = await geocodeApi.search(q, { limit: 6 })
      setCityResults(data ?? [])
      setSearchingCity(false)
    }, 280)
  }, [])

  const searchState = useCallback((q) => {
    clearTimeout(stateDebounce.current)
    if (q.length < 2) { setStateResults([]); return }
    stateDebounce.current = setTimeout(async () => {
      setSearchingState(true)
      const { data } = await geocodeApi.searchState(q, { limit: 6 })
      setStateResults(data ?? [])
      setSearchingState(false)
    }, 280)
  }, [])

  const searchCountry = useCallback((q) => {
    clearTimeout(countryDebounce.current)
    if (q.length < 2) { setCountryResults([]); return }
    countryDebounce.current = setTimeout(async () => {
      setSearchingCountry(true)
      const { data } = await geocodeApi.searchCountry(q, { limit: 6 })
      setCountryResults(data ?? [])
      setSearchingCountry(false)
    }, 280)
  }, [])

  // ── Selection handlers ──────────────────────────────────────────────────
  const handleSelectCity = (item) => {
    const parsed = parseNominatimResult(item)
    const cityName = parsed.city || item.display_name.split(',')[0].trim()

    setCityQuery(cityName)
    setCityResults([])

    // Auto-fill state + country from address components
    if (parsed.state)   { setStateQuery(parsed.state);   setStateResults([]) }
    if (parsed.country) { setCountryQuery(parsed.country); setCountryResults([]) }

    store.setCity({ name: cityName, lat: parsed.lat, lon: parsed.lon })
    store.setGeocoded({
      ...parsed,
      display_name: [cityName, parsed.state, parsed.country].filter(Boolean).join(', '),
    })
  }

  const handleSelectState = (item) => {
    const parsed   = parseNominatimResult(item)
    const stateName = parsed.state || item.display_name.split(',')[0].trim()

    setStateQuery(stateName)
    setStateResults([])

    // Auto-fill country
    if (parsed.country) { setCountryQuery(parsed.country); setCountryResults([]) }

    store.setState({ name: stateName, lat: parsed.lat, lon: parsed.lon })
    store.setGeocoded({
      ...parsed,
      display_name: [stateName, parsed.country].filter(Boolean).join(', '),
    })
  }

  const handleSelectCountry = (item) => {
    const parsed = parseNominatimResult(item)
    const countryName = parsed.country || item.display_name.split(',')[0].trim()

    setCountryQuery(countryName)
    setCountryResults([])

    store.setCountry({ name: countryName, code: parsed.country_code, lat: parsed.lat, lon: parsed.lon })
    store.setGeocoded({
      ...parsed,
      display_name: countryName,
    })
  }

  // ── Submit analysis ──────────────────────────────────────────────────────
  const handleAnalyze = async () => {
    if (!store.geocoded) {
      showNotification('Select a region first', 'warning')
      return
    }

    store.resetRun()
    store.setRunState({ status: 'queued', progress: 0 })

    if (isMockMode) { simulateMockPolling(); return }

    const payload = {
      region: {
        center:           { lat: store.geocoded.lat, lon: store.geocoded.lon },
        bbox:             store.geocoded.bbox,
        boundary_geojson: store.geocoded.boundary_geojson,
        display_name:     store.geocoded.display_name,
      },
      date:    store.analysisDate || new Date().toISOString().slice(0, 10),
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
          status:   stages[idx],
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
  const geo = store.geocoded

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="card-glass rounded-2xl glow-border p-4 sm:p-5 space-y-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-mono text-gold/60 uppercase tracking-wider">
          Region Selection
        </span>
        {isMockMode && (
          <span className="text-[9px] font-mono text-medium/60 px-2 py-0.5 rounded-full
                           border border-medium/20 bg-medium/5">
            MOCK
          </span>
        )}
      </div>

      {/* Three independent search fields */}
      <GeoSearchInput
        label="City / District"
        placeholder="e.g. Mumbai, New Orleans..."
        value={cityQuery}
        onChange={q => { setCityQuery(q); searchCity(q) }}
        results={cityResults}
        onSelect={handleSelectCity}
        isSearching={searchingCity}
        onClear={() => { setCityQuery(''); setCityResults([]) }}
      />

      <GeoSearchInput
        label="State / Province"
        placeholder="e.g. Kerala, Louisiana..."
        value={stateQuery}
        onChange={q => { setStateQuery(q); searchState(q) }}
        results={stateResults}
        onSelect={handleSelectState}
        isSearching={searchingState}
        onClear={() => { setStateQuery(''); setStateResults([]) }}
      />

      <GeoSearchInput
        label="Country"
        placeholder="e.g. India, United States..."
        value={countryQuery}
        onChange={q => { setCountryQuery(q); searchCountry(q) }}
        results={countryResults}
        onSelect={handleSelectCountry}
        isSearching={searchingCountry}
        onClear={() => { setCountryQuery(''); setCountryResults([]) }}
      />

      {/* Geocoded coordinates readout */}
      <AnimatePresence>
        {geo && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="bg-bg-2 border border-ice/10 rounded-xl px-3.5 py-2.5 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[9px] font-mono text-ice/50 uppercase tracking-wider">
                  Geocoded
                </span>
                <span className="text-[9px] font-mono text-ice/40">
                  {geo.country_code || ''}
                </span>
              </div>
              <div className="text-[11px] font-mono text-ice/80 truncate leading-tight">
                {geo.display_name}
              </div>
              <div className="flex items-center gap-3 text-[10px] font-mono text-text-3">
                <span>
                  <span className="text-text-2">lat </span>
                  {geo.lat?.toFixed(4)}
                </span>
                <span>
                  <span className="text-text-2">lon </span>
                  {geo.lon?.toFixed(4)}
                </span>
                {geo.bbox && (
                  <span className="text-[9px] text-text-3/60">
                    bbox ✓
                  </span>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Date picker */}
      <div>
        <label className="text-[10px] font-mono text-text-3 mb-1.5 block uppercase tracking-wider">
          Analysis Date
        </label>
        <input
          type="date"
          value={store.analysisDate || ''}
          onChange={e => store.setAnalysisDate(e.target.value)}
          className="w-full bg-bg-2 border border-white/10 rounded-xl px-3.5 py-2.5
                     text-sm text-text font-body focus:border-gold/40
                     focus:ring-1 focus:ring-gold/10 focus:outline-none transition-all"
        />
      </div>

      {/* Analyze button */}
      <motion.button
        whileHover={{ scale: 1.01 }}
        whileTap={{ scale: 0.98 }}
        onClick={handleAnalyze}
        disabled={!geo || isRunning}
        className="w-full py-3 rounded-xl font-display font-bold text-sm tracking-wide
                   bg-gold/20 border border-gold/30 text-gold-lt
                   hover:bg-gold/30 hover:border-gold/50 transition-all
                   disabled:opacity-35 disabled:cursor-not-allowed"
      >
        {isRunning ? (
          <span className="flex items-center justify-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-gold animate-pulse" />
            Analyzing...
          </span>
        ) : 'Analyze Flood Risk'}
      </motion.button>
    </motion.div>
  )
}
