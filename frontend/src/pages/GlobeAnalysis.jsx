import React from 'react'
import { motion } from 'framer-motion'
import CesiumGlobe from '../components/globe/CesiumGlobe.jsx'
import RegionForm from '../components/globe/RegionForm.jsx'
import ResultsPanel from '../components/globe/ResultsPanel.jsx'
import ProgressOverlay from '../components/globe/ProgressOverlay.jsx'
import { useGlobeStore } from '../stores/globeStore.js'

const STATUS_CONFIG = {
  idle:       { label: 'IDLE',      style: 'bg-white/3 border-white/8 text-text-3' },
  queued:     { label: 'QUEUED',    style: 'bg-gold/10 border-gold/25 text-gold' },
  preprocessing: { label: 'PREPROCESSING', style: 'bg-gold/10 border-gold/25 text-gold' },
  detecting:  { label: 'DETECTING', style: 'bg-gold/10 border-gold/25 text-gold' },
  scoring:    { label: 'SCORING',   style: 'bg-gold/10 border-gold/25 text-gold' },
  completed:  { label: 'COMPLETED', style: 'bg-low/10 border-low/25 text-low' },
  failed:     { label: 'FAILED',    style: 'bg-critical/10 border-critical/25 text-critical' },
}

export default function GlobeAnalysis() {
  const status = useGlobeStore(s => s.status)
  const result = useGlobeStore(s => s.result)
  const geocoded = useGlobeStore(s => s.geocoded)

  const isRunning = ['queued', 'preprocessing', 'detecting', 'scoring'].includes(status)
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.idle

  return (
    <div className="min-h-screen bg-bg noise pt-14 sm:pt-16 flex flex-col">
      <div className="flex-1 flex flex-col max-w-[1600px] mx-auto w-full
                       px-3 sm:px-4 py-3 sm:py-4 gap-3 sm:gap-4">

        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <span className="data-tag text-gold/50 mb-2 block">CESIUM GLOBE</span>
            <h1 className="font-display font-extrabold text-xl sm:text-3xl text-text tracking-tight">
              Global Flood Detection
            </h1>
            <p className="text-xs text-text-2 mt-1 max-w-md font-body">
              Select a region on the globe, choose a date, and run flood detection analysis
              with real-time pipeline visualization.
            </p>
          </div>

          <div className="flex items-center gap-3">
            {/* Status chip */}
            <div className={`px-3 py-1.5 rounded-xl text-[10px] font-mono border ${cfg.style}`}>
              {isRunning && <span className="w-1.5 h-1.5 rounded-full bg-gold inline-block animate-pulse mr-1.5" />}
              {cfg.label}
            </div>

            {/* Region display */}
            {geocoded?.display_name && (
              <div className="text-xs font-mono text-ice/70 px-3 py-1.5 rounded-xl
                              border border-ice/15 bg-ice/5">
                {geocoded.display_name}
              </div>
            )}
          </div>
        </div>

        {/* Main grid */}
        <div className="flex-1 flex flex-col lg:grid lg:grid-cols-12 gap-3 sm:gap-4"
             style={{ minHeight: 'min(500px, 65vh)' }}>

          {/* Globe panel — min-h ensures Cesium container has dimensions before init */}
          <div className="col-span-12 lg:col-span-7 xl:col-span-8 bg-bg-card rounded-2xl
                          glow-border overflow-hidden relative min-h-[420px]">
            <CesiumGlobe />
            {isRunning && <ProgressOverlay />}
            {status === 'failed' && <ProgressOverlay />}
          </div>

          {/* Sidebar */}
          <div className="col-span-12 lg:col-span-5 xl:col-span-4 flex flex-col gap-3 sm:gap-4
                          overflow-y-auto lg:max-h-[calc(100vh-8rem)]">
            <RegionForm />
            {result && <ResultsPanel />}
          </div>
        </div>
      </div>
    </div>
  )
}
