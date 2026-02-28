import React from 'react'
import { motion } from 'framer-motion'
import { useGlobeStore } from '../../stores/globeStore.js'

const SEVERITY_STYLES = {
  critical: { bg: 'bg-critical/10', border: 'border-critical/20', text: 'text-critical', label: 'Critical' },
  high:     { bg: 'bg-high/10',     border: 'border-high/20',     text: 'text-high',     label: 'High' },
  medium:   { bg: 'bg-medium/10',   border: 'border-medium/20',   text: 'text-medium',   label: 'Medium' },
  low:      { bg: 'bg-low/10',      border: 'border-low/20',      text: 'text-low',      label: 'Low' },
}

function SeverityBadge({ severity }) {
  const s = SEVERITY_STYLES[severity] ?? SEVERITY_STYLES.medium
  return (
    <span className={`px-1.5 py-0.5 rounded text-[9px] font-mono font-bold uppercase
                      border ${s.bg} ${s.border} ${s.text}`}>
      {s.label}
    </span>
  )
}

export default function ResultsPanel() {
  const result = useGlobeStore(s => s.result)
  const selectedZone = useGlobeStore(s => s.selectedZone)
  const setSelectedZone = useGlobeStore(s => s.setSelectedZone)

  if (!result) return null
  const { summary, flood_zones } = result
  const zones = flood_zones?.features ?? []

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="card-glass rounded-2xl glow-border overflow-hidden flex flex-col"
    >
      {/* Summary header */}
      <div className="px-4 py-3 border-b border-white/5">
        <div className="text-[10px] font-mono text-gold/60 uppercase tracking-wider mb-3">
          Detection Results
        </div>
        <div className="grid grid-cols-2 gap-2">
          {[
            { label: 'Flood Area',  val: `${summary.total_flood_area_km2.toLocaleString()} km\u00B2`, color: 'text-critical' },
            { label: 'Avg Depth',   val: `${summary.avg_depth_m.toFixed(1)} m`,                       color: 'text-high' },
            { label: 'Pop Exposed', val: `${(summary.population_exposed / 1e6).toFixed(1)}M`,         color: 'text-ice' },
            { label: 'Confidence',  val: `${(summary.confidence_avg * 100).toFixed(0)}%`,              color: 'text-low' },
          ].map(s => (
            <div key={s.label} className="card-glass rounded-xl p-2.5">
              <div className={`font-display font-bold text-base ${s.color}`}>{s.val}</div>
              <div className="text-[9px] font-mono text-text-3 uppercase">{s.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Zone table */}
      <div className="flex-1 overflow-y-auto max-h-[300px]">
        <div className="px-4 py-2 border-b border-white/5 sticky top-0 bg-bg-card/90 backdrop-blur-sm">
          <span className="text-[10px] font-mono text-text-3">
            {zones.length} Flood Zone{zones.length !== 1 ? 's' : ''} Detected
          </span>
        </div>
        {zones.map((feature, i) => {
          const p = feature.properties
          const isSelected = selectedZone === i
          return (
            <button
              key={p.zone_id || i}
              onClick={() => setSelectedZone(isSelected ? null : i)}
              className={`w-full text-left px-4 py-2.5 border-b border-white/[0.03]
                          transition-colors
                          ${isSelected
                            ? 'bg-gradient-to-r from-gold/10 to-transparent border-l-2 border-l-gold'
                            : 'hover:bg-white/[0.02]'}`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium text-text">
                  {p.admin_name ?? `Zone ${p.zone_id ?? i + 1}`}
                </span>
                <SeverityBadge severity={p.severity} />
              </div>
              <div className="flex items-center gap-3 text-[9px] font-mono text-text-2">
                <span>{p.area_km2?.toFixed(0)} km&sup2;</span>
                <span>{p.avg_depth_m?.toFixed(1)}m depth</span>
                <span>{(p.population_exposed / 1000).toFixed(0)}k pop</span>
              </div>
            </button>
          )
        })}
      </div>

      {/* Metadata footer */}
      <div className="px-4 py-2 border-t border-white/5 flex items-center justify-between">
        <span className="text-[9px] font-mono text-text-3">
          {summary.sensor} &middot; {summary.detector}
        </span>
        <span className="text-[9px] font-mono text-text-3">
          {summary.scene_id?.slice(0, 20) ?? 'N/A'}
        </span>
      </div>
    </motion.div>
  )
}
