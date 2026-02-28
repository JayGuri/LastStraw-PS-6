import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import DistrictHexMap from '../components/district/DistrictHexMap.jsx'
import SparkChart from '../components/district/SparkChart.jsx'
import RiskBadge from '../components/common/RiskBadge.jsx'
import { DISTRICTS, RISK_COLORS, getDistrictHistory, RUN_META } from '../data/districts.js'
import { useMapStore } from '../stores/mapStore.js'

const OVERLAYS = [
  { id: 'risk',       label: 'Risk Level',   color: '#d4900a' },
  { id: 'flood',      label: 'Flood Mask',   color: '#4ab0d8' },
  { id: 'population', label: 'Population',   color: '#7dd3f0' },
]

const FILTERS = ['all','Critical','High','Medium','Low','None']

const RISK_ORDER = { Critical: 0, High: 1, Medium: 2, Low: 3, None: 4 }

function DistrictDetailPanel({ district, onClose }) {
  const hist = getDistrictHistory(district.id)
  const c = RISK_COLORS[district.risk]
  const floodedKm2 = (district.floodPct * district.area / 100).toFixed(0)
  const riskScore = Math.round(district.floodPct * 0.6 + (district.pop / 10000000) * 40)

  const radarPoints = () => {
    const metrics = [
      district.floodPct / 70,
      district.pop / 9200000,
      district.floodPct / 70 * 0.8,
      district.conf,
      Math.min(riskScore / 100, 1),
    ]
    const labels = ['Flood Extent','Population','Hazard','Confidence','Risk Score']
    const cx = 80, cy = 80, r = 60
    const pts = metrics.map((v, i) => {
      const angle = (i / 5) * Math.PI * 2 - Math.PI / 2
      return { x: cx + r * v * Math.cos(angle), y: cy + r * v * Math.sin(angle), label: labels[i], v }
    })
    const polyPts = pts.map(p => `${p.x},${p.y}`).join(' ')
    return { pts, polyPts, cx, cy, r, labels, metrics }
  }

  const { pts, polyPts, cx, cy, r } = radarPoints()

  return (
    <motion.div
      initial={{ opacity: 0, x: 30 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: 30 }}
      transition={{ duration: 0.3 }}
      className="bg-bg-card rounded-2xl glow-border overflow-hidden flex flex-col"
    >
      {/* Header */}
      <div className="px-4 sm:px-5 py-3 sm:py-4 border-b border-white/5 flex items-start justify-between"
           style={{ background: `linear-gradient(135deg,${c.dim},transparent)` }}>
        <div>
          <div className="text-[10px] sm:text-xs font-mono text-text-3 mb-1">{district.id}</div>
          <h3 className="font-display font-bold text-lg sm:text-xl text-text">{district.name}</h3>
          <div className="flex items-center gap-2 mt-2">
            <RiskBadge risk={district.risk} size="sm" />
            <span className="text-[10px] sm:text-xs font-mono text-text-2">conf {(district.conf * 100).toFixed(0)}%</span>
          </div>
        </div>
        <button
          onClick={onClose}
          className="w-7 h-7 flex items-center justify-center rounded-lg
                     bg-white/5 hover:bg-white/10 text-text-2 hover:text-text transition-colors text-sm flex-shrink-0"
        >
          ✕
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 sm:p-5 space-y-4 sm:space-y-5">

        {/* Key stats */}
        <div className="grid grid-cols-2 gap-2 sm:gap-3">
          {[
            { label: 'Flood Coverage', val: `${district.floodPct}%`, color: c.hex },
            { label: 'Flooded Area',   val: `${floodedKm2} km²`,    color: c.hex },
            { label: 'Pop. Exposed',   val: `${(district.pop * district.floodPct / 100 / 1000).toFixed(0)}k`, color: '#4ab0d8' },
            { label: 'Total Area',     val: `${district.area} km²`, color: 'rgba(191,207,216,0.6)' },
          ].map(s => (
            <div key={s.label} className="bg-bg rounded-xl p-2.5 sm:p-3 border border-white/5">
              <div className="font-display font-bold text-base sm:text-lg" style={{ color: s.color }}>{s.val}</div>
              <div className="text-[9px] sm:text-[10px] font-mono text-text-3">{s.label}</div>
            </div>
          ))}
        </div>

        {/* Radar chart */}
        <div>
          <div className="text-[10px] sm:text-xs font-mono text-text-2 mb-2 sm:mb-3 uppercase tracking-wider">
            Risk Profile
          </div>
          <div className="flex justify-center">
            <svg viewBox="0 0 160 160" className="w-[140px] h-[140px] sm:w-[160px] sm:h-[160px]">
              {/* Grid rings */}
              {[0.25,0.5,0.75,1].map(t => (
                <polygon key={t}
                  points={[0,1,2,3,4].map(i => {
                    const angle = (i / 5) * Math.PI * 2 - Math.PI / 2
                    return `${cx + r * t * Math.cos(angle)},${cy + r * t * Math.sin(angle)}`
                  }).join(' ')}
                  fill="none"
                  stroke="rgba(255,255,255,0.05)"
                  strokeWidth="1"
                />
              ))}
              {/* Axis lines */}
              {pts.map((p, i) => (
                <line key={i} x1={cx} y1={cy} x2={cx + r * Math.cos((i/5)*Math.PI*2-Math.PI/2)}
                      y2={cy + r * Math.sin((i/5)*Math.PI*2-Math.PI/2)}
                      stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
              ))}
              {/* Data polygon */}
              <polygon points={polyPts}
                fill={c.hex + '25'}
                stroke={c.hex}
                strokeWidth="1.5"
              />
              {/* Data points */}
              {pts.map((p, i) => (
                <circle key={i} cx={p.x} cy={p.y} r="3"
                  fill={c.hex} stroke="#07090e" strokeWidth="1.5" />
              ))}
              {/* Labels */}
              {pts.map((p, i) => {
                const angle = (i / 5) * Math.PI * 2 - Math.PI / 2
                const lx = cx + (r + 16) * Math.cos(angle)
                const ly = cy + (r + 16) * Math.sin(angle)
                return (
                  <text key={i} x={lx} y={ly}
                    textAnchor="middle" dominantBaseline="middle"
                    fontSize="6.5" fontFamily="JetBrains Mono"
                    fill="rgba(140,165,180,0.5)">
                    {['Flood','Pop','Hazard','Conf','Risk'][i]}
                  </text>
                )
              })}
            </svg>
          </div>
        </div>

        {/* Flood time series */}
        <div>
          <div className="text-[10px] sm:text-xs font-mono text-text-2 mb-2 sm:mb-3 uppercase tracking-wider">
            Seasonal Flood History (2024)
          </div>
          <SparkChart data={hist} color={c.hex} height={100} />
        </div>

        {/* Quality flags */}
        <div className="bg-bg rounded-xl p-2.5 sm:p-3 border border-white/5">
          <div className="text-[9px] sm:text-[10px] font-mono text-text-3 mb-2 uppercase tracking-wider">
            Data Quality
          </div>
          <div className="flex flex-wrap gap-1.5 sm:gap-2">
            {[
              { label: 'Cloud cover < 20%',    ok: true  },
              { label: 'Orbit: DESCENDING',    ok: true  },
              { label: 'Scene complete',        ok: true  },
              { label: 'PostGIS validated',     ok: true  },
              { label: 'WorldPop intersected',  ok: true  },
              { label: 'JRC water subtracted',  ok: true  },
            ].map(f => (
              <span key={f.label}
                    className={`text-[9px] sm:text-[10px] px-1.5 sm:px-2 py-0.5 rounded font-mono border
                      ${f.ok ? 'text-low border-low/20 bg-low/5' : 'text-critical border-critical/20 bg-critical/5'}`}>
                {f.ok ? '✓' : '✗'} {f.label}
              </span>
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export default function DistrictIntelligence() {
  const { overlay, setOverlay, filterRisk, setFilterRisk, selectedDistrict, setSelectedDistrict, clearSelected } = useMapStore()
  const [sidebarSort, setSidebarSort] = useState('flood')

  const sorted = [...DISTRICTS].sort((a, b) =>
    sidebarSort === 'flood' ? b.floodPct - a.floodPct :
    sidebarSort === 'pop'   ? b.pop - a.pop :
    RISK_ORDER[a.risk] - RISK_ORDER[b.risk]
  )

  const critCount = DISTRICTS.filter(d => d.risk === 'Critical').length
  const highCount = DISTRICTS.filter(d => d.risk === 'High').length
  const totExp    = DISTRICTS.reduce((s, d) => s + d.pop * d.floodPct / 100, 0)

  return (
    <div className="min-h-screen bg-bg noise pt-14 sm:pt-16 flex flex-col">
      <div className="flex-1 flex flex-col max-w-[1600px] mx-auto w-full px-3 sm:px-4 py-3 sm:py-4 gap-3 sm:gap-4">

        {/* Header strip */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <div className="section-tag mb-2">District Intelligence</div>
            <h1 className="font-display font-extrabold text-xl sm:text-3xl text-text">
              Bangladesh Flood Risk Map
            </h1>
          </div>
          <div className="flex items-center gap-4 sm:gap-6 text-xs font-mono">
            {[
              { label: 'Critical', val: critCount, color: '#d84040' },
              { label: 'High',     val: highCount, color: '#d06828' },
              { label: 'Exposed',  val: `${(totExp / 1000000).toFixed(1)}M`, color: '#4ab0d8' },
            ].map(s => (
              <div key={s.label} className="text-center">
                <div className="font-display font-bold text-lg sm:text-xl" style={{ color: s.color }}>{s.val}</div>
                <div className="text-text-3 text-[10px] sm:text-xs">{s.label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Toolbar */}
        <div className="flex items-center gap-2 sm:gap-4 flex-wrap">
          {/* Overlay toggle */}
          <div className="flex gap-0.5 sm:gap-1 bg-bg-2 rounded-xl p-0.5 sm:p-1 border border-white/5">
            {OVERLAYS.map(o => (
              <button key={o.id}
                onClick={() => setOverlay(o.id)}
                className={`px-2 sm:px-3 py-1 sm:py-1.5 rounded-lg text-[10px] sm:text-xs font-medium transition-all
                  ${overlay === o.id ? 'text-bg font-bold' : 'text-text-2 hover:text-text'}`}
                style={overlay === o.id ? { background: o.color } : undefined}
              >
                {o.label}
              </button>
            ))}
          </div>

          <div className="w-px h-5 sm:h-6 bg-white/10 hidden sm:block" />

          {/* Risk filter */}
          <div className="flex gap-0.5 sm:gap-1 flex-wrap">
            {FILTERS.map(f => (
              <button key={f}
                onClick={() => setFilterRisk(f)}
                className={`px-2 sm:px-3 py-1 sm:py-1.5 rounded-lg text-[10px] sm:text-xs font-mono transition-all border
                  ${filterRisk === f
                    ? 'border-gold/30 bg-gold/10 text-gold-lt'
                    : 'border-white/5 text-text-3 hover:text-text hover:border-white/10'}`}
              >
                {f === 'all' ? 'All' : f}
              </button>
            ))}
          </div>

          <div className="ml-auto text-[10px] sm:text-xs font-mono text-text-3 hidden sm:block">
            {RUN_META.runId} · {RUN_META.sceneDate}
          </div>
        </div>

        {/* Main content: mobile stacks vertically, desktop uses 12-col grid */}
        <div className="flex-1 flex flex-col lg:grid lg:grid-cols-12 gap-3 sm:gap-4"
             style={{ minHeight: 'min(500px, 60vh)' }}>

          {/* Hex map */}
          <div className="col-span-12 lg:col-span-6 xl:col-span-7 bg-bg-card rounded-2xl
                          glow-border overflow-hidden relative min-h-[300px] sm:min-h-[400px] lg:min-h-0">
            <div className="absolute inset-0 p-2 sm:p-4">
              <DistrictHexMap onSelect={(d) => setSelectedDistrict(d)} />
            </div>

            {/* Legend */}
            <div className="absolute bottom-2 sm:bottom-3 left-2 sm:left-3 bg-bg/80 backdrop-blur-sm
                            rounded-xl p-2 sm:p-3 border border-white/5 space-y-1 sm:space-y-1.5">
              <div className="text-[9px] sm:text-[10px] font-mono text-text-3 uppercase tracking-wider mb-1 sm:mb-2">
                {overlay === 'risk' ? 'Risk Level' : overlay === 'flood' ? 'Flood %' : 'Population'}
              </div>
              {overlay === 'risk' ? (
                Object.entries(RISK_COLORS).map(([k, c]) => (
                  <div key={k} className="flex items-center gap-1.5 sm:gap-2">
                    <div className="w-2.5 sm:w-3 h-2.5 sm:h-3 rounded-sm" style={{ background: c.hex }} />
                    <span className="text-[9px] sm:text-[10px] font-mono text-text-2">{c.label}</span>
                  </div>
                ))
              ) : (
                <div className="text-[9px] sm:text-[10px] font-mono text-text-2">
                  {overlay === 'flood' ? '0% → 70%' : 'Low → High'}
                </div>
              )}
            </div>
          </div>

          {/* District sidebar list — hidden on mobile when detail panel is open */}
          <div className={`${selectedDistrict ? 'hidden xl:flex' : 'flex'}
                           col-span-12 lg:col-span-6 xl:col-span-2
                           flex-col bg-bg-card rounded-2xl glow-border overflow-hidden
                           max-h-[300px] lg:max-h-none`}>
            <div className="px-3 sm:px-4 py-2 sm:py-3 border-b border-white/5 flex items-center justify-between">
              <span className="text-[10px] sm:text-xs font-mono text-text-2">Districts</span>
              <select
                value={sidebarSort}
                onChange={e => setSidebarSort(e.target.value)}
                className="text-[9px] sm:text-[10px] font-mono text-text-2 bg-transparent border-0 outline-none cursor-pointer"
              >
                <option value="flood">↓ Flood %</option>
                <option value="pop">↓ Population</option>
                <option value="risk">↓ Risk</option>
              </select>
            </div>
            <div className="flex-1 overflow-y-auto">
              {sorted.map(d => {
                const c = RISK_COLORS[d.risk]
                const isSelected = selectedDistrict?.id === d.id
                return (
                  <button key={d.id}
                    onClick={() => setSelectedDistrict(isSelected ? null : d)}
                    className={`w-full text-left px-3 sm:px-4 py-2 sm:py-3 border-b border-white/[0.03]
                                hover:bg-white/[0.02] transition-colors
                                ${isSelected ? 'bg-gold/5 border-l-2 border-l-gold' : ''}`}
                  >
                    <div className="flex items-center justify-between mb-0.5">
                      <span className="text-[10px] sm:text-xs font-medium text-text">{d.name}</span>
                      <span className="text-[9px] sm:text-[10px] font-mono" style={{ color: c.hex }}>
                        {d.floodPct}%
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-0.5 bg-white/5 rounded-full overflow-hidden">
                        <div className="h-full rounded-full"
                             style={{ width: `${Math.min(d.floodPct, 70) / 70 * 100}%`, background: c.hex }} />
                      </div>
                      <span className="text-[8px] sm:text-[9px] font-mono text-text-3"
                            style={{ color: c.hex + 'aa' }}>
                        {d.risk.slice(0,4)}
                      </span>
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Detail panel */}
          <div className="col-span-12 lg:col-span-6 xl:col-span-3">
            <AnimatePresence mode="wait">
              {selectedDistrict ? (
                <DistrictDetailPanel
                  key={selectedDistrict.id}
                  district={selectedDistrict}
                  onClose={clearSelected}
                />
              ) : (
                <motion.div
                  key="placeholder"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="h-full bg-bg-card rounded-2xl glow-border flex flex-col
                             items-center justify-center text-center p-6 sm:p-10
                             min-h-[250px] lg:min-h-0"
                >
                  <div className="w-12 h-12 sm:w-16 sm:h-16 rounded-2xl bg-gold/10 border border-gold/15
                                  flex items-center justify-center text-2xl sm:text-3xl mb-4 sm:mb-5 animate-float">
                    🗺️
                  </div>
                  <div className="font-display font-bold text-text text-sm sm:text-base mb-2">
                    Select a District
                  </div>
                  <p className="text-xs sm:text-sm text-text-2">
                    Click any hexagon on the map or a row in the list to view
                    detailed flood risk analytics for that district.
                  </p>
                  <div className="mt-4 sm:mt-6 grid grid-cols-2 gap-2 sm:gap-3 w-full text-left">
                    {[
                      { label: 'Total Flooded', val: '18,400 km²' },
                      { label: 'Pop. Exposed',  val: '4.2M' },
                      { label: 'Avg Confidence','val': '87.4%' },
                      { label: 'Run Date',       val: '2024-11-04' },
                    ].map(s => (
                      <div key={s.label} className="bg-bg rounded-xl p-2 sm:p-3 border border-white/5">
                        <div className="font-mono font-bold text-gold-lt text-xs sm:text-sm">{s.val}</div>
                        <div className="text-[9px] sm:text-[10px] font-mono text-text-3">{s.label}</div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  )
}
