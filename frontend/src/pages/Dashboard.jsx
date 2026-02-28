import React, { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import SatelliteOrbit from '../components/dashboard/SatelliteOrbit.jsx'
import StatCounter from '../components/common/StatCounter.jsx'
import RiskBadge from '../components/common/RiskBadge.jsx'
import { DISTRICTS, RUN_META, RISK_COLORS } from '../data/districts.js'
import { useAppStore } from '../stores/appStore.js'

const HERO_STATS = [
  { label: 'Affected annually',      val: 250,   suffix: 'M',   prefix: '', decimals: 0 },
  { label: 'Late detections',        val: 72,    suffix: '%',   prefix: '', decimals: 0 },
  { label: 'SAR resolution',         val: 10,    suffix: 'm',   prefix: '', decimals: 0 },
  { label: 'Detection threshold',    val: 16,    suffix: ' dB', prefix: '−', decimals: 0 },
  { label: 'Revisit cycle',          val: 6,     suffix: ' days',prefix: '', decimals: 0 },
  { label: 'Pipeline latency',       val: 30,    suffix: ' min',prefix: '<', decimals: 0 },
]

const PROBLEM_BLOCKS = [
  {
    icon: '⚠️',
    title: 'Delayed Detection',
    body: '72% of flood events are detected only after irreversible damage. Manual satellite annotation takes 24–72 hours — too slow for emergency response.',
    stat: '72%', statLabel: 'too late',
    color: '#d84040',
  },
  {
    icon: '🌊',
    title: 'Scale of Impact',
    body: '250M people are affected by floods annually. Emerging markets lack automated EO pipelines that convert raw imagery into district-level risk intelligence.',
    stat: '250M', statLabel: 'affected',
    color: '#d06828',
  },
  {
    icon: '🏛️',
    title: 'Decision Gap',
    body: 'Governments and insurers need structured, queryable outputs — km² flooded per district, exposed population, risk scores — not raw rasters.',
    stat: '0', statLabel: 'structured APIs',
    color: '#c8a018',
  },
]

const TECH_STACK = [
  { cat: 'Runtime',     items: ['Python 3.11+','FastAPI','Uvicorn','Pydantic v2','asyncpg','SQLAlchemy 2.x'] },
  { cat: 'Geospatial',  items: ['GDAL 3.8','Rasterio','GeoPandas','Shapely 2.x','PostGIS','GeoAlchemy2'] },
  { cat: 'Satellite',   items: ['earthengine-api','pystac-client','sentinelhub','boto3','planetary-computer'] },
  { cat: 'Detection',   items: ['scikit-image','opencv-headless','numpy/scipy','ONNX Runtime','PyTorch CPU'] },
  { cat: 'Infra',       items: ['Celery + Redis','PostgreSQL 16','Alembic','MinIO/S3','Docker'] },
  { cat: 'Observability',items: ['structlog','Prometheus','Sentry SDK','OpenTelemetry','Flower'] },
]

const topDistricts = [...DISTRICTS]
  .sort((a, b) => b.floodPct - a.floodPct)
  .slice(0, 8)

const fade = { hidden: { opacity: 0, y: 20 }, visible: { opacity: 1, y: 0 } }

export default function Dashboard() {
  const { setActiveTab } = useAppStore()
  const [started, setStarted] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setStarted(true), 200)
    return () => clearTimeout(t)
  }, [])

  return (
    <div className="min-h-screen bg-bg grid-bg noise">

      {/* ── HERO ─────────────────────────────────────────── */}
      <section className="relative min-h-screen flex flex-col items-center justify-center
                          pt-16 sm:pt-20 overflow-hidden px-4 sm:px-6">

        {/* Orbit canvas background */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="w-[320px] h-[320px] sm:w-[500px] sm:h-[500px] lg:w-[700px] lg:h-[700px] opacity-80">
            <SatelliteOrbit />
          </div>
        </div>

        {/* Radial vignette */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_60%_at_50%_50%,transparent_0%,var(--bg)_70%)] pointer-events-none" />

        <motion.div
          className="relative z-10 text-center px-4 sm:px-6 max-w-4xl"
          initial="hidden" animate={started ? 'visible' : 'hidden'}
          variants={{ visible: { transition: { staggerChildren: 0.1 } } }}
        >
          <motion.div variants={fade} className="section-tag mb-4 sm:mb-6 text-[10px] sm:text-xs">
            🛰 Satellite Data → Insight Engine · Bangladesh Delta
          </motion.div>

          <motion.h1 variants={fade}
            className="font-display font-extrabold text-3xl sm:text-5xl lg:text-7xl leading-none mb-4 sm:mb-6 tracking-tight"
          >
            <span className="text-text">Climate </span>
            <span className="stat-val">Risk</span>
            <br />
            <span className="text-text">Intelligence</span>
          </motion.h1>

          <motion.p variants={fade} className="text-text-2 text-sm sm:text-lg max-w-2xl mx-auto mb-6 sm:mb-10">
            End-to-end flood detection engine — from raw Sentinel-1 SAR imagery to per-district
            risk scores in under 30 minutes. No manual annotation required.
          </motion.p>

          <motion.div variants={fade} className="flex flex-col sm:flex-row flex-wrap gap-3 sm:gap-4 justify-center">
            <button
              onClick={() => setActiveTab('mission')}
              className="px-6 sm:px-7 py-3 bg-gold text-bg font-display font-bold text-sm rounded-xl
                         shadow-gold-md hover:shadow-gold-lg hover:bg-gold-lt transition-all duration-200"
            >
              Run Pipeline →
            </button>
            <button
              onClick={() => setActiveTab('map')}
              className="px-6 sm:px-7 py-3 bg-ice/10 border border-ice/20 text-ice font-medium text-sm rounded-xl
                         hover:bg-ice/15 transition-all duration-200"
            >
              Explore Districts
            </button>
          </motion.div>
        </motion.div>

        {/* Scroll indicator */}
        <div className="absolute bottom-6 sm:bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 opacity-30">
          <span className="text-[10px] sm:text-xs font-mono text-text-2">SCROLL</span>
          <div className="w-px h-6 sm:h-8 bg-gradient-to-b from-text-2 to-transparent animate-pulse" />
        </div>
      </section>

      {/* ── STAT BAR ─────────────────────────────────────── */}
      <section className="relative z-10 -mt-10 sm:-mt-16 px-4 sm:px-6 pb-10 sm:pb-16">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 sm:gap-3">
            {HERO_STATS.map((s, i) => (
              <motion.div
                key={s.label}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 + i * 0.07, duration: 0.5 }}
                className="bg-bg-card rounded-xl p-3 sm:p-4 glow-border text-center"
              >
                <div className="font-display font-bold text-xl sm:text-2xl stat-val mb-1">
                  {started && (
                    <StatCounter
                      target={s.val}
                      suffix={s.suffix}
                      prefix={s.prefix}
                      decimals={s.decimals}
                      duration={1800}
                    />
                  )}
                </div>
                <div className="text-[10px] sm:text-xs font-mono text-text-2">{s.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── PROBLEM BREAKDOWN ────────────────────────────── */}
      <section className="px-4 sm:px-6 py-10 sm:py-16">
        <div className="max-w-6xl mx-auto">
          <div className="section-tag mb-4">The Problem</div>
          <h2 className="font-display font-bold text-2xl sm:text-3xl text-text mb-6 sm:mb-10">
            Why current systems fail
          </h2>
          <div className="grid sm:grid-cols-3 gap-4 sm:gap-6">
            {PROBLEM_BLOCKS.map((b, i) => (
              <motion.div
                key={b.title}
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="bg-bg-card rounded-2xl p-5 sm:p-6 glow-border relative overflow-hidden group"
              >
                <div className="absolute top-0 left-0 right-0 h-px"
                     style={{ background: `linear-gradient(90deg,transparent,${b.color}40,transparent)` }} />
                <div className="text-2xl sm:text-3xl mb-3 sm:mb-4">{b.icon}</div>
                <h3 className="font-display font-bold text-text mb-2 sm:mb-3">{b.title}</h3>
                <p className="text-text-2 text-xs sm:text-sm leading-relaxed mb-4 sm:mb-6">{b.body}</p>
                <div className="flex items-end gap-2">
                  <span className="font-display font-extrabold text-2xl sm:text-3xl"
                        style={{ color: b.color }}>
                    {b.stat}
                  </span>
                  <span className="text-[10px] sm:text-xs font-mono text-text-2 pb-1">{b.statLabel}</span>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── LIVE RUN SNAPSHOT ────────────────────────────── */}
      <section className="px-4 sm:px-6 py-10 sm:py-16 bg-bg-2">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-6 sm:mb-8 gap-4">
            <div>
              <div className="section-tag mb-3">Latest Run</div>
              <h2 className="font-display font-bold text-2xl sm:text-3xl text-text">
                Bangladesh · Nov 2024 Flood Event
              </h2>
            </div>
            <div className="sm:text-right text-xs font-mono text-text-2">
              <div className="text-gold font-medium mb-1">{RUN_META.runId}</div>
              <div>Scene: {RUN_META.sceneId}</div>
              <div className="mt-1 flex items-center gap-2 sm:justify-end">
                <span className="w-1.5 h-1.5 bg-low rounded-full" />
                Completed in {RUN_META.duration}m
              </div>
            </div>
          </div>

          {/* Run overview tiles */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4 mb-6 sm:mb-8">
            {[
              { label: 'Flooded Area',      val: `${(RUN_META.floodedArea/1000).toFixed(1)}k km²`, color: '#d84040' },
              { label: 'Pop. Exposed',      val: `${RUN_META.popExposed}M`,                        color: '#d06828' },
              { label: 'Avg Confidence',    val: `${RUN_META.confidence}%`,                        color: '#4ab0d8' },
              { label: 'Districts Analysed',val: `${RUN_META.districts}`,                          color: '#e8ab30' },
            ].map(tile => (
              <div key={tile.label} className="bg-bg-card rounded-xl p-3 sm:p-4 glow-border">
                <div className="font-display font-bold text-xl sm:text-2xl mb-1" style={{ color: tile.color }}>
                  {tile.val}
                </div>
                <div className="text-[10px] sm:text-xs font-mono text-text-2">{tile.label}</div>
              </div>
            ))}
          </div>

          {/* Top districts table */}
          <div className="bg-bg-card rounded-2xl glow-border overflow-hidden">
            <div className="px-4 sm:px-6 py-3 sm:py-4 border-b border-white/5 flex items-center justify-between">
              <span className="font-display font-semibold text-text text-xs sm:text-sm">High-Risk Districts</span>
              <span className="text-[10px] sm:text-xs font-mono text-text-2 hidden sm:inline">Sorted by flood coverage</span>
            </div>
            <div className="overflow-x-auto -webkit-overflow-scrolling-touch">
              <table className="w-full text-xs sm:text-sm" style={{ minWidth: 580 }}>
                <thead>
                  <tr className="border-b border-white/5">
                    {['District','Flood %','Area km²','Pop. Exposed','Risk','Confidence'].map(h => (
                      <th key={h} className="text-left px-3 sm:px-5 py-2 sm:py-3 text-[10px] sm:text-xs font-mono text-text-3 font-normal whitespace-nowrap">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {topDistricts.map((d, i) => (
                    <motion.tr
                      key={d.id}
                      initial={{ opacity: 0, x: -10 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true }}
                      transition={{ delay: i * 0.04 }}
                      className="border-b border-white/[0.03] hover:bg-white/[0.02] transition-colors"
                    >
                      <td className="px-3 sm:px-5 py-2 sm:py-3 font-medium text-text whitespace-nowrap">{d.name}</td>
                      <td className="px-3 sm:px-5 py-2 sm:py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-16 sm:w-24 h-1.5 bg-white/5 rounded-full overflow-hidden">
                            <div
                              className="h-full rounded-full"
                              style={{
                                width: `${Math.min(d.floodPct, 100)}%`,
                                background: RISK_COLORS[d.risk]?.hex,
                              }}
                            />
                          </div>
                          <span className="font-mono text-[10px] sm:text-xs" style={{ color: RISK_COLORS[d.risk]?.hex }}>
                            {d.floodPct}%
                          </span>
                        </div>
                      </td>
                      <td className="px-3 sm:px-5 py-2 sm:py-3 font-mono text-[10px] sm:text-xs text-text-2 whitespace-nowrap">
                        {(d.floodPct * d.area / 100).toFixed(0)} km²
                      </td>
                      <td className="px-3 sm:px-5 py-2 sm:py-3 font-mono text-[10px] sm:text-xs text-text-2">
                        {(d.pop / 1000).toFixed(0)}k
                      </td>
                      <td className="px-3 sm:px-5 py-2 sm:py-3"><RiskBadge risk={d.risk} size="xs" /></td>
                      <td className="px-3 sm:px-5 py-2 sm:py-3 font-mono text-[10px] sm:text-xs text-text-2">
                        {(d.conf * 100).toFixed(0)}%
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      {/* ── TECH STACK ───────────────────────────────────── */}
      <section className="px-4 sm:px-6 py-10 sm:py-16">
        <div className="max-w-6xl mx-auto">
          <div className="section-tag section-tag-ice mb-4">Tech Stack</div>
          <h2 className="font-display font-bold text-2xl sm:text-3xl text-text mb-6 sm:mb-10">
            Built on production-grade tools
          </h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
            {TECH_STACK.map((group, gi) => (
              <motion.div
                key={group.cat}
                initial={{ opacity: 0, scale: 0.96 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: gi * 0.06 }}
                className="bg-bg-card rounded-xl p-4 sm:p-5 glow-border-ice"
              >
                <div className="text-ice text-[10px] sm:text-xs font-mono font-medium mb-2 sm:mb-3 uppercase tracking-wider">
                  {group.cat}
                </div>
                <div className="flex flex-wrap gap-1.5 sm:gap-2">
                  {group.items.map(item => (
                    <span key={item}
                          className="text-[10px] sm:text-xs px-2 sm:px-2.5 py-0.5 sm:py-1 rounded-md font-mono
                                     bg-ice/5 border border-ice/10 text-text-2">
                      {item}
                    </span>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ──────────────────────────────────────────── */}
      <section className="px-4 sm:px-6 py-16 sm:py-24">
        <div className="max-w-2xl mx-auto text-center">
          <div className="font-display font-extrabold text-2xl sm:text-4xl text-text mb-3 sm:mb-4">
            Ready to run the engine?
          </div>
          <p className="text-text-2 text-sm sm:text-base mb-6 sm:mb-8">
            Ingest Sentinel-1 data, detect flood extents, score districts — all in under 30 minutes.
          </p>
          <button
            onClick={() => setActiveTab('mission')}
            className="px-8 sm:px-10 py-3 sm:py-4 bg-gold text-bg font-display font-bold rounded-xl
                       shadow-gold-lg hover:shadow-gold-lg hover:bg-gold-lt transition-all
                       text-sm sm:text-base animate-glow-pulse"
          >
            Launch Mission Control →
          </button>
        </div>
      </section>
    </div>
  )
}
