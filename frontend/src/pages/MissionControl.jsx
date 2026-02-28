import React, { useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { usePipelineStore } from '../stores/pipelineStore.js'
import LogFeed from '../components/common/LogFeed.jsx'

const STAGE_STATUS_STYLE = {
  idle:    { ring: 'border-white/10',  bg: 'bg-white/3',   icon: '○', iconColor: 'text-text-3' },
  running: { ring: 'border-gold/40',   bg: 'bg-gold/5',    icon: '◈', iconColor: 'text-gold animate-spin-slow' },
  done:    { ring: 'border-low/30',    bg: 'bg-low/5',     icon: '✓', iconColor: 'text-low' },
}

function StageCard({ stage, index, isActive }) {
  const style = STAGE_STATUS_STYLE[stage.status] ?? STAGE_STATUS_STYLE.idle
  const totalLogs = stage.logs.length

  return (
    <motion.div
      layout
      animate={{
        borderColor: stage.status === 'running'
          ? ['rgba(212,144,10,0.2)', 'rgba(212,144,10,0.6)', 'rgba(212,144,10,0.2)']
          : undefined,
      }}
      transition={{ duration: 1.4, repeat: stage.status === 'running' ? Infinity : 0 }}
      className={`
        rounded-2xl border p-5 relative overflow-hidden transition-all duration-500
        ${stage.status === 'running' ? 'bg-gold/[0.04]' : stage.status === 'done' ? 'bg-low/[0.03]' : 'bg-bg-card'}
      `}
    >
      {/* Top stripe */}
      <div className={`absolute top-0 left-0 right-0 h-px transition-all duration-500
        ${stage.status === 'running' ? 'bg-gradient-to-r from-transparent via-gold to-transparent'
        : stage.status === 'done' ? 'bg-gradient-to-r from-transparent via-low to-transparent'
        : 'bg-transparent'}`}
      />

      {/* Shimmer overlay when running */}
      {stage.status === 'running' && (
        <div className="absolute inset-0 shimmer pointer-events-none" />
      )}

      <div className="relative flex items-start gap-4">
        {/* Status icon */}
        <div className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0
                         border ${style.ring} ${style.bg} transition-all duration-300`}>
          <span className={`text-lg font-bold ${style.iconColor}`}>{style.icon}</span>
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-mono text-text-3">MODULE {stage.num}</span>
            {stage.status === 'running' && (
              <span className="text-xs font-mono text-gold animate-pulse">● PROCESSING</span>
            )}
            {stage.status === 'done' && (
              <span className="text-xs font-mono text-low">✓ DONE</span>
            )}
          </div>
          <h3 className={`font-display font-bold text-base mb-2 transition-colors
            ${stage.status === 'running' ? 'text-gold-lt' : stage.status === 'done' ? 'text-text' : 'text-text-2'}`}>
            {stage.icon} {stage.name}
          </h3>
          <p className="text-xs text-text-2 leading-relaxed mb-3">{stage.description}</p>

          {/* I/O row */}
          <div className="grid grid-cols-2 gap-3 text-xs font-mono">
            <div>
              <div className="text-text-3 text-[10px] mb-1 uppercase tracking-wider">Inputs</div>
              {stage.inputs.map(inp => (
                <div key={inp} className="text-ice/60 flex items-start gap-1">
                  <span className="text-ice/30 mt-0.5">›</span>{inp}
                </div>
              ))}
            </div>
            <div>
              <div className="text-text-3 text-[10px] mb-1 uppercase tracking-wider">Outputs</div>
              {stage.outputs.map(out => (
                <div key={out} className="text-gold/60 flex items-start gap-1">
                  <span className="text-gold/30 mt-0.5">›</span>{out}
                </div>
              ))}
            </div>
          </div>

          {/* Libraries */}
          <div className="flex flex-wrap gap-1.5 mt-3">
            {stage.libs.map(lib => (
              <span key={lib}
                    className="text-[10px] px-2 py-0.5 rounded font-mono
                               bg-ice/5 border border-ice/10 text-ice/50">
                {lib}
              </span>
            ))}
          </div>

          {/* Progress bar */}
          {stage.status === 'running' && (
            <motion.div className="mt-3 h-0.5 bg-white/5 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gold rounded-full"
                initial={{ width: '0%' }}
                animate={{ width: '95%' }}
                transition={{ duration: stage.duration / 1000, ease: 'linear' }}
              />
            </motion.div>
          )}
          {stage.status === 'done' && (
            <div className="mt-3 h-0.5 bg-low/30 rounded-full">
              <div className="h-full w-full bg-low rounded-full" />
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )
}

function PipelineConnector({ active, done }) {
  return (
    <div className="flex items-center justify-center my-1 h-6">
      <svg width="2" height="24" viewBox="0 0 2 24">
        <line x1="1" y1="0" x2="1" y2="24"
              stroke={done ? '#38a058' : active ? '#d4900a' : 'rgba(255,255,255,0.06)'}
              strokeWidth="2"
              strokeDasharray={active ? '4 3' : undefined}
              className={active ? 'flow-active' : undefined}
        />
      </svg>
    </div>
  )
}

export default function MissionControl() {
  const {
    stages, running, done, logs, elapsed, totalDuration,
    startPipeline, resetPipeline, currentStageIdx
  } = usePipelineStore()

  const progress = totalDuration
    ? Math.min((elapsed / (totalDuration / 1000)) * 100, 100)
    : 0

  return (
    <div className="min-h-screen bg-bg noise pt-20">
      <div className="max-w-7xl mx-auto px-6 py-8">

        {/* Header */}
        <div className="flex items-start justify-between mb-8">
          <div>
            <div className="section-tag mb-3">Pipeline Execution</div>
            <h1 className="font-display font-extrabold text-4xl text-text">
              Mission Control
            </h1>
            <p className="text-text-2 mt-2 text-sm max-w-lg">
              Simulate the full SAR flood-detection pipeline — from satellite ingestion to
              per-district risk scores. Watch each stage execute in real time.
            </p>
          </div>

          <div className="flex items-center gap-3 mt-1">
            {!running && !done && (
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.97 }}
                onClick={startPipeline}
                className="flex items-center gap-2 px-6 py-3 bg-gold text-bg font-display font-bold
                           text-sm rounded-xl shadow-gold-md hover:bg-gold-lt transition-all"
              >
                <span>▶</span> Run Pipeline
              </motion.button>
            )}
            {(running || done) && (
              <button
                onClick={resetPipeline}
                className="px-5 py-3 border border-white/10 text-text-2 text-sm rounded-xl
                           hover:border-white/20 hover:text-text transition-all font-medium"
              >
                ↺ Reset
              </button>
            )}

            {/* Status chip */}
            <div className={`px-4 py-2 rounded-xl text-xs font-mono border
              ${running ? 'bg-gold/10 border-gold/25 text-gold'
              : done    ? 'bg-low/10 border-low/25 text-low'
              :           'bg-white/3 border-white/8 text-text-3'}`}>
              {running ? `⬡ RUNNING · ${elapsed}s`
               : done  ? `✓ COMPLETED · ${elapsed}s`
               :         '○ IDLE'}
            </div>
          </div>
        </div>

        {/* Global progress bar */}
        {(running || done) && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
            <div className="flex items-center justify-between text-xs font-mono text-text-2 mb-2">
              <span>BD-DELTA-01 · S1_GRD · UNet detector</span>
              <span>{progress.toFixed(0)}%</span>
            </div>
            <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
              <motion.div
                className="h-full rounded-full"
                style={{ background: done ? '#38a058' : 'linear-gradient(90deg,#d4900a,#e8ab30)' }}
                animate={{ width: `${progress}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
          </motion.div>
        )}

        <div className="grid lg:grid-cols-3 gap-6">

          {/* Left — stage pipeline */}
          <div className="lg:col-span-2">
            <div className="space-y-0">
              {stages.map((stage, i) => (
                <div key={stage.id}>
                  <StageCard
                    stage={stage}
                    index={i}
                    isActive={currentStageIdx === i}
                  />
                  {i < stages.length - 1 && (
                    <PipelineConnector
                      active={currentStageIdx === i && running}
                      done={stage.status === 'done'}
                    />
                  )}
                </div>
              ))}
            </div>

            {/* Completion card */}
            <AnimatePresence>
              {done && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="mt-4 rounded-2xl p-6 bg-low/5 border border-low/20 text-center"
                >
                  <div className="text-4xl mb-3">✅</div>
                  <div className="font-display font-bold text-xl text-low mb-2">
                    Pipeline Completed
                  </div>
                  <div className="text-sm text-text-2 font-mono">
                    RUN-20241105-BD-001 · {elapsed}s · 18,400 km² flood detected
                  </div>
                  <div className="grid grid-cols-3 gap-4 mt-5">
                    {[
                      { l: 'Districts', v: '64' },
                      { l: 'Flooded km²', v: '18,400' },
                      { l: 'Confidence', v: '87.4%' },
                    ].map(s => (
                      <div key={s.l} className="bg-low/5 rounded-xl p-3">
                        <div className="font-display font-bold text-low text-xl">{s.v}</div>
                        <div className="text-xs font-mono text-text-2">{s.l}</div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Right — log feed + metadata */}
          <div className="space-y-5">
            {/* Live log terminal */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <div className="section-tag-ice section-tag text-xs">⟨/⟩ Pipeline Logs</div>
                <span className="text-xs font-mono text-text-3">{logs.length} entries</span>
              </div>
              <LogFeed logs={logs} maxHeight={380} />
            </div>

            {/* Run metadata */}
            <div className="bg-bg-card rounded-xl p-5 glow-border">
              <div className="text-xs font-mono text-gold mb-3 uppercase tracking-wider">
                Run Metadata
              </div>
              {[
                { k: 'run_id',    v: 'RUN-20241105-BD-001' },
                { k: 'scene_id',  v: 'S1_GRD_20241104T001023' },
                { k: 'aoi_id',    v: 'BD-DELTA-01' },
                { k: 'sensor',    v: 'Sentinel-1 GRD' },
                { k: 'detector',  v: 'UNet (Sen1Floods11)' },
                { k: 'threshold', v: '−16.0 dB' },
                { k: 'crs',       v: 'EPSG:32645 → 4326' },
                { k: 'tile_size', v: '512 × 512 px' },
              ].map(({ k, v }) => (
                <div key={k} className="flex justify-between py-1.5 border-b border-white/[0.03] last:border-0">
                  <span className="text-xs font-mono text-text-3">{k}</span>
                  <span className="text-xs font-mono text-text-2">{v}</span>
                </div>
              ))}
            </div>

            {/* Stage timing */}
            <div className="bg-bg-card rounded-xl p-5 glow-border">
              <div className="text-xs font-mono text-ice mb-3 uppercase tracking-wider">
                Stage Timing
              </div>
              {stages.map(s => (
                <div key={s.id} className="flex items-center gap-3 py-1.5 border-b border-white/[0.03] last:border-0">
                  <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                    s.status === 'done'    ? 'bg-low' :
                    s.status === 'running' ? 'bg-gold animate-pulse' : 'bg-white/10'
                  }`} />
                  <span className="text-xs text-text-2 flex-1">{s.name}</span>
                  <span className={`text-xs font-mono ${
                    s.status === 'done' ? 'text-low' : s.status === 'running' ? 'text-gold' : 'text-text-3'
                  }`}>
                    {s.status === 'done' ? `${(s.duration / 1000).toFixed(0)}s` :
                     s.status === 'running' ? '...' : '--'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
