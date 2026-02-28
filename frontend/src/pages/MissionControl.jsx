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
        card-glass rounded-2xl p-4 sm:p-5 relative overflow-hidden transition-all duration-500
        ${stage.status === 'idle'    ? 'opacity-50' : ''}
        ${stage.status === 'running' ? 'border-t-2 border-t-gold shadow-gold-sm' : ''}
        ${stage.status === 'done'    ? 'border-t-2 border-t-low' : ''}
      `}
    >
      {/* Top stripe (running / done glow) */}
      <div className={`absolute top-0 left-0 right-0 h-px transition-all duration-500
        ${stage.status === 'running' ? 'bg-gradient-to-r from-transparent via-gold to-transparent'
        : stage.status === 'done' ? 'bg-gradient-to-r from-transparent via-low to-transparent'
        : 'bg-transparent'}`}
      />

      {/* Shimmer overlay when running */}
      {stage.status === 'running' && (
        <div className="absolute inset-0 shimmer pointer-events-none" />
      )}

      <div className="relative flex items-start gap-3 sm:gap-4">
        {/* Status icon */}
        <div className={`w-8 h-8 sm:w-10 sm:h-10 rounded-xl flex items-center justify-center flex-shrink-0
                         border ${style.ring} ${style.bg} transition-all duration-300`}>
          <span className={`text-base sm:text-lg font-bold ${style.iconColor}`}>{style.icon}</span>
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span className="data-tag text-gold/60">MODULE {stage.num}</span>
            {stage.status === 'running' && (
              <span className="text-[10px] sm:text-xs font-mono text-gold animate-pulse">● PROCESSING</span>
            )}
            {stage.status === 'done' && (
              <span className="text-[10px] sm:text-xs font-mono text-low">✓ DONE</span>
            )}
          </div>
          <h3 className={`font-display font-bold text-sm sm:text-base mb-1 sm:mb-2 transition-colors
            ${stage.status === 'running' ? 'text-gold-lt' : stage.status === 'done' ? 'text-text' : 'text-text-2'}`}>
            {stage.icon} {stage.name}
          </h3>
          <p className="text-[10px] sm:text-xs text-text-2 leading-relaxed mb-2 sm:mb-3">{stage.description}</p>

          {/* I/O row */}
          <div className="grid grid-cols-2 gap-2 sm:gap-3 text-[10px] sm:text-xs font-mono">
            <div>
              <div className="data-tag text-ice/50 mb-1">Inputs</div>
              {stage.inputs.map(inp => (
                <div key={inp} className="text-ice/60 flex items-start gap-1">
                  <span className="text-ice/30 mt-0.5">›</span><span className="break-all">{inp}</span>
                </div>
              ))}
            </div>
            <div>
              <div className="data-tag text-gold/50 mb-1">Outputs</div>
              {stage.outputs.map(out => (
                <div key={out} className="text-gold/60 flex items-start gap-1">
                  <span className="text-gold/30 mt-0.5">›</span><span className="break-all">{out}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Libraries */}
          <div className="flex flex-wrap gap-1 sm:gap-1.5 mt-2 sm:mt-3">
            {stage.libs.map(lib => (
              <span key={lib}
                    className="font-mono text-[10px] px-1.5 sm:px-2 py-0.5 rounded
                               bg-ice/5 border border-ice/10 text-ice/50">
                {lib}
              </span>
            ))}
          </div>

          {/* Progress bar */}
          {stage.status === 'running' && (
            <motion.div className="mt-2 sm:mt-3 h-0.5 bg-white/5 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gold rounded-full"
                initial={{ width: '0%' }}
                animate={{ width: '95%' }}
                transition={{ duration: stage.duration / 1000, ease: 'linear' }}
              />
            </motion.div>
          )}
          {stage.status === 'done' && (
            <div className="mt-2 sm:mt-3 h-0.5 bg-low/30 rounded-full">
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
    <div className="flex items-center justify-center my-1 h-4 sm:h-6">
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

  // Format elapsed seconds → HH:MM:SS
  const elapsedHH = String(Math.floor(elapsed / 3600)).padStart(2, '0')
  const elapsedMM = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0')
  const elapsedSS = String(elapsed % 60).padStart(2, '0')
  const elapsedFormatted = `${elapsedHH}:${elapsedMM}:${elapsedSS}`

  return (
    <div className="min-h-screen pt-14 noise">
      <div className="max-w-7xl mx-auto px-3 sm:px-6 py-4 sm:py-8">

        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 mb-4 sm:mb-6">
          <div>
            <span className="data-tag text-gold/50 mb-3 block">PIPELINE EXECUTOR</span>
            <div className="section-tag mb-2 sm:mb-3">Pipeline Execution</div>
            <h1 className="font-display font-extrabold text-2xl sm:text-4xl text-gradient-gold tracking-tight">
              Mission Control
            </h1>
            <p className="font-body text-text-2 mt-1 sm:mt-2 text-xs sm:text-sm max-w-lg leading-relaxed">
              Simulate the full SAR flood-detection pipeline — from satellite ingestion to
              per-district risk scores. Watch each stage execute in real time.
            </p>
          </div>

          <div className="flex flex-col items-end gap-3">
            {/* Elapsed HH:MM:SS timer */}
            {(running || done) && (
              <div className="font-mono text-2xl text-gold/80 tabular-nums tracking-widest">
                {elapsedFormatted}
              </div>
            )}

            <div className="flex items-center gap-2 sm:gap-3">
              {!running && !done && (
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.97 }}
                  onClick={startPipeline}
                  className="flex items-center gap-2 px-4 sm:px-6 py-2.5 sm:py-3 bg-gold text-bg font-display font-bold
                             text-xs sm:text-sm rounded-xl shadow-gold-md hover:bg-gold-lt transition-all"
                >
                  <span>▶</span> Run Pipeline
                </motion.button>
              )}
              {(running || done) && (
                <button
                  onClick={resetPipeline}
                  className="px-4 sm:px-5 py-2.5 sm:py-3 border border-white/10 text-text-2 text-xs sm:text-sm rounded-xl
                             hover:border-white/20 hover:text-text transition-all font-medium"
                >
                  ↺ Reset
                </button>
              )}

              {/* Status chip */}
              <div className={`px-3 sm:px-4 py-1.5 sm:py-2 rounded-xl text-[10px] sm:text-xs font-mono border
                ${running ? 'bg-gold/10 border-gold/25 text-gold'
                : done    ? 'bg-low/10 border-low/25 text-low'
                :           'bg-white/3 border-white/8 text-text-3'}`}>
                {running ? `⬡ RUNNING · ${elapsed}s`
                 : done  ? `✓ COMPLETED · ${elapsed}s`
                 :         '○ IDLE'}
              </div>
            </div>
          </div>
        </div>

        {/* Full-width progress bar — directly below header */}
        <div className="h-[2px] rounded-full mb-6 sm:mb-8 overflow-hidden"
             style={{ background: 'rgba(255,255,255,0.04)' }}>
          <div
            className="h-full rounded-full bg-gradient-to-r from-gold/40 to-gold transition-all duration-1000"
            style={{ width: `${progress}%` }}
          />
        </div>

        <div className="grid lg:grid-cols-3 gap-4 sm:gap-6">

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
                  className="mt-4 rounded-2xl p-4 sm:p-6 border border-low/20 text-center
                             bg-gradient-to-b from-low/[0.06] to-transparent"
                >
                  <div className="text-3xl sm:text-4xl mb-2 sm:mb-3">✅</div>
                  <div className="font-display font-bold text-lg sm:text-xl text-low mb-2">
                    Pipeline Completed
                  </div>
                  <div className="text-xs sm:text-sm text-text-2 font-mono">
                    RUN-20241105-BD-001 · {elapsed}s · 18,400 km² flood detected
                  </div>
                  <div className="grid grid-cols-3 gap-2 sm:gap-4 mt-4 sm:mt-5">
                    {[
                      { l: 'Districts',   v: '64'     },
                      { l: 'Flooded km²', v: '18,400' },
                      { l: 'Confidence',  v: '87.4%'  },
                    ].map(s => (
                      <div key={s.l} className="bg-low/5 rounded-xl p-2 sm:p-3">
                        <div className="font-display font-bold text-gradient-gold text-lg sm:text-xl">{s.v}</div>
                        <div className="text-[10px] sm:text-xs font-mono text-text-2">{s.l}</div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Right — log feed + metadata */}
          <div className="space-y-4 sm:space-y-5">

            {/* Live log terminal */}
            <div className="bg-[#060810] border border-white/5 rounded-xl overflow-hidden">
              {/* macOS chrome */}
              <div className="flex items-center gap-1.5 px-4 py-3 border-b border-white/5 bg-white/[0.02]">
                <div className="w-3 h-3 rounded-full bg-[#ff5f57]" />
                <div className="w-3 h-3 rounded-full bg-[#febc2e]" />
                <div className="w-3 h-3 rounded-full bg-[#28c840]" />
                <span className="ml-auto font-mono text-[10px] text-text-3 tracking-widest uppercase">
                  Pipeline Logs
                </span>
                <span className="font-mono text-[10px] text-text-3 ml-4">{logs.length} entries</span>
              </div>
              <LogFeed logs={logs} maxHeight={320} />
              {/* Blinking cursor when running */}
              {running && (
                <div
                  className="font-mono text-xs text-gold/60 px-3 py-1"
                  style={{ animation: 'blink 1s step-end infinite' }}
                >
                  ▋
                </div>
              )}
            </div>

            {/* Run metadata */}
            <div className="bg-bg-card rounded-xl p-4 sm:p-5 glow-border">
              <div className="text-[10px] sm:text-xs font-mono text-gold mb-2 sm:mb-3 uppercase tracking-wider">
                Run Metadata
              </div>
              {[
                { k: 'run_id',    v: 'RUN-20241105-BD-001'   },
                { k: 'scene_id',  v: 'S1_GRD_20241104T001023'},
                { k: 'aoi_id',    v: 'BD-DELTA-01'           },
                { k: 'sensor',    v: 'Sentinel-1 GRD'        },
                { k: 'detector',  v: 'UNet (Sen1Floods11)'   },
                { k: 'threshold', v: '−16.0 dB'              },
                { k: 'crs',       v: 'EPSG:32645 → 4326'     },
                { k: 'tile_size', v: '512 × 512 px'          },
              ].map(({ k, v }) => (
                <div key={k}
                     className="flex justify-between border-b border-white/[0.04] py-2 last:border-0">
                  <span className="text-[9px] sm:text-[10px] font-mono text-text-3 uppercase tracking-wider">{k}</span>
                  <span className="font-mono text-ice text-sm text-right break-all ml-4">{v}</span>
                </div>
              ))}
            </div>

            {/* Stage timing */}
            <div className="bg-bg-card rounded-xl p-4 sm:p-5 glow-border">
              <div className="text-[10px] sm:text-xs font-mono text-ice mb-2 sm:mb-3 uppercase tracking-wider">
                Stage Timing
              </div>
              {stages.map(s => (
                <div key={s.id} className="flex items-center gap-2 sm:gap-3 py-1 sm:py-1.5 border-b border-white/[0.03] last:border-0">
                  <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                    s.status === 'done'    ? 'bg-low' :
                    s.status === 'running' ? 'bg-gold animate-pulse' : 'bg-white/10'
                  }`} />
                  <span className="text-[10px] sm:text-xs text-text-2 flex-1 truncate">{s.name}</span>
                  <span className={`text-[10px] sm:text-xs font-mono flex-shrink-0 ${
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
