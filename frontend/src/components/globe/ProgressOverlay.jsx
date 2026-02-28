import React from 'react'
import { motion } from 'framer-motion'
import { useGlobeStore } from '../../stores/globeStore.js'

const STAGES = [
  { id: 'queued',        label: 'Queued',           icon: 'Q' },
  { id: 'preprocessing', label: 'Preprocessing',    icon: 'P' },
  { id: 'detecting',     label: 'Flood Detection',  icon: 'D' },
  { id: 'scoring',       label: 'Risk Scoring',     icon: 'R' },
]

export default function ProgressOverlay() {
  const status = useGlobeStore(s => s.status)
  const progress = useGlobeStore(s => s.progress)
  const error = useGlobeStore(s => s.error)

  const currentIdx = STAGES.findIndex(s => s.id === status)

  if (status === 'failed') {
    return (
      <div className="absolute inset-x-4 bottom-4 card-glass rounded-xl p-4
                       border border-critical/30 bg-critical/5 backdrop-blur-md z-10">
        <div className="text-sm font-display font-bold text-critical mb-1">
          Detection Failed
        </div>
        <div className="text-xs font-mono text-text-2">{error ?? 'Unknown error'}</div>
      </div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="absolute inset-x-4 bottom-4 card-glass rounded-xl p-4
                  border border-gold/20 bg-bg/80 backdrop-blur-md z-10"
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-mono text-gold-lt uppercase tracking-wider">
          Analyzing...
        </span>
        <span className="text-xs font-mono text-text-2">
          {progress}%
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-1 bg-white/5 rounded-full overflow-hidden mb-3">
        <motion.div
          className="h-full rounded-full bg-gradient-to-r from-gold/60 to-gold"
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.5 }}
        />
      </div>

      {/* Stage indicators */}
      <div className="flex items-center gap-3">
        {STAGES.map((stage, i) => {
          const isDone = i < currentIdx
          const isCurrent = i === currentIdx
          return (
            <div key={stage.id}
                 className={`flex items-center gap-1.5 text-[10px] font-mono
                   ${isDone ? 'text-low' : isCurrent ? 'text-gold-lt' : 'text-text-3'}`}>
              {isDone ? (
                <span className="w-4 h-4 rounded-full bg-low/20 flex items-center justify-center text-low text-[8px]">
                  &check;
                </span>
              ) : isCurrent ? (
                <span className="w-4 h-4 rounded-full bg-gold/20 flex items-center justify-center">
                  <span className="w-1.5 h-1.5 rounded-full bg-gold animate-pulse" />
                </span>
              ) : (
                <span className="w-4 h-4 rounded-full bg-white/5 flex items-center justify-center text-text-3 text-[8px]">
                  {stage.icon}
                </span>
              )}
              <span className="hidden sm:inline">{stage.label}</span>
            </div>
          )
        })}
      </div>
    </motion.div>
  )
}
