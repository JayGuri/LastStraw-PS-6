import { create } from 'zustand'
import { PIPELINE_STAGES } from '../data/pipelineStages.js'

const INIT_STAGES = PIPELINE_STAGES.map(s => ({ ...s, status: 'idle', logsDone: [] }))

export const usePipelineStore = create((set, get) => ({
  stages: INIT_STAGES,
  running: false,
  done: false,
  currentStageIdx: -1,
  logs: [],
  totalDuration: 0,
  elapsed: 0,

  resetPipeline: () => set({
    stages: INIT_STAGES,
    running: false,
    done: false,
    currentStageIdx: -1,
    logs: [],
    elapsed: 0,
  }),

  addLog: (log) => set(s => ({ logs: [...s.logs, { ...log, ts: Date.now(), id: Math.random() }] })),

  startPipeline: () => {
    const store = get()
    if (store.running) return

    set({ running: true, done: false, currentStageIdx: 0,
          stages: PIPELINE_STAGES.map((s, i) => ({ ...s, status: i === 0 ? 'running' : 'idle', logsDone: [] })),
          logs: [],
          elapsed: 0 })

    const total = PIPELINE_STAGES.reduce((s, p) => s + p.duration, 0)
    set({ totalDuration: total })

    // Ticker
    const tickerStart = Date.now()
    const ticker = setInterval(() => {
      set({ elapsed: Math.round((Date.now() - tickerStart) / 1000) })
    }, 1000)

    const runStage = (idx) => {
      if (idx >= PIPELINE_STAGES.length) {
        clearInterval(ticker)
        set({ running: false, done: true, currentStageIdx: -1 })
        return
      }

      const stage = PIPELINE_STAGES[idx]
      set(s => ({
        stages: s.stages.map((st, i) =>
          i === idx ? { ...st, status: 'running' } :
          i < idx   ? { ...st, status: 'done'    } : st
        ),
        currentStageIdx: idx,
      }))

      let logIdx = 0
      const logInterval = setInterval(() => {
        if (logIdx < stage.logs.length) {
          const entry = stage.logs[logIdx]
          set(s => ({
            logs: [...s.logs, { ...entry, stageId: stage.id, id: Math.random(), ts: Date.now() }],
          }))
          logIdx++
        } else {
          clearInterval(logInterval)
          set(s => ({
            stages: s.stages.map((st, i) =>
              i === idx ? { ...st, status: 'done' } : st
            ),
          }))
          setTimeout(() => runStage(idx + 1), 400)
        }
      }, stage.duration / stage.logs.length)
    }

    runStage(0)
  },
}))
