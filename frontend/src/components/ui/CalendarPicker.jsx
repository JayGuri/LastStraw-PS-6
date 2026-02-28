import React, { useState, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

/**
 * Google Calendar–style date picker: month grid, today highlight, clean typography.
 * Returns date as YYYY-MM-DD via onChange.
 */
export default function CalendarPicker({ value, onChange, label = 'Date' }) {
  const [open, setOpen] = useState(false)
  const [viewDate, setViewDate] = useState(() => {
    if (value) {
      const [y, m] = value.split('-').map(Number)
      return new Date(y, m - 1, 1)
    }
    return new Date(new Date().getFullYear(), new Date().getMonth(), 1)
  })

  const displayLabel = value
    ? (() => {
        const d = new Date(value)
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
      })()
    : 'Select date'

  const { year, month, weeks } = useMemo(() => {
    const y = viewDate.getFullYear()
    const m = viewDate.getMonth()
    const first = new Date(y, m, 1)
    const last = new Date(y, m + 1, 0)
    const startPad = first.getDay()
    const daysInMonth = last.getDate()
    const totalCells = startPad + daysInMonth
    const rows = Math.ceil(totalCells / 7)
    const weeks = []
    let day = 1
    for (let r = 0; r < rows; r++) {
      const week = []
      for (let c = 0; c < 7; c++) {
        const i = r * 7 + c
        if (i < startPad || day > daysInMonth) {
          week.push(null)
        } else {
          week.push(day++)
        }
      }
      weeks.push(week)
    }
    return {
      year: y,
      month: m,
      monthName: viewDate.toLocaleDateString('en-US', { month: 'long' }),
      weeks,
    }
  }, [viewDate])

  const today = useMemo(() => {
    const t = new Date()
    return t.getFullYear() === year && t.getMonth() === month ? t.getDate() : null
  }, [year, month])

  const selectDay = (day) => {
    if (!day) return
    const yyyy = year
    const mm = String(month + 1).padStart(2, '0')
    const dd = String(day).padStart(2, '0')
    onChange(`${yyyy}-${mm}-${dd}`)
    setOpen(false)
  }

  const prevMonth = () => setViewDate(new Date(year, month - 1, 1))
  const nextMonth = () => setViewDate(new Date(year, month + 1, 1))

  return (
    <div className="relative">
      {label && (
        <label className="text-[10px] font-mono text-text-3 mb-1.5 block uppercase tracking-wider">
          {label}
        </label>
      )}
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between gap-2
                   bg-bg-2 border border-white/10 rounded-xl px-3.5 py-2.5
                   text-sm text-text font-body
                   focus:border-gold/40 focus:ring-1 focus:ring-gold/10 focus:outline-none
                   transition-all hover:border-white/15"
      >
        <span className={value ? 'text-text' : 'text-text-3'}>{displayLabel}</span>
        <span className="text-text-3">▾</span>
      </button>

      <AnimatePresence>
        {open && (
          <>
            <div
              className="fixed inset-0 z-40"
              aria-hidden="true"
              onClick={() => setOpen(false)}
            />
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.15 }}
              className="absolute z-50 top-full left-0 mt-1.5 w-[280px]
                         bg-[#1a1d26] border border-white/10 rounded-xl shadow-xl
                         overflow-hidden"
            >
              {/* Month header */}
              <div className="flex items-center justify-between px-3 py-2.5 border-b border-white/8">
                <button
                  type="button"
                  onClick={prevMonth}
                  className="p-1.5 rounded-lg text-text-2 hover:text-text hover:bg-white/8 transition-colors"
                  aria-label="Previous month"
                >
                  ‹
                </button>
                <span className="text-sm font-medium text-text">
                  {viewDate.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
                </span>
                <button
                  type="button"
                  onClick={nextMonth}
                  className="p-1.5 rounded-lg text-text-2 hover:text-text hover:bg-white/8 transition-colors"
                  aria-label="Next month"
                >
                  ›
                </button>
              </div>

              {/* Weekday headers */}
              <div className="grid grid-cols-7 gap-px px-2 pt-2 pb-1">
                {WEEKDAYS.map((d) => (
                  <div
                    key={d}
                    className="text-center text-[10px] font-medium text-text-3 py-1"
                  >
                    {d}
                  </div>
                ))}
              </div>

              {/* Days */}
              <div className="grid grid-cols-7 gap-0.5 px-2 pb-3">
                {weeks.flatMap((week, wi) =>
                  week.map((day, di) => {
                    const key = `${wi}-${di}-${day ?? 'e'}`
                    const isToday = day === today
                    const selected = value && day !== null && value === `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`
                    return (
                      <button
                        key={key}
                        type="button"
                        onClick={() => selectDay(day)}
                        disabled={!day}
                        className={`
                          w-8 h-8 rounded-lg text-[13px] font-normal transition-colors
                          ${!day ? 'invisible' : ''}
                          ${selected
                            ? 'bg-gold/25 text-gold-lt font-medium'
                            : isToday
                              ? 'bg-white/10 text-text hover:bg-white/15'
                              : 'text-text-2 hover:bg-white/8 hover:text-text'}
                        `}
                      >
                        {day ?? ''}
                      </button>
                    )
                  })
                )}
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </div>
  )
}
