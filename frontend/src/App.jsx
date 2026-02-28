import React from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import Nav from './components/common/Nav.jsx'
import Dashboard from './pages/Dashboard.jsx'
import MissionControl from './pages/MissionControl.jsx'
import DistrictIntelligence from './pages/DistrictIntelligence.jsx'
import ApiTerminal from './pages/ApiTerminal.jsx'
import { useAppStore } from './stores/appStore.js'

const PAGES = {
  dashboard: Dashboard,
  mission:   MissionControl,
  map:       DistrictIntelligence,
  api:       ApiTerminal,
}

function Notification() {
  const notification = useAppStore(s => s.notification)
  const COLORS = {
    info:    { bg: 'bg-ice/10',      border: 'border-ice/20',      text: 'text-ice'    },
    success: { bg: 'bg-low/10',      border: 'border-low/20',      text: 'text-low'    },
    warning: { bg: 'bg-medium/10',   border: 'border-medium/20',   text: 'text-medium' },
    error:   { bg: 'bg-critical/10', border: 'border-critical/20', text: 'text-critical'},
  }
  const c = COLORS[notification?.type] ?? COLORS.info

  return (
    <AnimatePresence>
      {notification && (
        <motion.div
          key={notification.id}
          initial={{ opacity: 0, y: -12, scale: 0.96 }}
          animate={{ opacity: 1, y: 0,   scale: 1 }}
          exit={{ opacity: 0, y: -8, scale: 0.96 }}
          className={`fixed top-16 sm:top-20 right-3 sm:right-5 z-[200] px-3 sm:px-4 py-2.5 sm:py-3 rounded-xl
                      border text-xs sm:text-sm font-medium max-w-[calc(100vw-24px)] sm:max-w-xs shadow-card
                      backdrop-blur-md ${c.bg} ${c.border} ${c.text}`}
        >
          {notification.msg}
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export default function App() {
  const activeTab = useAppStore(s => s.activeTab)
  const PageComponent = PAGES[activeTab] ?? Dashboard

  return (
    <div className="relative">
      {/* Scanline overlay — sits above everything, pointer-events-none */}
      <div className="scanline-effect pointer-events-none fixed inset-0 z-[9997]" />

      <Nav />
      <Notification />

      {/* Everything below the fixed nav — pt-16 clears the 64px nav bar */}
      <div className="pt-16">
        {/* Mock data banner */}
        <div className="mock-banner">
          ⚠ RUNNING ON MOCK DATA — switch to LIVE API in nav when backend is ready
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.25, ease: 'easeOut' }}
          >
            <PageComponent />
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  )
}
