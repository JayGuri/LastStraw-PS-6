import React from 'react'
import { useAppStore } from '../../stores/appStore.js'

const TABS = [
  { id: 'dashboard', label: 'Dashboard',        icon: '◈' },
  { id: 'mission',   label: 'Mission Control',  icon: '⬡' },
  { id: 'map',       label: 'District Intel',   icon: '◉' },
  { id: 'api',       label: 'API Terminal',     icon: '⌘' },
]

export default function Nav() {
  const { activeTab, setActiveTab } = useAppStore()

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 h-16 flex items-center px-6
                    bg-bg/80 backdrop-blur-md border-b border-white/5">
      {/* Logo */}
      <div className="flex items-center gap-3 mr-10">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center
                        bg-gold/10 border border-gold/20 shadow-gold-sm">
          <span className="text-gold text-sm">🛰</span>
        </div>
        <div>
          <div className="font-display font-bold text-sm tracking-wider text-text leading-none">
            COSMEON
          </div>
          <div className="text-[10px] font-mono text-text-2 tracking-widest uppercase">
            Climate Risk Intelligence
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-1 flex-1">
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`
              flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium
              transition-all duration-200
              ${activeTab === tab.id
                ? 'bg-gold/10 text-gold-lt border border-gold/20 shadow-gold-sm'
                : 'text-text-2 hover:text-text hover:bg-white/4'}
            `}
          >
            <span className="text-xs opacity-70">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </div>

      {/* Right side — live indicator */}
      <div className="flex items-center gap-2 text-xs font-mono text-text-2">
        <span className="w-1.5 h-1.5 rounded-full bg-low animate-pulse-slow" />
        LIVE · BD-DELTA-01
      </div>
    </nav>
  )
}
