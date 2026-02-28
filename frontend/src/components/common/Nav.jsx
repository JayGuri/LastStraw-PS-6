import React from 'react'
import { useAppStore } from '../../stores/appStore.js'

const TABS = [
  { id: 'landing', label: 'Home',            icon: '◉', emoji: '🏠' },
  { id: 'mission', label: 'Mission Control', icon: '⬡', emoji: '⚡' },
  { id: 'globe',   label: 'Globe Analysis',   icon: '⊕', emoji: '🌐' },
]

export default function Nav() {
  const { activeTab, setActiveTab, isMockMode, toggleMockMode } = useAppStore()

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 h-16 flex items-center px-6
                    bg-[rgba(7,9,14,0.82)] backdrop-blur-xl border-b border-white/5">

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
            {/* Mobile: emoji icon only */}
            <span className="md:hidden text-base leading-none">{tab.emoji}</span>

            {/* Desktop: original symbol icon + label */}
            <span className="hidden md:inline text-xs opacity-70">{tab.icon}</span>
            <span className="hidden md:inline">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Right side — mock toggle + live indicator */}
      <div className="flex items-center gap-3">

        {/* Mock / Live API toggle */}
        <button
          onClick={toggleMockMode}
          className={`font-mono text-[10px] px-3 py-1 rounded-full border transition-all duration-300
            ${isMockMode
              ? 'border-medium/30 text-medium/70 bg-medium/5 hover:bg-medium/10'
              : 'border-low/30    text-low/70    bg-low/5    hover:bg-low/10'}`}
        >
          {isMockMode ? (
            '⚠ MOCK'
          ) : (
            <span className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-low inline-block animate-pulse" />
              LIVE API
            </span>
          )}
        </button>

        {/* Live indicator */}
        <div className="flex items-center gap-2 text-xs font-mono text-text-2">
          <span className="w-1.5 h-1.5 rounded-full bg-low animate-pulse-slow" />
          LIVE · BD-DELTA-01
        </div>
      </div>
    </nav>
  )
}
