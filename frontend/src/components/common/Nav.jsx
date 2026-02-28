import React, { useState } from 'react'
import { useAppStore } from '../../stores/appStore.js'

const TABS = [
  { id: 'dashboard', label: 'Dashboard',        icon: '◈' },
  { id: 'mission',   label: 'Mission Control',  icon: '⬡' },
  { id: 'map',       label: 'District Intel',   icon: '◉' },
  { id: 'api',       label: 'API Terminal',     icon: '⌘' },
]

export default function Nav() {
  const { activeTab, setActiveTab } = useAppStore()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <>
      <nav className="fixed top-0 left-0 right-0 z-50 h-14 sm:h-16 flex items-center
                      px-3 sm:px-6 bg-bg/80 backdrop-blur-md border-b border-white/5">
        {/* Logo */}
        <div className="flex items-center gap-2 sm:gap-3 mr-4 sm:mr-10">
          <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-lg flex items-center justify-center
                          bg-gold/10 border border-gold/20 shadow-gold-sm flex-shrink-0">
            <span className="text-gold text-xs sm:text-sm">🛰</span>
          </div>
          <div>
            <div className="font-display font-bold text-xs sm:text-sm tracking-wider text-text leading-none">
              COSMEON
            </div>
            <div className="text-[9px] sm:text-[10px] font-mono text-text-2 tracking-widest uppercase hidden sm:block">
              Climate Risk Intelligence
            </div>
          </div>
        </div>

        {/* Desktop Tabs — hidden on mobile */}
        <div className="hidden md:flex items-center gap-1 flex-1">
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                flex items-center gap-2 px-3 lg:px-4 py-2 rounded-lg text-sm font-medium
                transition-all duration-200
                ${activeTab === tab.id
                  ? 'bg-gold/10 text-gold-lt border border-gold/20 shadow-gold-sm'
                  : 'text-text-2 hover:text-text hover:bg-white/4'}
              `}
            >
              <span className="text-xs opacity-70">{tab.icon}</span>
              <span className="hidden lg:inline">{tab.label}</span>
              <span className="lg:hidden">{tab.label.split(' ')[0]}</span>
            </button>
          ))}
        </div>

        {/* Mobile Tabs — compact, shown on small screens */}
        <div className="flex md:hidden items-center gap-0.5 flex-1 overflow-x-auto scrollbar-none">
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => { setActiveTab(tab.id); setMobileMenuOpen(false) }}
              className={`
                flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium
                transition-all duration-200 flex-shrink-0
                ${activeTab === tab.id
                  ? 'bg-gold/10 text-gold-lt border border-gold/20'
                  : 'text-text-2'}
              `}
            >
              <span className="text-[10px] opacity-70">{tab.icon}</span>
              <span className="hidden xs:inline">{tab.label.split(' ')[0]}</span>
            </button>
          ))}
        </div>

        {/* Right side — live indicator */}
        <div className="flex items-center gap-2 text-xs font-mono text-text-2 flex-shrink-0">
          <span className="w-1.5 h-1.5 rounded-full bg-low animate-pulse-slow" />
          <span className="hidden sm:inline">LIVE · BD-DELTA-01</span>
          <span className="sm:hidden">LIVE</span>
        </div>
      </nav>

      {/* Spacer so content below nav isn't hidden — nav is fixed */}
    </>
  )
}
