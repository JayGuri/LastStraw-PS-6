import React, { useState, useRef, useEffect } from 'react'
import { useAppStore } from '../../stores/appStore.js'
import { useAuth } from '../../hooks/useAuth.js'

const TABS = [
  { id: 'landing', label: 'Home',            icon: '◉', emoji: '🏠' },
  { id: 'mission', label: 'Mission Control', icon: '⬡', emoji: '⚡' },
  { id: 'globe',   label: 'Globe Analysis',   icon: '⊕', emoji: '🌐' },
]

const SUBSCRIPTION_TIERS = {
  free:       { label: 'Free',       badge: 'bg-text-3/20 text-text-2 border-white/10', short: 'Free' },
  pro:        { label: 'Pro',        badge: 'bg-gold/15 text-gold-lt border-gold/30',   short: 'Pro' },
  enterprise: { label: 'Enterprise', badge: 'bg-ice/15 text-ice-lt border-ice/25',      short: 'Ent' },
}
const defaultTier = SUBSCRIPTION_TIERS.free

function TierBadge({ level }) {
  const tier = SUBSCRIPTION_TIERS[level] ?? defaultTier
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-[10px] font-medium border ${tier.badge}`}>
      {tier.short}
    </span>
  )
}

export default function Nav() {
  const { activeTab, setActiveTab, isMockMode, toggleMockMode } = useAppStore()
  const { user, isAuthenticated, logout } = useAuth()
  const [profileOpen, setProfileOpen] = useState(false)
  const profileRef = useRef(null)

  useEffect(() => {
    const handler = (e) => {
      if (profileRef.current && !profileRef.current.contains(e.target)) setProfileOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const tier = user?.subscription_level ?? 'free'

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 h-14 flex items-center px-4 sm:px-6
                    bg-[rgba(10,12,20,0.88)] backdrop-blur-xl border-b border-white/[0.06]">

      {/* Logo */}
      <div className="flex items-center gap-3 mr-6 sm:mr-8">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center
                        bg-gold/10 border border-gold/20">
          <span className="text-gold text-sm">🛰</span>
        </div>
        <div>
          <div className="font-display font-bold text-sm tracking-wider text-text leading-none">
            COSMEON
          </div>
          <div className="text-[10px] font-mono text-text-3 tracking-widest uppercase">
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
              flex items-center gap-2 px-3 sm:px-4 py-2 rounded-lg text-sm font-medium
              transition-all duration-200
              ${activeTab === tab.id
                ? 'bg-gold/10 text-gold-lt border border-gold/20'
                : 'text-text-2 hover:text-text hover:bg-white/[0.04]'}
            `}
          >
            <span className="md:hidden text-base leading-none">{tab.emoji}</span>
            <span className="hidden md:inline text-xs opacity-70">{tab.icon}</span>
            <span className="hidden md:inline">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Right: mock toggle + profile or Sign in */}
      <div className="flex items-center gap-2 sm:gap-3">
        <button
          onClick={toggleMockMode}
          className={`font-mono text-[10px] px-2.5 py-1 rounded-full border transition-all
            ${isMockMode
              ? 'border-medium/30 text-medium/70 bg-medium/5'
              : 'border-low/30 text-low/70 bg-low/5'}`}
        >
          {isMockMode ? 'MOCK' : 'LIVE'}
        </button>

        {isAuthenticated ? (
          <div ref={profileRef} className="relative">
            <button
              onClick={() => setProfileOpen(o => !o)}
              className="flex items-center gap-2 pl-2 pr-2.5 py-1.5 rounded-xl
                         border border-white/10 hover:border-white/15 hover:bg-white/[0.04]
                         transition-all"
            >
              <div className="w-8 h-8 rounded-lg bg-gold/15 border border-gold/20
                              flex items-center justify-center text-gold text-sm font-medium">
                {(user?.email ?? '?').charAt(0).toUpperCase()}
              </div>
              <TierBadge level={tier} />
            </button>

            {profileOpen && (
              <div className="absolute right-0 top-full mt-1.5 w-56 py-2 rounded-xl
                              bg-[#12151e] border border-white/10 shadow-xl">
                <div className="px-3 py-2 border-b border-white/8">
                  <p className="text-xs font-medium text-text truncate">{user?.email}</p>
                  <div className="mt-1">
                    <TierBadge level={tier} />
                  </div>
                </div>
                <button
                  onClick={() => { logout(); setProfileOpen(false) }}
                  className="w-full text-left px-3 py-2 text-xs text-text-2 hover:bg-white/5 hover:text-critical transition-colors"
                >
                  Sign out
                </button>
              </div>
            )}
          </div>
        ) : (
          <button
            onClick={() => setActiveTab('login')}
            className="px-3 py-1.5 rounded-lg text-sm font-medium
                       bg-gold/15 text-gold-lt border border-gold/25
                       hover:bg-gold/25 transition-all"
          >
            Sign in
          </button>
        )}
      </div>
    </nav>
  )
}
