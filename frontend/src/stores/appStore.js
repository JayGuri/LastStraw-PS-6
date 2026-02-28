import { create } from 'zustand'

const storedMock = localStorage.getItem('cosmeon_mock_mode')
const initMockMode = storedMock === null ? true : storedMock !== 'false'

export const useAppStore = create((set, get) => ({
  activeTab: 'dashboard', // dashboard | mission | map | api
  setActiveTab: (tab) => set({ activeTab: tab }),

  // Global notification
  notification: null,
  showNotification: (msg, type = 'info') => {
    set({ notification: { msg, type, id: Date.now() } })
    setTimeout(() => set({ notification: null }), 4000)
  },

  // Stats counters (tick up on load)
  statsReady: false,
  setStatsReady: () => set({ statsReady: true }),

  // Mock / Live API mode
  isMockMode: initMockMode,
  toggleMockMode: () => {
    const current = get().isMockMode
    const next = !current
    localStorage.setItem('cosmeon_mock_mode', String(next))
    set({ isMockMode: next })
    get().showNotification(
      next ? 'Switched to mock data mode' : 'Connecting to live API...',
      next ? 'warning' : 'info'
    )
  },
}))
