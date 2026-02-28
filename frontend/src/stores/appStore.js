import { create } from 'zustand'

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
}))
