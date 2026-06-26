import { create } from 'zustand'
import type { DetectionResult } from '../api/client'

interface AppState {
  activeJob: DetectionResult | null
  setActiveJob: (job: DetectionResult | null) => void
}

export const useStore = create<AppState>((set) => ({
  activeJob: null,
  setActiveJob: (job) => set({ activeJob: job }),
}))
