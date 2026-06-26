import axios from 'axios'

export const api = axios.create({
  baseURL: '/api',
})

export interface FrameScore {
  frame_index: number
  fake_probability: number
}

export interface FrameResults {
  frames: FrameScore[]
  max_fake_probability: number
  pct_fake_frames: number
  sampled_frame_count: number
}

export interface DetectionResult {
  job_id: number
  status: 'pending' | 'processing' | 'done' | 'error'
  verdict: 'real' | 'fake' | null
  fake_probability: number | null
  frame_results: FrameResults | null
  error_message: string | null
}

export interface DetectionJobRecord {
  id: number
  media_type: 'image' | 'video'
  original_filename: string
  status: string
  verdict: 'real' | 'fake' | null
  fake_probability: number | null
  error_message: string | null
  created_at: string
}

export interface DetectionJobDetail extends DetectionJobRecord {
  frame_results: FrameResults | null
}

export const detectApi = {
  image: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return api.post<DetectionResult>('/detect/image', form).then((r) => r.data)
  },
  video: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return api.post<DetectionResult>('/detect/video', form).then((r) => r.data)
  },
  jobStatus: (jobId: number) => api.get<DetectionResult>(`/detect/job/${jobId}`).then((r) => r.data),
}

export const historyApi = {
  list: (params?: { media_type?: string; verdict?: string; limit?: number; offset?: number }) =>
    api.get<DetectionJobRecord[]>('/history', { params }).then((r) => r.data),
  detail: (jobId: number) => api.get<DetectionJobDetail>(`/history/${jobId}`).then((r) => r.data),
}
