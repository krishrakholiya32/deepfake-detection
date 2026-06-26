import { useEffect, useState } from 'react'
import { historyApi, type DetectionJobRecord } from '../api/client'
import HistoryTable from '../components/HistoryTable'

export default function History() {
  const [jobs, setJobs] = useState<DetectionJobRecord[]>([])

  useEffect(() => {
    historyApi.list({ limit: 50 }).then(setJobs)
  }, [])

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-semibold">History</h2>
      <div className="rounded-lg border border-df-border bg-df-surface p-4">
        <HistoryTable jobs={jobs} />
      </div>
    </div>
  )
}
