import { format } from 'date-fns'
import type { DetectionJobRecord } from '../api/client'

interface Props {
  jobs: DetectionJobRecord[]
}

export default function HistoryTable({ jobs }: Props) {
  return (
    <table className="w-full text-left text-sm">
      <thead className="text-df-muted">
        <tr className="border-b border-df-border">
          <th className="py-2">File</th>
          <th className="py-2">Type</th>
          <th className="py-2">Verdict</th>
          <th className="py-2">Fake %</th>
          <th className="py-2">Status</th>
          <th className="py-2">When</th>
        </tr>
      </thead>
      <tbody>
        {jobs.map((job) => (
          <tr key={job.id} className="border-b border-df-border/50">
            <td className="py-2">{job.original_filename}</td>
            <td className="py-2 capitalize">{job.media_type}</td>
            <td className={`py-2 font-semibold ${job.verdict === 'fake' ? 'text-df-fake' : 'text-df-real'}`}>
              {job.verdict ?? '—'}
            </td>
            <td className="py-2">
              {job.fake_probability !== null ? `${(job.fake_probability * 100).toFixed(1)}%` : '—'}
            </td>
            <td className="py-2 capitalize">{job.status}</td>
            <td className="py-2">{format(new Date(job.created_at), 'MMM d, HH:mm')}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
