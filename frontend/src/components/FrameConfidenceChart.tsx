import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { FrameResults } from '../api/client'

interface Props {
  frameResults: FrameResults
}

export default function FrameConfidenceChart({ frameResults }: Props) {
  const data = frameResults.frames.map((f) => ({
    frame: f.frame_index,
    fakeProbability: Math.round(f.fake_probability * 100),
  }))

  return (
    <div className="rounded-lg border border-df-border bg-df-surface p-4">
      <h3 className="mb-2 text-sm font-semibold text-df-muted">Fake probability over sampled frames</h3>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data}>
          <XAxis dataKey="frame" stroke="#8ab4c8" fontSize={12} />
          <YAxis domain={[0, 100]} stroke="#8ab4c8" fontSize={12} />
          <Tooltip formatter={(v: number) => `${v}%`} />
          <Line type="monotone" dataKey="fakeProbability" stroke="#ff4b4b" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
      <p className="mt-2 text-xs text-df-muted">
        {frameResults.sampled_frame_count} frames analyzed · max{' '}
        {(frameResults.max_fake_probability * 100).toFixed(1)}% · {(frameResults.pct_fake_frames * 100).toFixed(0)}%
        of frames flagged fake
      </p>
    </div>
  )
}
