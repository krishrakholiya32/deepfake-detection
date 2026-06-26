interface Props {
  verdict: 'real' | 'fake' | null
  fakeProbability: number | null
}

export default function VerdictBadge({ verdict, fakeProbability }: Props) {
  if (!verdict) return null
  const isFake = verdict === 'fake'
  return (
    <div
      className={`inline-flex items-center gap-3 rounded-lg border px-5 py-3 ${
        isFake ? 'border-df-fake bg-df-fake/10 text-df-fake' : 'border-df-real bg-df-real/10 text-df-real'
      }`}
    >
      <span className="text-xl font-bold uppercase tracking-wide">{verdict}</span>
      {fakeProbability !== null && (
        <span className="text-sm opacity-80">{(fakeProbability * 100).toFixed(1)}% fake probability</span>
      )}
    </div>
  )
}
