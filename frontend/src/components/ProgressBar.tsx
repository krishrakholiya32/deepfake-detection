interface Props {
  status: string
}

export default function ProgressBar({ status }: Props) {
  return (
    <div className="flex items-center gap-3 text-df-muted">
      <div className="h-2 w-48 overflow-hidden rounded-full bg-df-surface">
        <div className="h-full w-1/3 animate-pulse rounded-full bg-df-accent" />
      </div>
      <span className="text-sm capitalize">{status}…</span>
    </div>
  )
}
