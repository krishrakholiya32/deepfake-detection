import { useEffect, useRef, useState } from 'react'
import { detectApi, type DetectionResult } from '../api/client'

const POLL_INTERVAL_MS = 1500

export function useJobPolling(initial: DetectionResult | null) {
  const [result, setResult] = useState<DetectionResult | null>(initial)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    setResult(initial)
    if (!initial || initial.status === 'done' || initial.status === 'error') return

    timerRef.current = setInterval(async () => {
      const updated = await detectApi.jobStatus(initial.job_id)
      setResult(updated)
      if (updated.status === 'done' || updated.status === 'error') {
        if (timerRef.current) clearInterval(timerRef.current)
      }
    }, POLL_INTERVAL_MS)

    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initial?.job_id])

  return result
}
