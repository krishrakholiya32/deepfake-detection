import { useState } from 'react'
import { detectApi, type DetectionResult } from '../api/client'
import UploadDropzone from '../components/UploadDropzone'
import VerdictBadge from '../components/VerdictBadge'
import ProgressBar from '../components/ProgressBar'
import FrameConfidenceChart from '../components/FrameConfidenceChart'
import { useJobPolling } from '../hooks/useJobPolling'

export default function Upload() {
  const [initialResult, setInitialResult] = useState<DetectionResult | null>(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const result = useJobPolling(initialResult)

  async function handleFile(file: File) {
    setError(null)
    setUploading(true)
    try {
      const isVideo = file.type.startsWith('video/')
      const res = isVideo ? await detectApi.video(file) : await detectApi.image(file)
      setInitialResult(res)
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? 'Detection failed')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <h2 className="text-2xl font-semibold">Detect a deepfake</h2>
      <UploadDropzone onFile={handleFile} disabled={uploading} />

      {error && <p className="text-df-fake">{error}</p>}

      {result && result.status !== 'done' && result.status !== 'error' && (
        <ProgressBar status={result.status} />
      )}

      {result?.status === 'error' && <p className="text-df-fake">{result.error_message}</p>}

      {result?.status === 'done' && (
        <div className="space-y-4">
          <VerdictBadge verdict={result.verdict} fakeProbability={result.fake_probability} />
          {result.frame_results && <FrameConfidenceChart frameResults={result.frame_results} />}
        </div>
      )}
    </div>
  )
}
