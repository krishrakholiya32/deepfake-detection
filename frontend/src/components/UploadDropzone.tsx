import { useRef, useState } from 'react'
import { UploadCloud } from 'lucide-react'

interface Props {
  onFile: (file: File) => void
  disabled?: boolean
}

export default function UploadDropzone({ onFile, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)

  return (
    <div
      onClick={() => !disabled && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setDragOver(true)
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragOver(false)
        const file = e.dataTransfer.files[0]
        if (file) onFile(file)
      }}
      className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed p-12 transition-colors ${
        dragOver ? 'border-df-accent bg-df-accent/5' : 'border-df-border'
      } ${disabled ? 'pointer-events-none opacity-50' : ''}`}
    >
      <UploadCloud size={36} className="text-df-muted" />
      <p className="text-df-muted">Drop an image or video, or click to browse</p>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,video/mp4,video/quicktime,video/webm"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) onFile(file)
          e.target.value = ''
        }}
      />
    </div>
  )
}
