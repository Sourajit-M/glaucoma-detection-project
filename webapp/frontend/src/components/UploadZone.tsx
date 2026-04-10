import { useCallback, useState } from 'react'
import { Upload, ImageIcon } from 'lucide-react'

interface Props {
  onFile: (file: File) => void
  disabled?: boolean
}

export default function UploadZone({ onFile, disabled }: Props) {
  const [preview, setPreview] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)

  const handleFile = useCallback((file: File) => {
    const url = URL.createObjectURL(file)
    setPreview(url)
    onFile(file)
  }, [onFile])

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }

  return (
    <label
      className={`block border-2 border-dashed rounded-xl cursor-pointer
        transition-colors overflow-hidden
        ${dragOver ? 'border-blue-400 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
    >
      <input
        type="file"
        className="hidden"
        accept=".jpg,.jpeg,.png,.bmp"
        onChange={onInputChange}
        disabled={disabled}
      />

      {preview ? (
        <img
          src={preview}
          alt="Selected fundus image"
          className="w-full h-64 object-cover"
        />
      ) : (
        <div className="flex flex-col items-center justify-center h-64 gap-3 text-gray-400">
          <ImageIcon size={40} strokeWidth={1.2} />
          <p className="text-sm font-medium text-gray-600">
            Drop a fundus image or click to browse
          </p>
          <p className="text-xs">JPG · PNG · BMP · max 10 MB</p>
        </div>
      )}
    </label>
  )
}