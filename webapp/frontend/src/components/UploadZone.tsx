import { useCallback, useState, useEffect } from 'react'

interface Props {
  file: File | null
  onFile: (file: File) => void
  disabled?: boolean
}

export default function UploadZone({ file, onFile, disabled }: Props) {
  const [preview, setPreview] = useState<string | null>(null)
  const [dragOver, setDragOver] = useState(false)

  useEffect(() => {
    if (!file) {
      setPreview(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const handleFile = useCallback((file: File) => {
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
      className={`block border-2 border-dashed rounded-2xl cursor-pointer
        transition-all duration-300 overflow-hidden relative group
        ${dragOver ? 'border-blue-400 bg-blue-50/50 shadow-[0_0_30px_rgba(59,130,246,0.15)] animate-pulse-border' 
                   : 'border-blue-200 hover:border-blue-400 hover:bg-white/60'}
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        glass-card`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
    >
      <input type="file" className="hidden"
        accept=".jpg,.jpeg,.png,.bmp,.tiff"
        onChange={onInputChange} disabled={disabled} />

      {preview ? (
        <div className="relative">
          <img src={preview} alt="Selected"
            className="w-full h-80 object-cover rounded-xl transition-transform duration-700 ease-in-out group-hover:scale-[1.02]" />
          <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-end">
            <p className="text-white p-4 text-sm font-medium">Click or drag to replace image</p>
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center h-80 gap-5 p-6">
          <div className="w-20 h-20 rounded-full bg-blue-50 flex items-center justify-center group-hover:scale-110 transition-transform duration-500 shadow-sm border border-blue-100">
            <svg viewBox="0 0 40 40" className="w-10 h-10 text-blue-500"
              fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="20" cy="20" r="14" />
              <circle cx="20" cy="20" r="6" />
              <line x1="20" y1="2" x2="20" y2="8" />
              <line x1="20" y1="32" x2="20" y2="38" />
              <line x1="2" y1="20" x2="8" y2="20" />
              <line x1="32" y1="20" x2="38" y2="20" />
            </svg>
          </div>
          <div className="text-center">
            <p style={{ fontFamily: 'var(--font-mono)' }}
              className="text-sm font-bold tracking-widest text-blue-600 mb-2">
              DROP A FUNDUS IMAGE HERE
            </p>
            <p className="text-sm text-gray-500 max-w-xs mx-auto">
              Supported formats: JPG, PNG, TIFF (Max 25MB)
            </p>
          </div>
        </div>
      )}
    </label>
  )
}