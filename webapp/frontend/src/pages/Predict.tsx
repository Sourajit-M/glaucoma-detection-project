import { useState } from 'react'
import { Loader2 } from 'lucide-react'
import UploadZone   from '../components/uploadZone'
import ResultCard   from '../components/ResultCard'
import HeatmapViewer from '../components/HeatmapViewer'
import { usePrediction } from '../hooks/usePrediction'

export default function Predict() {
  const [file, setFile] = useState<File | null>(null)
  const { mutate, isPending, data, error, reset } = usePrediction()

  const handleAnalyse = () => {
    if (file) mutate(file)
  }

  const handleNewImage = () => {
    setFile(null)
    reset()
  }

  return (
    <main className="max-w-5xl mx-auto px-6 py-8">

      <h1 className="text-2xl font-semibold text-gray-900 mb-1">
        Glaucoma detection
      </h1>
      <p className="text-sm text-gray-500 mb-8">
        Upload a retinal fundus image to receive a prediction,
        Grad-CAM heatmap, and cup-to-disc ratio.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Left — upload */}
        <div className="space-y-4">
          <UploadZone onFile={setFile} disabled={isPending} />

          {error && (
            <div className="text-sm text-red-700 bg-red-50 border border-red-200
              rounded-lg px-4 py-3">
              {(error as any).response?.data?.detail ?? 'Something went wrong. Please try again.'}
            </div>
          )}

          <div className="flex gap-3">
            <button
              onClick={handleAnalyse}
              disabled={!file || isPending}
              className="flex-1 flex items-center justify-center gap-2
                bg-blue-600 hover:bg-blue-700 disabled:bg-gray-200
                disabled:text-gray-400 text-white text-sm font-medium
                px-4 py-2.5 rounded-lg transition-colors"
            >
              {isPending && <Loader2 size={16} className="animate-spin" />}
              {isPending ? 'Analysing…' : 'Analyse image'}
            </button>

            {data && (
              <button
                onClick={handleNewImage}
                className="px-4 py-2.5 text-sm text-gray-600 border
                  border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              >
                New image
              </button>
            )}
          </div>
        </div>

        {/* Right — results */}
        <div className="space-y-4">
          {data ? (
            <>
              <ResultCard result={data} />
              <HeatmapViewer images={data.images} />
            </>
          ) : (
            <div className="h-64 flex items-center justify-center
              border-2 border-dashed border-gray-200 rounded-xl">
              <p className="text-sm text-gray-400">
                Results will appear here after analysis
              </p>
            </div>
          )}
        </div>

      </div>
    </main>
  )
}