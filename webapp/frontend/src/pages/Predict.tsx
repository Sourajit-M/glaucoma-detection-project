import { useState } from 'react'
import UploadZone    from '../components/UploadZone'
import ResultCard    from '../components/ResultCard'
import HeatmapViewer from '../components/HeatmapViewer'
import { usePrediction } from '../hooks/usePrediction'

export default function Predict() {
  const [file, setFile] = useState<File | null>(null)
  const { mutate, isPending, data, error, reset } = usePrediction()

  const handleAnalyse = () => { if (file) mutate(file) }

  const handleNewImage = () => { setFile(null); reset() }

  return (
    <main className="animate-fade-in-up">
      {/* Clinical disclaimer banner */}
      <div className="border-b border-white/40 px-6 py-3 flex items-start gap-3
        bg-blue-50/50 backdrop-blur-md text-xs text-blue-800 shadow-sm">
        <span className="mt-0.5 w-4 h-4 rounded-full bg-blue-500 text-white
          flex items-center justify-center flex-shrink-0 text-xs font-bold">i</span>
        <p style={{ fontFamily: 'var(--font-mono)' }} className="tracking-wide leading-relaxed">
          CLINICAL DISCLAIMER: THIS SOFTWARE IS INTENDED FOR RESEARCH AND PROFESSIONAL
          USE ONLY. ANALYSIS RESULTS ARE PROBABILISTIC AND MUST BE VERIFIED BY A
          BOARD-CERTIFIED OPHTHALMOLOGIST.
        </p>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-10">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">

          {/* Left — input */}
          <div className="space-y-6">
            <div className="glass-panel p-8">
              <p style={{ fontFamily: 'var(--font-mono)' }}
                className="text-xs tracking-widest text-blue-500 font-semibold mb-2">
                DIAGNOSTIC INPUT
              </p>
              <h2 className="text-xl font-light text-gray-800 mb-6">
                Upload a high-resolution fundus retinal image for automated
                cup-to-disc ratio analysis and glaucoma probability scoring.
              </h2>

              <UploadZone file={file} onFile={setFile} disabled={isPending} />

              {error && (
                <div style={{ fontFamily: 'var(--font-mono)' }}
                  className="mt-6 text-xs text-red-600 bg-red-50 border border-red-200
                    rounded-lg px-4 py-3 tracking-wide flex items-center gap-2">
                  <span className="font-bold text-red-700">ERROR:</span>
                  {(error as any).response?.data?.detail ?? 'Request failed. Please retry.'}
                </div>
              )}

              <div className="flex gap-4 mt-6">
                <button onClick={handleAnalyse}
                  disabled={!file || isPending}
                  style={{ fontFamily: 'var(--font-mono)' }}
                  className={`flex-1 rounded-xl text-white text-xs font-semibold tracking-widest px-4 py-3.5
                    transition-all duration-300 shadow-md ${!file || isPending ? 'bg-gray-300 shadow-none text-gray-500 cursor-not-allowed' : 'bg-gradient-to-r from-blue-600 to-cyan-500 hover:shadow-[0_8px_20px_rgba(59,130,246,0.3)] hover:-translate-y-0.5'}`}>
                  {isPending ? 'PROCESSING...' : 'ANALYSE IMAGE'}
                </button>

                {data && (
                  <button onClick={handleNewImage}
                    style={{ fontFamily: 'var(--font-mono)' }}
                    className="px-6 py-3.5 text-xs font-semibold tracking-widest rounded-xl bg-white border
                      border-gray-200 text-gray-600 hover:bg-gray-50 hover:border-gray-300 transition-all shadow-sm">
                    RESET
                  </button>
                )}
              </div>
            </div>

            {/* Model info cards */}
            <div className="grid grid-cols-2 gap-4">
              {[
                { label: 'MODEL VERSION', value: 'ResNet18-V1.0' },
                { label: 'SENSITIVITY',   value: '93.0% CL'      },
              ].map(m => (
                <div key={m.label}
                  className="glass-card p-5">
                  <p style={{ fontFamily: 'var(--font-mono)' }}
                    className="text-xs font-medium tracking-widest text-gray-400 mb-1">
                    {m.label}
                  </p>
                  <p style={{ fontFamily: 'var(--font-mono)' }}
                    className="text-lg font-semibold text-gray-800">
                    {m.value}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Right — results */}
          <div className="space-y-4">
            <p style={{ fontFamily: 'var(--font-mono)' }}
              className="text-xs font-semibold tracking-widest text-blue-500 pl-2">
              ANALYSIS RESULTS
            </p>

            {data ? (
              <div className="animate-fade-in-up space-y-6">
                <ResultCard result={data} />
                <HeatmapViewer images={data.images} />
              </div>
            ) : (
              <div className="glass-panel h-[32rem] flex flex-col items-center justify-center
                 gap-5 border-dashed border-gray-300 bg-white/40">
                <div className="w-16 h-16 rounded-full bg-blue-50 flex items-center justify-center animate-pulse">
                  <svg viewBox="0 0 48 48" className="w-8 h-8 text-blue-400"
                    fill="none" stroke="currentColor" strokeWidth="1.5">
                    <rect x="8" y="4" width="32" height="40" rx="3" />
                    <line x1="14" y1="14" x2="34" y2="14" />
                    <line x1="14" y1="20" x2="34" y2="20" />
                    <circle cx="24" cy="32" r="6" />
                    <line x1="24" y1="26" x2="24" y2="28" />
                  </svg>
                </div>
                <p style={{ fontFamily: 'var(--font-mono)' }}
                  className="text-xs font-medium tracking-widest text-gray-400">
                  AWAITING INPUT STREAM
                </p>
              </div>
            )}
          </div>

        </div>
      </div>
    </main>
  )
}