import type { PredictionResponse } from '../types/api'

interface Props { result: PredictionResponse }

export default function ResultCard({ result }: Props) {
  const isGlaucoma = result.prediction === 'glaucoma'

  return (
    <div className="glass-panel p-6 space-y-5 relative overflow-hidden">
      <div className={`absolute top-0 right-0 w-32 h-32 rounded-full blur-3xl opacity-20 -translate-y-1/2 translate-x-1/2 ${isGlaucoma ? 'bg-red-500' : 'bg-green-500'}`}></div>

      <div className="grid grid-cols-2 gap-4">
        {/* Prediction */}
        <div className={`p-5 rounded-2xl border shadow-sm ${isGlaucoma ? 'bg-red-50/50 border-red-100' : 'bg-emerald-50/50 border-emerald-100'}`}>
          <p style={{ fontFamily: 'var(--font-mono)' }}
            className={`text-xs font-bold tracking-widest mb-2
              ${isGlaucoma ? 'text-red-500' : 'text-emerald-600'}`}>
            PREDICTION
          </p>
          <p style={{ fontFamily: 'var(--font-mono)' }}
            className={`text-2xl font-bold tracking-wider
              ${isGlaucoma ? 'text-red-600' : 'text-emerald-600'}`}>
            {result.prediction.toUpperCase()}
          </p>
        </div>

        {/* Probability */}
        <div className="bg-gradient-to-br from-white/80 to-white/40 p-5 rounded-2xl border border-white shadow-sm">
          <p style={{ fontFamily: 'var(--font-mono)' }}
            className="text-xs font-bold tracking-widest text-blue-500 mb-2">
            PROBABILITY
          </p>
          <p style={{ fontFamily: 'var(--font-mono)' }}
            className="text-2xl font-bold text-gray-800">
            {(result.probability * 100).toFixed(1)}
            <span className="text-sm font-medium text-gray-500">%</span>
          </p>
        </div>

        {/* CDR */}
        <div className="bg-gradient-to-br from-white/80 to-white/40 p-5 rounded-2xl border border-white shadow-sm">
          <p style={{ fontFamily: 'var(--font-mono)' }}
            className="text-xs font-bold tracking-widest text-purple-500 mb-2">
            CUP-DISC RATIO
          </p>
          <p style={{ fontFamily: 'var(--font-mono)' }}
            className="text-2xl font-bold text-gray-800">
            {result.cdr.toFixed(3)}
          </p>
        </div>

        {/* Confidence */}
        <div className="bg-gradient-to-br from-white/80 to-white/40 p-5 rounded-2xl border border-white shadow-sm">
          <p style={{ fontFamily: 'var(--font-mono)' }}
            className="text-xs font-bold tracking-widest text-orange-500 mb-2">
            CONFIDENCE
          </p>
          <p style={{ fontFamily: 'var(--font-mono)' }}
            className="text-2xl font-bold text-gray-800">
            {result.confidence.toUpperCase()}
          </p>
        </div>
      </div>

      {/* Semantic indicators */}
      <div className="flex items-center gap-4 py-2 border-t border-gray-100 pt-4">
        <p style={{ fontFamily: 'var(--font-mono)' }}
          className="text-xs font-bold tracking-widest text-gray-400">
          INDICATORS
        </p>
        <div className="flex gap-4">
          <span className="flex items-center gap-1.5 text-xs font-semibold text-gray-600">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]" />
            HIGH RISK
          </span>
          <span className="flex items-center gap-1.5 text-xs font-semibold text-gray-600">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
            PHYSIOLOGICAL
          </span>
          <span className="flex items-center gap-1.5 text-xs font-semibold text-gray-600">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.5)]" />
            BORDERLINE
          </span>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="bg-blue-50/50 p-4 rounded-xl border border-blue-100/50">
        <p className="text-sm text-gray-600 leading-relaxed font-medium">
          {result.clinical_note}
        </p>
      </div>

      {/* Timing */}
      <p style={{ fontFamily: 'var(--font-mono)' }}
        className="text-[10px] text-gray-400 font-semibold tracking-widest text-right">
        INFERENCE {result.processing_time_ms}ms
      </p>
    </div>
  )
}