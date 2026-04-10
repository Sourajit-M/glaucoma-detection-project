import type { PredictionResponse } from '../types/api'

interface Props {
  result: PredictionResponse
}

const confidenceColour = {
  high:   'bg-green-100 text-green-800',
  medium: 'bg-yellow-100 text-yellow-800',
  low:    'bg-gray-100 text-gray-700',
}

const cdrRiskColour = {
  elevated:   'bg-red-100 text-red-800',
  borderline: 'bg-yellow-100 text-yellow-800',
  normal:     'bg-green-100 text-green-800',
}

export default function ResultCard({ result }: Props) {
  const isGlaucoma = result.prediction === 'glaucoma'

  return (
    <div className="border border-gray-200 rounded-xl p-5 space-y-4">

      {/* Verdict */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">
            Prediction
          </p>
          <p className={`text-3xl font-semibold ${
            isGlaucoma ? 'text-red-600' : 'text-green-600'
          }`}>
            {result.prediction.toUpperCase()}
          </p>
        </div>
        <div className="text-right">
          <p className="text-xs text-gray-500 uppercase tracking-wide mb-1">
            Probability
          </p>
          <p className="text-3xl font-semibold text-gray-900">
            {(result.probability * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Badges */}
      <div className="flex gap-2 flex-wrap">
        <span className={`text-xs font-medium px-2.5 py-1 rounded-full
          ${confidenceColour[result.confidence]}`}>
          {result.confidence} confidence
        </span>
        <span className={`text-xs font-medium px-2.5 py-1 rounded-full
          ${cdrRiskColour[result.cdr_risk]}`}>
          CDR {result.cdr.toFixed(3)} — {result.cdr_risk}
        </span>
        <span className="text-xs font-medium px-2.5 py-1 rounded-full
          bg-gray-100 text-gray-600">
          {result.processing_time_ms} ms
        </span>
      </div>

      {/* Clinical disclaimer */}
      <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200
        rounded-lg px-3 py-2 leading-relaxed">
        {result.clinical_note}
      </p>

    </div>
  )
}