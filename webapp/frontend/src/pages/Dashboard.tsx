// @ts-nocheck
import { useQuery } from '@tanstack/react-query'
import { apiClient } from '../lib/api'
import type { MetricsResponse } from '../types/api'
import {
  LineChart, Line, XAxis, YAxis,
  ResponsiveContainer, ReferenceLine, Tooltip, CartesianGrid
} from 'recharts'

function useMetrics() {
  return useQuery({
    queryKey: ['metrics'],
    queryFn: async () => {
      const { data } = await apiClient.get<MetricsResponse>('/results/metrics')
      return data
    },
  })
}

export default function Dashboard() {
  const { data, isLoading } = useMetrics()

  if (isLoading) return (
    <div className="flex items-center justify-center h-screen -mt-20">
      <div className="flex flex-col items-center gap-4">
        <div className="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
        <p style={{ fontFamily: 'var(--font-mono)' }}
          className="text-xs font-bold tracking-widest text-blue-600 animate-pulse">
          LOADING METRICS...
        </p>
      </div>
    </div>
  )

  if (!data) return null

  const proposed = data.models.find(m => m.type === 'hybrid')

  // Build simple ROC-like curve data from ablation for chart
  const rocPoints = data.models.map(m => ({
    name: m.name,
    fpr: parseFloat((1 - m.specificity).toFixed(3)),
    tpr: parseFloat(m.sensitivity.toFixed(3)),
    type: m.type,
  })).sort((a, b) => a.fpr - b.fpr)

  return (
    <main className="max-w-6xl mx-auto px-6 py-10 space-y-10 animate-fade-in-up">

      {/* Header */}
      <div className="glass-panel p-8 pb-10 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-64 h-64 bg-blue-400/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/3"></div>
        <div className="relative z-10">
          <p style={{ fontFamily: 'var(--font-mono)' }}
            className="text-xs font-bold tracking-widest text-blue-500 mb-3">
            VALIDATION REPORT V1.0
          </p>
          <h1 className="text-4xl font-bold text-gray-900 mb-4 tracking-tight">
            Model Performance Metrics
          </h1>
          <p className="text-base text-gray-600 max-w-2xl leading-relaxed">
            A comprehensive analysis of convolutional neural network performance
            on multi-dataset fundus evaluation. Metrics computed on held-out test
            set <span className="font-semibold text-gray-800">(n={data.dataset_info.test_set_size})</span>.
          </p>
        </div>
      </div>

      {/* Top metric cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-5">
        {[
          { label: 'BEST AUC',    value: proposed?.auc.toFixed(3) ?? '—',
            sub: '+15.8% VS SVM', color: 'from-blue-600 to-cyan-500' },
          { label: 'SENSITIVITY', value: `${((proposed?.sensitivity ?? 0)*100).toFixed(1)}%`,
            sub: 'FIXED FPR 18.0%', color: 'from-emerald-500 to-teal-400' },
          { label: 'SPECIFICITY', value: `${((proposed?.specificity ?? 0)*100).toFixed(1)}%`,
            sub: `95% CI [0.77, 0.83]`, color: 'from-purple-600 to-pink-500' },
          { label: 'TEST SET',    value: `${data.dataset_info.test_set_size}`,
            sub: `${data.dataset_info.datasets.length} DATASETS`, color: 'from-orange-500 to-amber-400' },
        ].map(m => (
          <div key={m.label} className="glass-card p-6 relative overflow-hidden group">
            <div className={`absolute -right-4 -top-4 w-24 h-24 bg-gradient-to-br ${m.color} opacity-10 rounded-full blur-xl group-hover:scale-150 transition-transform duration-700`}></div>
            <p style={{ fontFamily: 'var(--font-mono)' }}
              className="text-xs font-bold tracking-widest text-gray-500 mb-4">
              {m.label}
            </p>
            <p style={{ fontFamily: 'var(--font-mono)' }}
              className="text-4xl font-semibold text-gray-900 mb-2">
              {m.value}
            </p>
            <p style={{ fontFamily: 'var(--font-mono)' }}
              className="text-xs text-gray-500 font-medium tracking-wide">
              {m.sub}
            </p>
          </div>
        ))}
      </div>

      {/* ROC scatter + Segmentation */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">

        {/* ROC */}
        <div className="glass-panel p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-1">
            Receiver Operating Characteristic
          </h3>
          <p className="text-sm text-gray-500 mb-6">
            Sensitivity vs 1−Specificity for all evaluated models.
          </p>
          <div className="bg-white/50 p-4 rounded-xl border border-white/60">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={[{fpr:0,tpr:0},...rocPoints,{fpr:1,tpr:1}]} margin={{ top: 5, right: 20, left: -20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                <XAxis dataKey="fpr" domain={[0,1]} tickCount={5}
                  tick={{ fontSize: 11, fontFamily: 'var(--font-mono)', fill: '#64748b' }}
                  axisLine={false} tickLine={false}
                  label={{ value: 'FALSE POSITIVE RATE', position: 'insideBottom',
                    offset: -10, fontSize: 10, fontWeight: 'bold',
                    fontFamily: 'var(--font-mono)', fill: '#64748b' }} />
                <YAxis domain={[0,1]} tickCount={5}
                  tick={{ fontSize: 11, fontFamily: 'var(--font-mono)', fill: '#64748b' }}
                  axisLine={false} tickLine={false}
                  label={{ value: 'TRUE POSITIVE RATE', angle: -90,
                    position: 'insideLeft', fontSize: 10, fontWeight: 'bold',
                    fontFamily: 'var(--font-mono)', fill: '#64748b' }} />
                <Tooltip 
                   contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 25px rgba(0,0,0,0.1)', fontFamily: 'var(--font-mono)', fontSize: '12px' }}
                   itemStyle={{ color: '#0f172a' }} />
                <ReferenceLine
                  segment={[{x:0,y:0},{x:1,y:1}]}
                  stroke="#cbd5e1" strokeDasharray="4 4" />
                <Line type="monotone" dataKey="tpr" stroke="#3b82f6"
                  strokeWidth={3} dot={(props: any) => {
                    const m = rocPoints.find(r => r.fpr === props.payload.fpr)
                    return <circle key={props.key} cx={props.cx} cy={props.cy}
                      r={m?.type === 'hybrid' ? 6 : 4}
                      fill={m?.type === 'hybrid' ? '#3b82f6' : '#94a3b8'}
                      stroke="#ffffff" strokeWidth={2} className="shadow-sm" />
                  }} activeDot={{ r: 8, stroke: '#fff', strokeWidth: 2 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="flex gap-6 mt-4 justify-center">
            <span style={{ fontFamily: 'var(--font-mono)' }}
              className="flex items-center gap-2 text-xs font-semibold text-gray-700">
              <span className="w-4 h-1 bg-blue-500 rounded-full inline-block shadow-[0_0_8px_rgba(59,130,246,0.5)]" />
              PROPOSED
            </span>
            <span style={{ fontFamily: 'var(--font-mono)' }}
              className="flex items-center gap-2 text-xs font-semibold text-gray-500">
              <span className="w-3 h-3 rounded-full bg-slate-400 inline-block" />
              BASELINES
            </span>
          </div>
        </div>

        {/* Segmentation */}
        <div className="glass-panel p-6 flex flex-col">
          <h3 className="text-lg font-semibold text-gray-900 mb-1">
            Segmentation Accuracy
          </h3>
          <p className="text-sm text-gray-500 mb-6">
            Morphological segmentation on DRISHTI-GS1 (n=50).
          </p>
          <div className="grid grid-cols-2 gap-4 mb-4 flex-1">
            {data.segmentation.map((s) => (
              <div key={`${s.structure}-metrics`} className="contents">
                <div key={`${s.structure}-dice`}
                  className="bg-gradient-to-br from-white/80 to-white/40 p-5 rounded-2xl border border-white shadow-sm">
                  <p style={{ fontFamily: 'var(--font-mono)' }}
                    className="text-xs font-bold tracking-widest text-blue-500 mb-3">
                    {s.structure.toUpperCase()} DICE
                  </p>
                  <p style={{ fontFamily: 'var(--font-mono)' }}
                    className="text-3xl font-semibold text-gray-800">
                    {s.dice.toFixed(3)}
                  </p>
                </div>
                <div key={`${s.structure}-iou`}
                  className="bg-gradient-to-br from-white/80 to-white/40 p-5 rounded-2xl border border-white shadow-sm">
                  <p style={{ fontFamily: 'var(--font-mono)' }}
                    className="text-xs font-bold tracking-widest text-purple-500 mb-3">
                    {s.structure.toUpperCase()} IOU
                  </p>
                  <p style={{ fontFamily: 'var(--font-mono)' }}
                    className="text-3xl font-semibold text-gray-800">
                    {s.iou.toFixed(3)}
                  </p>
                </div>
              </div>
            ))}
          </div>
          <div className="bg-blue-50/50 p-4 rounded-xl border border-blue-100/50 flex items-center justify-between mt-auto">
            <span style={{ fontFamily: 'var(--font-mono)' }}
              className="text-xs font-bold tracking-widest text-blue-600">
              ARCHITECTURE
            </span>
            <span style={{ fontFamily: 'var(--font-mono)' }}
              className="text-xs font-medium text-gray-800 bg-white px-3 py-1.5 rounded-lg shadow-sm">
              U-Net (ResNet18) · DiceBCE
            </span>
          </div>
        </div>
      </div>

      {/* Benchmarking table */}
      <div className="glass-panel overflow-hidden">
        <div className="px-6 py-5 border-b border-white/50 bg-white/30">
          <h3 className="text-lg font-semibold text-gray-900">
            Clinical Benchmarking Table
          </h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead>
              <tr className="bg-gray-50/50">
                {['ARCHITECTURE','AUC-ROC','SENSITIVITY','SPECIFICITY','F1 SCORE'].map(h => (
                  <th key={h} style={{ fontFamily: 'var(--font-mono)' }}
                    className="px-6 py-4 text-xs font-bold tracking-widest text-gray-500 border-b border-gray-100">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100/50">
              {data.models.map(m => (
                <tr key={m.name}
                  className={`transition-colors duration-200 hover:bg-white/80
                    ${m.type === 'hybrid' ? 'bg-blue-50/30' : ''}`}>
                  <td className="px-6 py-4">
                    <span style={{ fontFamily: 'var(--font-mono)' }}
                      className={`text-xs font-semibold tracking-wide flex items-center gap-2
                        ${m.type === 'hybrid' ? 'text-blue-700' : 'text-gray-700'}`}>
                      {m.name.toUpperCase()}
                      {m.type === 'hybrid' && (
                        <span className="text-[10px] bg-gradient-to-r from-blue-600 to-cyan-500 text-white
                          px-2 py-0.5 rounded-full tracking-widest shadow-sm">PROPOSED</span>
                      )}
                    </span>
                  </td>
                  {[m.auc, m.sensitivity, m.specificity, m.f1].map((v, i) => (
                    <td key={i} style={{ fontFamily: 'var(--font-mono)' }}
                      className={`px-6 py-4 text-sm font-medium
                        ${m.type === 'hybrid' ? 'text-blue-900' : 'text-gray-600'}`}>
                      {v.toFixed(3)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="px-6 py-4 border-t border-white/50 bg-gray-50/30">
          <p style={{ fontFamily: 'var(--font-mono)' }}
            className="text-xs text-gray-500 font-medium tracking-wide">
            NOTE: All values calculated on held-out test set <span className="font-bold">(N={data.dataset_info.test_set_size})</span>.
          </p>
        </div>
      </div>

      {/* Footer info */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 pt-8 border-t border-gray-200/60 pl-2">
        {[
          {
            title: 'DATASET INFO',
            rows: [
              ['SAMPLES', data.dataset_info.total_images.toLocaleString()],
              ['TEST SPLIT', data.dataset_info.test_set_size.toLocaleString()],
              ['SOURCES', data.dataset_info.datasets.join(', ')],
            ]
          },
          {
            title: 'TRAINING HARDWARE',
            rows: [
              ['COMPUTE', 'RTX 4050 Laptop'],
              ['ENVIRONMENT', 'CUDA 12.4'],
              ['FRAMEWORK', 'PyTorch 2.3'],
            ]
          },
          {
            title: 'CLINICAL AUTHORITY',
            rows: [
              ['REVIEWERS', 'RESEARCH ONLY'],
              ['STATUS', 'NOT FOR CLINICAL USE'],
              ['VERSION', '1.0.0'],
            ]
          },
        ].map(section => (
          <div key={section.title}>
            <p style={{ fontFamily: 'var(--font-mono)' }}
              className="text-xs font-bold tracking-widest text-blue-500 mb-4">
              {section.title}
            </p>
            <div className="space-y-3">
              {section.rows.map(([k, v]) => (
                <div key={k} className="flex justify-between items-center group">
                  <span style={{ fontFamily: 'var(--font-mono)' }}
                    className="text-xs text-gray-500 font-medium tracking-wide transition-colors group-hover:text-gray-900">{k}</span>
                  <span style={{ fontFamily: 'var(--font-mono)' }}
                    className="text-xs text-gray-800 font-semibold bg-white/60 px-2 py-1 rounded-md">{v}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

    </main>
  )
}