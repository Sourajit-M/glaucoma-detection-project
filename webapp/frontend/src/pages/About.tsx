import { FileText, Database } from 'lucide-react'
import { SiGithub } from 'react-icons/si'

const pipeline = [
  { step: '01', title: 'Preprocessing',
    desc: 'CLAHE contrast enhancement + circular masking to isolate the retinal disc region.' },
  { step: '02', title: 'CNN classification',
    desc: 'ResNet18 pretrained on ImageNet, fine-tuned with two-stage protocol. AUC 0.945.' },
  { step: '03', title: 'Segmentation',
    desc: 'U-Net with ResNet18 encoder segments optic disc and cup. Dice 0.968 / 0.879.' },
  { step: '04', title: 'CDR fusion',
    desc: 'CNN probability + U-Net CDR fused via logistic regression meta-learner. Final AUC 0.947.' },
]

const datasets = [
  { name: 'ACRIMA',          n: '705',   task: 'Classification' },
  { name: 'RIM-ONE DL',      n: '485',   task: 'Classification' },
  { name: 'EyePACS-AIROGS',  n: '3,540', task: 'Classification' },
  { name: 'DRISHTI-GS1',     n: '50',    task: 'Segmentation'   },
]

const stack = [
  'Python 3.11', 'PyTorch 2.3', 'ONNX Runtime',
  'FastAPI', 'React 18', 'TypeScript', 'Docker',
]

export default function About() {
  return (
    <main className="max-w-4xl mx-auto px-6 py-10 space-y-12 animate-fade-in-up">

      {/* Header */}
      <div className="glass-panel p-8 relative overflow-hidden">
        <div className="absolute top-0 right-0 w-80 h-80 bg-blue-500/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2"></div>
        <div className="relative z-10">
          <p style={{ fontFamily: 'var(--font-mono)' }}
            className="text-xs font-bold tracking-widest text-blue-500 mb-3">
            SYSTEM OVERVIEW
          </p>
          <h1 className="text-4xl font-bold text-gray-900 mb-4 tracking-tight">
            GlaucomaDetect
          </h1>
          <p className="text-base text-gray-600 max-w-2xl leading-relaxed mb-8">
            An interpretable hybrid deep learning system for glaucoma detection
            from retinal fundus images. Combines <span className="font-medium text-gray-800">ResNet18</span> transfer learning,
            <span className="font-medium text-gray-800"> U-Net</span> structural segmentation, and <span className="font-medium text-gray-800">Grad-CAM</span> explainability into a
            unified inference pipeline.
          </p>
          <div className="flex gap-4 flex-wrap">
            <a href="https://github.com" target="_blank" rel="noreferrer"
              style={{ fontFamily: 'var(--font-mono)' }}
              className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-cyan-500 text-white text-xs font-bold
                tracking-widest px-5 py-3 rounded-xl shadow-md hover:shadow-[0_8px_20px_rgba(59,130,246,0.3)] hover:-translate-y-0.5 transition-all duration-300">
              <SiGithub size={14} /> GITHUB REPOSITORY
            </a>
            <a href="#"
              style={{ fontFamily: 'var(--font-mono)' }}
              className="flex items-center gap-2 bg-white/80 border border-gray-200
                text-xs font-bold tracking-widest px-5 py-3 rounded-xl
                hover:bg-white hover:border-blue-300 shadow-sm transition-all duration-300 text-gray-700">
              <FileText size={14} className="text-blue-500" /> RESEARCH PAPER
            </a>
          </div>
        </div>
      </div>

      {/* Pipeline */}
      <div>
        <p style={{ fontFamily: 'var(--font-mono)' }}
          className="text-xs font-bold tracking-widest text-blue-500 mb-6 pl-2">
          INFERENCE PIPELINE
        </p>
        <div className="glass-panel overflow-hidden">
          {pipeline.map((p, i) => (
            <div key={p.step}
              className={`flex gap-6 p-6 transition-colors hover:bg-white/60
                ${i > 0 ? 'border-t border-white/50' : ''}`}>
              <span style={{ fontFamily: 'var(--font-mono)' }}
                className="text-3xl font-light text-blue-200 flex-shrink-0 w-10 mt-1">
                {p.step}
              </span>
              <div>
                <p style={{ fontFamily: 'var(--font-mono)' }}
                  className="text-sm font-bold tracking-widest text-gray-900 mb-2">
                  {p.title.toUpperCase()}
                </p>
                <p className="text-sm text-gray-600 leading-relaxed font-medium">
                  {p.desc}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Datasets */}
      <div>
        <p style={{ fontFamily: 'var(--font-mono)' }}
          className="text-xs font-bold tracking-widest text-blue-500 mb-6 pl-2 flex items-center gap-2">
          <Database size={14} /> DATASETS
        </p>
        <div className="glass-panel overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="bg-white/50 border-b border-white/60">
                {['DATASET','IMAGES','TASK'].map(h => (
                  <th key={h} style={{ fontFamily: 'var(--font-mono)' }}
                    className="text-left px-6 py-4 text-xs font-bold tracking-widest
                      text-gray-500">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-white/40">
              {datasets.map((d, i) => (
                <tr key={d.name}
                  className="hover:bg-white/60 transition-colors">
                  <td style={{ fontFamily: 'var(--font-mono)' }}
                    className="px-6 py-4 text-xs font-bold text-gray-800 tracking-wide">
                    {d.name}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}
                    className="px-6 py-4 text-xs font-medium text-gray-600">{d.n}</td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}
                    className="px-6 py-4 text-xs font-medium text-blue-600">{d.task}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Tech stack */}
      <div>
        <p style={{ fontFamily: 'var(--font-mono)' }}
          className="text-xs font-bold tracking-widest text-blue-500 mb-6 pl-2">
          TECHNOLOGY STACK
        </p>
        <div className="flex flex-wrap gap-3">
          {stack.map(s => (
            <span key={s} style={{ fontFamily: 'var(--font-mono)' }}
              className="text-xs font-semibold tracking-wide border border-blue-200 bg-white/50 shadow-sm
                px-4 py-2 rounded-lg text-gray-700 hover:border-blue-400 hover:text-blue-600 hover:-translate-y-0.5
                transition-all duration-300">
              {s}
            </span>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-gray-200/60 pt-6 mt-8 pl-2">
        <p style={{ fontFamily: 'var(--font-mono)' }}
          className="text-xs text-gray-400 font-medium tracking-widest">
          © 2025 GLAUCOMADETECT. DISCLAIMER: FOR RESEARCH USE ONLY.
          NOT A PRIMARY DIAGNOSTIC TOOL.
        </p>
      </div>

    </main>
  )
}