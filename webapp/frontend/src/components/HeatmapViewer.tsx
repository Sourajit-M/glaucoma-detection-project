import { useState } from 'react'
import type { ImageSet } from '../types/api'

interface Props { images: ImageSet }

const tabs: { key: keyof ImageSet; label: string }[] = [
  { key: 'original',             label: 'ORIGINAL'      },
  { key: 'heatmap_overlay',      label: 'GRAD-CAM'      },
  { key: 'segmentation_overlay', label: 'SEGMENTATION'  },
  { key: 'disc_mask',            label: 'DISC MASK'     },
  { key: 'cup_mask',             label: 'CUP MASK'      },
]

export default function HeatmapViewer({ images }: Props) {
  const [active, setActive] = useState<keyof ImageSet>('original')

  return (
    <div className="border border-gray-200">
      <div className="flex border-b border-gray-200 overflow-x-auto">
        {tabs.map(t => (
          <button key={t.key} onClick={() => setActive(t.key)}
            style={{ fontFamily: 'var(--font-mono)' }}
            className={`px-4 py-2.5 text-xs tracking-widest whitespace-nowrap
              transition-colors
              ${active === t.key
                ? 'text-black border-b-2 border-black bg-white'
                : 'text-gray-400 hover:text-gray-700 bg-gray-50'}`}>
            {t.label}
          </button>
        ))}
      </div>
      <img
        src={`data:image/png;base64,${images[active]}`}
        alt={active}
        className="w-full object-contain bg-black"
        style={{ maxHeight: '300px' }}
      />
    </div>
  )
}