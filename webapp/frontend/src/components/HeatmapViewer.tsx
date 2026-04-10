import { useState } from 'react'
import type { ImageSet } from '../types/api'

interface Props {
  images: ImageSet
}

const tabs = [
  { key: 'original'             as keyof ImageSet, label: 'Original'      },
  { key: 'heatmap_overlay'      as keyof ImageSet, label: 'Grad-CAM'      },
  { key: 'segmentation_overlay' as keyof ImageSet, label: 'Segmentation'  },
  { key: 'disc_mask'            as keyof ImageSet, label: 'Disc mask'     },
  { key: 'cup_mask'             as keyof ImageSet, label: 'Cup mask'      },
]

export default function HeatmapViewer({ images }: Props) {
  const [active, setActive] = useState<keyof ImageSet>('original')

  return (
    <div className="border border-gray-200 rounded-xl overflow-hidden">

      {/* Tab bar */}
      <div className="flex border-b border-gray-200 bg-gray-50 overflow-x-auto">
        {tabs.map(t => (
          <button
            key={t.key}
            onClick={() => setActive(t.key)}
            className={`px-4 py-2.5 text-xs font-medium whitespace-nowrap
              transition-colors
              ${active === t.key
                ? 'text-blue-600 border-b-2 border-blue-500 bg-white'
                : 'text-gray-500 hover:text-gray-700'}`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Image */}
      <img
        src={`data:image/png;base64,${images[active]}`}
        alt={active}
        className="w-full object-contain bg-black"
        style={{ maxHeight: '320px' }}
      />

    </div>
  )
}